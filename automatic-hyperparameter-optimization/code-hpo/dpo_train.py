#!/usr/bin/env python3
"""
Distributed DPO Training Script for SageMaker
Supports multi-GPU training with torchrun and handles SageMaker environment variables
"""

import os
import sys
import json
import random
import socket
import logging
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
from contextlib import contextmanager
import argparse
import json
from typing import Dict, Any

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from accelerate.utils import gather_object

# Import your data processor
from dpo_data import load_data_processor

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SageMakerEnvironment:
    """
    Handles SageMaker-specific environment variables and configuration.

    SageMaker sets various environment variables that we use for distributed training:
    - SM_HOSTS: List of all hosts in the cluster
    - SM_CURRENT_HOST: Current host name
    - SM_MODEL_DIR: Where to save the final model
    - SM_CHANNEL_*: Input data channels
    - SM_NUM_GPUS: Number of GPUs per instance
    """

    def __init__(self):
        """Initialize and parse SageMaker environment variables."""
        # Training job configuration
        self.num_gpus = int(os.environ.get('SM_NUM_GPUS', torch.cuda.device_count()))
        self.hosts = json.loads(os.environ.get('SM_HOSTS', '[]'))
        self.current_host = os.environ.get('SM_CURRENT_HOST', socket.gethostname())
        self.num_nodes = len(self.hosts) if self.hosts else 1
        self.is_master = self.current_host == self.hosts[0] if self.hosts else True

        # Paths - SageMaker specific
        self.model_dir = Path(os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
        self.output_dir = Path(os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
        self.input_dir = Path(os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
        self.code_dir = Path(os.environ.get('SM_MODULE_DIR', '/opt/ml/code'))

        # Distributed training setup
        self.world_size = int(os.environ.get('WORLD_SIZE', self.num_gpus * self.num_nodes))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Network configuration for multi-node
        if self.num_nodes > 1:
            self.master_addr = os.environ.get('MASTER_ADDR', self.hosts[0])
            self.master_port = os.environ.get('MASTER_PORT', '7777')
        else:
            self.master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
            self.master_port = os.environ.get('MASTER_PORT', '29500')

        self._setup_environment()
        self._log_environment()

    def _setup_environment(self):
        """Set up environment variables for distributed training."""
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['RANK'] = str(self.rank)

        # Set CUDA device
        torch.cuda.set_device(self.local_rank)

    def _log_environment(self):
        """Log the current environment setup for debugging."""
        if self.is_master:
            logger.info("=" * 50)
            logger.info("SageMaker Environment Configuration")
            logger.info("=" * 50)
            logger.info(f"Number of nodes: {self.num_nodes}")
            logger.info(f"Number of GPUs per node: {self.num_gpus}")
            logger.info(f"World size: {self.world_size}")
            logger.info(f"Current host: {self.current_host}")
            logger.info(f"Master address: {self.master_addr}:{self.master_port}")
            logger.info(f"Model directory: {self.model_dir}")
            logger.info("=" * 50)


class DistributedDPOTrainer:
    """
    Main trainer class that handles distributed DPO training.

    This class coordinates the entire training pipeline including:
    - Distributed process initialization
    - Model and data distribution across GPUs
    - Synchronized training with gradient accumulation
    - Proper checkpointing and model saving
    """

    def __init__(self, config: Dict[str, Any], sm_env: SageMakerEnvironment):
        """
        Initialize the distributed trainer.

        Args:
            config: Configuration dictionary (already includes hyperparameter overrides)
            sm_env: SageMaker environment configuration
        """
        self.sm_env = sm_env
        self.config = config
        self.accelerator = None

        # Log configuration if main process
        if self.sm_env.rank == 0:
            logger.info("Final configuration after hyperparameter override:")
            logger.info(json.dumps(self.config, indent=2))

        # Initialize distributed training
        self._init_distributed()

    def _update_paths_for_sagemaker(self, config: Dict[str, Any]):
        """Update configuration paths to use SageMaker directories."""
        if 'training' in config:
            config['training']['output_dir'] = str(self.sm_env.output_dir / 'checkpoints')
            config['training']['final_model_dir'] = str(self.sm_env.model_dir)

    def _init_distributed(self):
        """
        Initialize distributed training using PyTorch distributed.

        This sets up the process group for communication between GPUs
        and handles both single-node and multi-node setups.
        """
        if self.sm_env.world_size > 1:
            logger.info(f"Initializing distributed training on rank {self.sm_env.rank}")

            # Initialize the process group
            dist.init_process_group(
                backend='nccl',  # NCCL is optimized for GPU communication
                init_method=f'tcp://{self.sm_env.master_addr}:{self.sm_env.master_port}',
                world_size=self.sm_env.world_size,
                rank=self.sm_env.rank
            )

            # Verify initialization
            logger.info(f"Process group initialized. Backend: {dist.get_backend()}")

            # Synchronize all processes
            dist.barrier()

    @contextmanager
    def _distributed_context(self):
        """
        Context manager for distributed operations.

        Ensures proper setup and cleanup of distributed resources.
        """
        try:
            yield
        finally:
            if dist.is_initialized():
                dist.barrier()

    def setup_model_and_tokenizer(self) -> Tuple[Any, Any, LoraConfig]:
        """
        Set up model and tokenizer for distributed training.

        This function handles:
        - Loading the model with proper device placement
        - Configuring LoRA for parameter-efficient training
        - Setting up tokenizer with proper padding

        Returns:
            Tuple of (model, tokenizer, peft_config)
        """
        model_config = self.config.get('model', {})
        quant_config = self.config.get('quantization', {})
        lora_config = self.config.get('lora', {})

        model_name = model_config.get('name', "Qwen/Qwen2.5-0.5B-Instruct")

        # Only master logs model loading
        if self.sm_env.rank == 0:
            logger.info(f"Loading model: {model_name}")

        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get('load_in_4bit', True),
            bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_compute_dtype=getattr(torch, quant_config.get('bnb_4bit_compute_dtype', 'float16')),
            bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
        )

        # Load tokenizer (same on all ranks)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=model_config.get('trust_remote_code', True),
            padding_side=model_config.get('padding_side', 'left')
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load model with device-specific placement
        # Each process loads the model on its assigned GPU
        device_map = {'': self.sm_env.local_rank}

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=model_config.get('trust_remote_code', True),
            torch_dtype=getattr(torch, model_config.get('torch_dtype', 'float16')),
        )

        # Configure LoRA
        peft_config = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            bias=lora_config.get('bias', 'none'),
            task_type=lora_config.get('task_type', 'CAUSAL_LM'),
        )

        # Apply LoRA
        model = get_peft_model(model, peft_config)

        # Only master prints model info
        if self.sm_env.rank == 0:
            model.print_trainable_parameters()

        # Prepare model for distributed training
        if self.sm_env.world_size > 1:
            # Important: Set find_unused_parameters=False for efficiency
            # DPO models typically use all parameters
            model = DDP(
                model,
                device_ids=[self.sm_env.local_rank],
                output_device=self.sm_env.local_rank,
                find_unused_parameters=False
            )

        return model, tokenizer, peft_config

    def prepare_datasets(self, train_dataset) -> DataLoader:
        """
        Prepare datasets for distributed training.

        This function:
        - Creates a DistributedSampler for proper data sharding
        - Ensures each GPU gets a unique subset of data
        - Handles proper shuffling across epochs

        Args:
            train_dataset: The training dataset

        Returns:
            DataLoader configured for distributed training
        """
        training_config = self.config.get('training', {})

        # Create distributed sampler
        # This ensures each GPU gets different data
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.sm_env.world_size,
            rank=self.sm_env.rank,
            shuffle=True,
            seed=self.config.get('seeds', {}).get('torch', 42)
        )

        # Create DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_config.get('per_device_train_batch_size', 1),
            sampler=train_sampler,
            num_workers=training_config.get('dataloader_num_workers', 2),
            pin_memory=True,  # Faster GPU transfer
            drop_last=True    # Required for DDP to have same batch size across GPUs
        )

        return train_dataloader, train_sampler

    def train_dpo_model(self, model, tokenizer, train_dataset, peft_config):
        """
        Execute distributed DPO training.

        This is the main training loop that:
        - Configures DPO training arguments
        - Handles distributed synchronization
        - Manages checkpointing and model saving

        Args:
            model: The model to train (already wrapped in DDP if needed)
            tokenizer: The tokenizer
            train_dataset: Training dataset
            peft_config: PEFT configuration
        """
        if self.sm_env.rank == 0:
            logger.info("Starting distributed DPO training...")

        training_config = self.config.get('training', {})
        dpo_config = self.config.get('dpo', {})

        # Calculate effective batch size
        # In distributed training: effective_batch_size = per_device_batch_size * gradient_accumulation * world_size
        per_device_batch_size = training_config.get('per_device_train_batch_size', 1)
        gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 4)
        effective_batch_size = per_device_batch_size * gradient_accumulation_steps * self.sm_env.world_size

        if self.sm_env.rank == 0:
            logger.info(f"Effective batch size: {effective_batch_size}")
            logger.info(f"  Per device: {per_device_batch_size}")
            logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
            logger.info(f"  World size: {self.sm_env.world_size}")
            logger.info(f"  Training epoch: {training_config.get('num_train_epochs', 1)}")

        # DPO Training Configuration
        training_args = DPOConfig(
            output_dir=training_config.get('output_dir', str(self.sm_env.output_dir / 'checkpoints')),
            num_train_epochs=training_config.get('num_train_epochs', 1),
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 1),
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            learning_rate=training_config.get('learning_rate', 5e-5),
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
            warmup_steps=training_config.get('warmup_steps', 100),
            logging_steps=training_config.get('logging_steps', 10),
            save_steps=training_config.get('save_steps', 100),
            save_total_limit=training_config.get('save_total_limit', 2),

            # Distributed training specific
            local_rank=self.sm_env.local_rank,
            ddp_find_unused_parameters=False,  # More efficient
            ddp_backend='nccl',  # Best for GPU

            # Disable features that don't work well with distributed
            push_to_hub=False,
            report_to='none' if self.sm_env.rank != 0 else training_config.get('report_to', 'none'),

            # Only save on main process
            save_on_each_node=False,

            # DPO specific parameters
            beta=dpo_config.get('beta', 0.1),
            loss_type=dpo_config.get('loss_type', 'sigmoid'),
            max_length=dpo_config.get('max_length', 512),
            max_prompt_length=dpo_config.get('max_prompt_length', 256),

            # Optimization
            fp16=True,  # Use mixed precision
            dataloader_num_workers=training_config.get('dataloader_num_workers', 2),
            remove_unused_columns=False,

            # Show steps with epoch info
            logging_strategy="epoch",
            # logging_steps=1,  # Log every step to see progression
        )

        # Create reference model (needed for DPO)
        # Each rank loads its own reference model
        ref_model = self._create_reference_model()

        # Unwrap model if using DDP for trainer
        train_model = model.module if hasattr(model, 'module') else model

        # Initialize DPO Trainer
        trainer = DPOTrainer(
            model=train_model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        # Synchronize before training
        if dist.is_initialized():
            dist.barrier()

        # Start training
        if self.sm_env.rank == 0:
            logger.info("Beginning distributed DPO training...")

        trainer.train()

        # Synchronize after training
        if dist.is_initialized():
            dist.barrier()

        # Save the final model (only on rank 0)
        if self.sm_env.rank == 0:
            logger.info("Saving trained model...")
            final_model_path = self.sm_env.model_dir

            # Save the model
            trainer.save_model(str(final_model_path))
            tokenizer.save_pretrained(str(final_model_path))

            # Save training metadata
            metadata = {
                'training_completed': datetime.now().isoformat(),
                'world_size': self.sm_env.world_size,
                'effective_batch_size': effective_batch_size,
                'model_name': self.config.get('model', {}).get('name'),
                'training_steps': trainer.state.global_step,
            }

            with open(final_model_path / 'training_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved to {final_model_path}")

        # Final synchronization
        if dist.is_initialized():
            dist.barrier()

        return trainer

    def _create_reference_model(self):
        """
        Create reference model for DPO training.

        The reference model is a frozen copy of the original model
        used to compute the DPO loss.
        """
        model_config = self.config.get('model', {})
        quant_config = self.config.get('quantization', {})

        model_name = model_config.get('name', "Qwen/Qwen2.5-0.5B-Instruct")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get('load_in_4bit', True),
            bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_compute_dtype=getattr(torch, quant_config.get('bnb_4bit_compute_dtype', 'float16')),
            bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
        )

        # Each rank loads the reference model on its GPU
        device_map = {'': self.sm_env.local_rank}

        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=model_config.get('trust_remote_code', True),
            torch_dtype=getattr(torch, model_config.get('torch_dtype', 'float16')),
        )

        return ref_model

    def run_training(self):
        """
        Execute the complete distributed training pipeline.

        This orchestrates:
        1. Environment setup and validation
        2. Data loading and distribution
        3. Model initialization
        4. Distributed training
        5. Cleanup and finalization
        """
        try:
            # Set random seeds for reproducibility
            self._setup_seeds()

            # Only master process logs pipeline steps
            if self.sm_env.rank == 0:
                logger.info("=" * 50)
                logger.info("Starting Distributed DPO Training Pipeline")
                logger.info(f"Model: {self.config.get('model', {}).get('name')}")
                logger.info(f"World Size: {self.sm_env.world_size}")
                logger.info("=" * 50)

            # Step 1: Process data
            with self._distributed_context():
                if self.sm_env.rank == 0:
                    logger.info("Step 1: Processing training data...")

                data_processor = load_data_processor(self.config)
                train_dataset = data_processor.process_data()

                # Prepare distributed data loading
                train_dataloader, train_sampler = self.prepare_datasets(train_dataset)

            # Step 2: Setup model and tokenizer
            with self._distributed_context():
                if self.sm_env.rank == 0:
                    logger.info("Step 2: Setting up model and tokenizer...")

                model, tokenizer, peft_config = self.setup_model_and_tokenizer()

            # Step 3: Train the model
            with self._distributed_context():
                if self.sm_env.rank == 0:
                    logger.info("Step 3: Starting distributed DPO training...")

                trainer = self.train_dpo_model(model, tokenizer, train_dataset, peft_config)

            # Step 4: Finalize
            if self.sm_env.rank == 0:
                logger.info("=" * 50)
                logger.info("Training completed successfully!")
                logger.info(f"Model saved to: {self.sm_env.model_dir}")
                logger.info("=" * 50)

        except Exception as e:
            logger.error(f"Training failed on rank {self.sm_env.rank}: {str(e)}", exc_info=True)
            raise
        finally:
            # Cleanup distributed training
            self._cleanup_distributed()

    def _setup_seeds(self):
        """Set random seeds across all frameworks for reproducibility."""
        seeds_config = self.config.get('seeds', {})

        # Get seeds
        torch_seed = seeds_config.get('torch', 42)
        python_seed = seeds_config.get('python', 42)
        numpy_seed = seeds_config.get('numpy', 42)

        # Apply seeds
        random.seed(python_seed + self.sm_env.rank)  # Different seed per rank
        np.random.seed(numpy_seed + self.sm_env.rank)
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)

        # Use transformers set_seed for comprehensive seeding
        set_seed(torch_seed)

        if self.sm_env.rank == 0:
            logger.info("Random seeds configured for reproducibility")

    def _cleanup_distributed(self):
        """Clean up distributed training resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info(f"Cleaned up distributed training on rank {self.sm_env.rank}")

def parse_args():
    """
    Parse command-line arguments including SageMaker hyperparameters.

    SageMaker passes hyperparameters as command-line arguments.
    """
    parser = argparse.ArgumentParser(description='DPO Training with Hyperparameter Override')

    # Config file path (optional, can be overridden)
    parser.add_argument('--config_path', type=str, 
                       default='dpo_config.yaml',
                       help='Path to configuration file')

    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=None,
                       help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=None,
                       help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=None,
                       help='LoRA dropout rate')

    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=None,
                       help='Training batch size per device')
    parser.add_argument('--num_train_epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                       help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=None,
                       help='Number of warmup steps')
    parser.add_argument('--lr_scheduler_type', type=str, default=None,
                       help='Learning rate scheduler type')

    # DPO specific hyperparameters
    parser.add_argument('--dpo_beta', type=float, default=None,
                       help='DPO beta (temperature) parameter')
    parser.add_argument('--dpo_loss_type', type=str, default=None,
                       help='DPO loss type (sigmoid, hinge, etc.)')
    parser.add_argument('--max_length', type=int, default=None,
                       help='Maximum sequence length')
    parser.add_argument('--max_prompt_length', type=int, default=None,
                       help='Maximum prompt length')

    # Data hyperparameters
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of training samples to use')
    parser.add_argument('--data_processor_type', type=str, default=None,
                       help='Data processor type (test, anthropic_hh_rlhf)')

    # Model selection
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name to use')

    # Advanced: JSON string for complex overrides
    parser.add_argument('--config_overrides', type=str, default=None,
                       help='JSON string with config overrides')

    return parser.parse_args()


def merge_hyperparameters(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge command-line hyperparameters into configuration dictionary.

    Args:
        config: Base configuration from YAML
        args: Parsed command-line arguments

    Returns:
        Updated configuration with hyperparameter overrides
    """
    # Create a copy to avoid modifying original
    config = config.copy()

    # LoRA parameters
    if args.lora_r is not None:
        config.setdefault('lora', {})['r'] = args.lora_r
    if args.lora_alpha is not None:
        config.setdefault('lora', {})['lora_alpha'] = args.lora_alpha
    if args.lora_dropout is not None:
        config.setdefault('lora', {})['lora_dropout'] = args.lora_dropout

    # Training parameters
    if args.learning_rate is not None:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
    if args.per_device_train_batch_size is not None:
        config.setdefault('training', {})['per_device_train_batch_size'] = args.per_device_train_batch_size
    if args.num_train_epochs is not None:
        config.setdefault('training', {})['num_train_epochs'] = args.num_train_epochs
    if args.gradient_accumulation_steps is not None:
        config.setdefault('training', {})['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.warmup_steps is not None:
        config.setdefault('training', {})['warmup_steps'] = args.warmup_steps
    if args.lr_scheduler_type is not None:
        config.setdefault('training', {})['lr_scheduler_type'] = args.lr_scheduler_type

    # DPO parameters
    if args.dpo_beta is not None:
        config.setdefault('dpo', {})['beta'] = args.dpo_beta
    if args.dpo_loss_type is not None:
        config.setdefault('dpo', {})['loss_type'] = args.dpo_loss_type
    if args.max_length is not None:
        config.setdefault('dpo', {})['max_length'] = args.max_length
    if args.max_prompt_length is not None:
        config.setdefault('dpo', {})['max_prompt_length'] = args.max_prompt_length

    # Data parameters
    if args.num_samples is not None:
        config.setdefault('data', {})['num_samples'] = args.num_samples
    if args.data_processor_type is not None:
        config.setdefault('data', {})['processor_type'] = args.data_processor_type

    # Model parameters
    if args.model_name is not None:
        config.setdefault('model', {})['name'] = args.model_name

    # Advanced: Handle JSON overrides
    if args.config_overrides:
        try:
            overrides = json.loads(args.config_overrides)
            config = deep_merge_dicts(config, overrides)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse config_overrides JSON: {e}")

    return config


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def main():
    """
    Main entry point for distributed training with hyperparameter support.
    """
    try:
        # Parse command-line arguments
        args = parse_args()

        # Initialize SageMaker environment
        sm_env = SageMakerEnvironment()

        # Load base configuration
        config_path = Path(args.config_path)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent / config_path

        with open(config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        # Merge hyperparameters
        config = merge_hyperparameters(base_config, args)

        # Log hyperparameter overrides on main process
        if sm_env.rank == 0:
            logger.info("Hyperparameter overrides applied:")
            for arg_name, arg_value in vars(args).items():
                if arg_value is not None and arg_name not in ['config_path', 'config_overrides']:
                    logger.info(f"  {arg_name}: {arg_value}")

        # Create and run trainer with merged config
        trainer = DistributedDPOTrainer(config, sm_env)
        trainer.run_training()

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()