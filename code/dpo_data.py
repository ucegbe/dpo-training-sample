#!/usr/bin/env python3
"""
DPO Data Configuration and Processing
Handles data generation and dataset creation for DPO training
With distributed training support
"""

import os
import random
import logging
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from datasets import Dataset, load_dataset
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


#######################
### Data Processors ###
#######################


class DPODataProcessor(ABC):
    """
    Abstract base class for DPO data processors with distributed training support.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor with configuration.

        Args:
            config: Configuration dictionary containing data parameters
        """
        self.config = config
        self.data_config = config.get('data', {})

        # Distributed training properties
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_distributed = self.world_size > 1
        self.is_main_process = self.rank == 0

        # Set random seed for consistency across processes
        self._set_random_seed()

    def _set_random_seed(self):
        """Set random seed for reproducibility across all processes."""
        seed = self.data_config.get('seed', 42)
        random.seed(seed)
        # If using numpy
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass

        if self.is_main_process:
            logger.info(f"Data random seed set to: {seed}")

    @abstractmethod
    def load_data(self) -> Dataset:
        """
        Load data from a custom source (file, database, API, etc.).
        Must be implemented by subclasses.

        Returns:
            Dataset with 'prompt', 'chosen', 'rejected' keys
        """
        pass

    def process_data(self) -> Dataset:
        """
        Process and return a dataset ready for DPO training.

        Returns:
            Processed dataset ready for training
        """
        dataset = self.load_data()

        # Only log samples from main process to avoid duplicate logs
        if self.is_main_process:
            self.log(dataset)

        # Synchronize processes after data loading
        if self.is_distributed and dist.is_initialized():
            dist.barrier()

        return dataset

    def log(self, dataset: Dataset) -> None:
        """Log sample data (only on main process)."""
        log_samples = self.data_config.get('log_samples', 0)
        if log_samples:
            logger.info(f"Dataset size: {len(dataset)}")
            logger.info(f"Showing {log_samples} samples:")

            for i in range(min(log_samples, len(dataset))):
                item = dataset[i]
                logger.info(f"Sample {i+1}:")
                logger.info(f"  Prompt: {item['prompt'][:100]}...")  # Truncate long prompts
                logger.info(f"  Chosen: {item['chosen'][:100]}...")
                logger.info(f"  Rejected: {item['rejected'][:100]}...")
                logger.info("")


class DPODataProcessorTest(DPODataProcessor):
    """
    Test implementation of DPODataProcessor that generates mock data.
    Ensures consistent data generation across distributed processes.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the test data processor with configuration.

        Args:
            config: Configuration dictionary containing data parameters
        """
        super().__init__(config)

        # Base prompts for variety
        self.base_prompts = [
            "Explain the concept of artificial intelligence.",
            "Describe the process of photosynthesis.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Explain quantum computing in simple terms.",
            "What is the importance of biodiversity?",
            "Describe the water cycle.",
            "How do neural networks function?",
            "What are the effects of climate change?",
            "Explain the theory of relativity."
        ]

        # Words with varying amounts of 'r' characters
        self.high_r_words = [
            "remarkable", "extraordinary", "terrific", "marvelous", "superior", 
            "brilliant", "spectacular", "tremendous", "incredible", "wonderful",
            "powerful", "revolutionary", "transformative", "progressive", "remarkable",
            "extraordinary", "terrific", "marvelous", "superior", "brilliant"
        ]

        self.low_r_words = [
            "good", "nice", "fine", "okay", "decent", "adequate", "suitable",
            "acceptable", "satisfying", "pleasant", "positive", "beneficial",
            "useful", "helpful", "valuable", "effective", "efficient", "optimal"
        ]

        self.num_samples = self.data_config.get('num_samples', 50)
        self.log_samples = self.data_config.get('log_samples', 3)

    def load_data(self) -> Dataset:
        """
        Generate mock DPO training data with consistent results across processes.

        Returns:
            Dataset with 'prompt', 'chosen', 'rejected' keys
        """
        if self.is_main_process:
            logger.info(f"Generating {self.num_samples} mock training samples...")

        # Use a local Random instance with fixed seed for consistency
        # This ensures all processes generate the same data
        local_random = random.Random(self.data_config.get('seed', 42))

        mock_data = []

        for i in range(self.num_samples):
            # Select a random prompt using local random instance
            prompt = local_random.choice(self.base_prompts)

            # Generate chosen response (high 'r' count)
            chosen_words = local_random.sample(self.high_r_words, k=local_random.randint(3, 6))
            chosen = f"This is a {' and '.join(chosen_words)} explanation. " + \
                    f"The research reveals remarkable results regarding this topic. " + \
                    f"Furthermore, the extraordinary characteristics demonstrate superior performance."

            # Generate rejected response (low 'r' count)  
            rejected_words = local_random.sample(self.low_r_words, k=local_random.randint(3, 6))
            rejected = f"This is a {' and '.join(rejected_words)} explanation. " + \
                      f"The study shows good outcomes about this topic. " + \
                      f"Also, the basic qualities show decent outcomes."

            # Verify that chosen has more 'r' characters
            chosen_r_count = chosen.lower().count('r')
            rejected_r_count = rejected.lower().count('r')

            # If rejected has more or equal 'r's, add more 'r's to chosen
            if rejected_r_count >= chosen_r_count:
                chosen += " The research provides remarkable and extraordinary results."

            mock_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })

        return Dataset.from_list(mock_data)


class DPODataProcessorAnthropicHHRLHF(DPODataProcessor):
    """
    Huggingface: "Baidicoot/anthropic-hh-rlhf"
    Handles distributed loading efficiently.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the custom data processor.

        Args:
            config: Configuration dictionary containing data parameters
        """
        super().__init__(config)

        self.data_handle = "Baidicoot/anthropic-hh-rlhf"
        self.data_split = self.data_config.get('data_split', 'train')
        self.num_samples = self.data_config.get('num_samples', 0)

    def load_data(self) -> Dataset:
        """
        Load dataset with caching support for distributed training.

        Returns:
            Dataset ready for training
        """
        if self.is_main_process:
            logger.info(f"Loading dataset: {self.data_handle} (split: {self.data_split})")

        # Load dataset (HuggingFace datasets handles caching automatically)
        # All processes will load from cache after first process downloads
        dataset = load_dataset(
            self.data_handle,
            split=self.data_split,
            # Keep the dataset in memory for faster access
            keep_in_memory=True if self.num_samples and self.num_samples < 10000 else False
        )

        # Select subset if specified
        if self.num_samples:
            # Use consistent indices across all processes
            indices = list(range(min(self.num_samples, len(dataset))))
            dataset = dataset.select(indices)

            if self.is_main_process:
                logger.info(f"Selected {len(dataset)} samples from dataset")

        return dataset


##################
### Data Utils ###
##################


# Registry of available data processors
DATA_PROCESSOR_REGISTRY = {
    'test': DPODataProcessorTest,
    'anthropic_hh_rlhf': DPODataProcessorAnthropicHHRLHF,
}


def load_data_processor(config: Dict[str, Any]) -> DPODataProcessor:
    """
    Factory function to create a DPODataProcessor instance based on processor name.

    Args:
        config: Configuration dictionary containing processor_type

    Returns:
        Initialized DPODataProcessor instance

    Raises:
        ValueError: If processor_type is not found in registry
    """
    processor_type = config.get('data', {}).get('processor_type', 'test')

    if processor_type not in DATA_PROCESSOR_REGISTRY:
        available_processors = list(DATA_PROCESSOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown processor type: '{processor_type}'. "
            f"Available processors: {available_processors}"
        )

    processor_class = DATA_PROCESSOR_REGISTRY[processor_type]
    processor = processor_class(config)

    # Log processor info only from main process
    if processor.is_main_process:
        logger.info(f"Loaded data processor: {processor_type}")

    return processor


def register_data_processor(name: str, processor_class: type):
    """
    Register a new data processor class.

    Args:
        name: Name to register the processor under
        processor_class: The processor class to register
    """
    if not issubclass(processor_class, DPODataProcessor):
        raise ValueError(f"Processor class must be a subclass of DPODataProcessor")

    DATA_PROCESSOR_REGISTRY[name] = processor_class

    # Only log from main process
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        logger.info(f"Registered data processor: {name}")


def get_available_processors() -> List[str]:
    """
    Get list of available processor names.

    Returns:
        List of available processor names
    """
    return list(DATA_PROCESSOR_REGISTRY.keys())