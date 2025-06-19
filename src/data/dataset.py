# src/data/dataset.py
import json
import logging
from typing import Dict, Tuple, List, Any, Optional
from datasets import load_dataset, Dataset, Audio, concatenate_datasets

from transformers import (
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2Processor
)

from src.data.preprocessing import (
    clean_text_batch, 
    extract_all_chars, 
    create_vocabulary, 
    prepare_dataset
)

from src.utils.config import ASRConfig


def load_datasets(config: ASRConfig) -> Tuple[Dataset, Dataset]:
    """Load and prepare datasets for training and evaluation.
    
    Args:
        config: Configuration object containing dataset parameters
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Load custom dataset if specified
    if hasattr(config, 'use_custom_dataset') and config.use_custom_train_dataset:
        if hasattr(config, 'custom_dataset_path') and config.dataset_path:
            logging.info(f"Loading custom training dataset from "
                         f"{config.custom_dataset_path}")
            
            dataset = Dataset.load_from_disk(config.custom_train_dataset_path)

        else:
            raise ValueError("custom_dataset_path must be specified when use_custom_train_dataset is True")
    else:
        # not implemented yet
        raise NotImplementedError(
            "Loading from HuggingFace datasets is not implemented yet"
        )
    
    train_dataset = dataset['train']
    test_dataset = dataset['validation']

    # Remove unwanted features
    features_to_remove = [
        "audio_id", "gender", "age_group", "category", 
    ]
    
    # Sample dataset if specified
    if config.sample:
        train_dataset = train_dataset.select(range(config.sample_size))
    
    # Remove columns and clean text
    train_dataset = train_dataset.remove_columns(features_to_remove)
    test_dataset = test_dataset.remove_columns(features_to_remove)
    
    # Preprocess text transcripts by removing special characters
    train_dataset = train_dataset.map(
        lambda batch: clean_text_batch(batch, config.chars_to_remove_regex)
    )
    test_dataset = test_dataset.map(
        lambda batch: clean_text_batch(batch, config.chars_to_remove_regex)
    )
    
    return train_dataset, test_dataset


def build_vocabulary(train_dataset: Dataset, test_dataset: Dataset, output_path: str = "vocab.json") -> Dict[str, int]:
    """Build vocabulary from datasets and save it to a file.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        output_path: Path to save vocabulary JSON file
        
    Returns:
        Vocabulary dictionary
    """
    # Extract all characters
    vocab_train = train_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train_dataset.column_names
    )
    
    vocab_test = test_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=test_dataset.column_names
    )
    
    # Combine vocabularies
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    
    # Add special tokens
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    # Save vocabulary to file
    with open(output_path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    
    return vocab_dict


def create_processor(vocab_path: str = "./vocab") -> Wav2Vec2Processor:
    """Create Wav2Vec2Processor from tokenizer and feature extractor.
    
    Args:
        vocab_path: Path to directory containing vocabulary file
        
    Returns:
        Wav2Vec2Processor for processing audio and text
    """
    # Initialize tokenizer
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    
    # Initialize feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    
    # Combine into processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    return processor


def prepare_datasets(train_dataset: Dataset, eval_dataset: Dataset, processor: Wav2Vec2Processor) -> Tuple[Dataset, Dataset]:
    """Prepare datasets for training by adding processed inputs.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        processor: Wav2Vec2Processor for processing audio and text
        
    Returns:
        Tuple of prepared (train_dataset, test_dataset)
    """
    # Cast audio column to Audio with correct sampling rate, if not already 16000 Hz
    # commented out because it is not needed for the current dataset

    # train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    # test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    
    # Prepare datasets
    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=eval_dataset.column_names
    )
    
    return train_dataset, eval_dataset