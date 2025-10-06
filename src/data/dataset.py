# src/data/dataset.py
import json
import logging
from typing import Dict, Tuple, List, Any, Optional, Union
from datasets import load_dataset, Dataset, Audio, DatasetDict

from transformers import (
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor, 
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertProcessor,
    Wav2Vec2Processor
)

from src.data.preprocessing import (
    clean_text_batch, 
    extract_all_chars, 
    prepare_dataset, 
    prepare_dataset_batch
)

from src.utils.config import ASRConfig


def load_datasets(config: ASRConfig) -> Tuple[Dataset, Dataset]:
    """Load and prepare datasets for training and evaluation.
    
    Args:
        config: Configuration object containing dataset parameters
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Load custom dataset if specified
    if hasattr(config, 'use_custom_dataset') and config.use_custom_dataset:
        if hasattr(config, 'dataset_path') and config.dataset_path:
            logging.info(f"Loading custom training dataset locally from "
                         f"{config.dataset_path}...")
            
            dataset = DatasetDict.load_from_disk(config.dataset_path)

        else:
            raise ValueError(f"dataset_path to a local dataset must be specified "
                             f"when use_custom_dataset is True")
    else:
        # Load dataset from HF hub
        logging.info(f"Loading training dataset from HF hub from "
                     f"{config.dataset_path}...")
        dataset = load_dataset(
            config.dataset_path,
            verification_mode="no_checks", 
        )
    
    train_dataset = dataset[config.train_split]
    dev_dataset = dataset[config.eval_split]

    # Remove unwanted features
    #features_to_remove = [
    #    "audio_id", "gender", "age_group", "category", 
    #]
    
    # Sample dataset if specified
    if config.sample:
        logging.info(f"Sampling dataset to {config.sample_size} samples...")
        train_dataset = train_dataset.select(range(config.sample_size))
        #dev_dataset = dev_dataset.select(range(3989))
    
    # Remove features not used in training 
    logging.info(f"Removing unnecessary columns...")
    features_to_keep = [
        "audio", "transcription", "audio_duration",
    ]

    features_to_remove = [f for f in train_dataset.features if f not in features_to_keep]

    train_dataset = train_dataset.remove_columns(features_to_remove)
    dev_dataset = dev_dataset.remove_columns(features_to_remove)
    
    # Preprocess text transcripts by removing special characters
    logging.info(f"Preprocessing text transcripts...")
    train_dataset = train_dataset.map(
        lambda batch: clean_text_batch(batch),
        batched=True,
        batch_size=256,
    )
    dev_dataset = dev_dataset.map(
        lambda batch: clean_text_batch(batch),
        batched=True,
        batch_size=256,
    )
    
    return train_dataset, dev_dataset


def build_vocabulary(train_dataset: Dataset,
                     dev_dataset: Dataset,
                     output_path: str = "./vocab.json") -> Dict[str, int]:
    """Build vocabulary from datasets and save it to a file.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        output_path: Path to save vocabulary JSON file
        
    Returns:
        Vocabulary dictionary
    """
    # Extract all characters
    train_vocab = train_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train_dataset.column_names
    )
    
    dev_vocab = dev_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dev_dataset.column_names
    )
    
    # Combine vocabularies
    vocab_list = list(set(train_vocab["vocab"][0]) | set(dev_vocab["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    
    # Add special tokens
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    # Save vocabulary to file
    with open(f"{output_path}/vocab.json", 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file,  indent=4)
    
    return vocab_dict


ASRProcessor = Union[Wav2Vec2Processor, Wav2Vec2BertProcessor]

def create_processor(
        config: ASRConfig, 
        vocab_path: str = "./vocab") -> ASRProcessor:
    """Create a processor from tokenizer and feature extractor.
    
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

    # debug 
    print("Tokenizer class:", tokenizer.__class__)
    print("Special tokens:", tokenizer.special_tokens_map)
    print("All tokens:", tokenizer.get_vocab().keys())

    
    # Initialize feature extractor
    if config.pretrained_model == "facebook/w2v-bert-2.0":
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

        # Combine into processor
        processor = Wav2Vec2BertProcessor(
            feature_extractor=feature_extractor, 
            tokenizer=tokenizer
        )

    else:
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


def prepare_datasets(train_dataset: Dataset, 
                     eval_dataset: Dataset, 
                     processor: Wav2Vec2Processor) -> Tuple[Dataset, Dataset]:
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
    # train_dataset = train_dataset.map(
    #     lambda batch: prepare_dataset(batch, processor),
    #     num_proc=8,
    #     remove_columns=train_dataset.column_names
    # )
    
    # eval_dataset = eval_dataset.map(
    #     lambda batch: prepare_dataset(batch, processor),
    #     num_proc=8,
    #     remove_columns=eval_dataset.column_names
    # )

    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset_batch(batch, processor),
        batched=True,
        batch_size=32, # has to be based on available memory
        remove_columns=train_dataset.column_names
    )

    eval_dataset = eval_dataset.map(
        lambda batch: prepare_dataset_batch(batch, processor),
        batched=True,
        batch_size=32, # has to be based on available memory
        remove_columns=eval_dataset.column_names
    )   
    
    return train_dataset, eval_dataset
