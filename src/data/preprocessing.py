# src/data/preprocessing.py
import re
from typing import Dict, List, Any
from datasets import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2BertProcessor


def clean_text(text):
    """clean single text string"""
    # replace problematic chars with standard equivalents
    #text = text.replace('\xa0', ' ')
    #text = text.replace('，', ' ')  # remove chinese comma
    #text = text.replace('．', '.')
    
    # keep only allowed characters
    allowed = "abcdefghijklmnopqrstuvwxyz \'" 
    text = ''.join(c for c in text if c.lower() in allowed)
    
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

def clean_text_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """clean text in a batch of data"""

    # apply cleaning to all transcriptions in the batch
    batch["clean_transcription"] = [
        clean_text(text) for text in batch["transcription"]
    ]
    return batch


def extract_all_chars(batch: Dict[str, List[str]]) -> Dict[str, List[Any]]:
    """Extract unique characters from all sentences in a batch.
    
    Args:
        batch: Dictionary containing batch data with 'sentence' field
        
    Returns:
        Dictionary with vocabulary and text information
    """
    all_text = " ".join(batch["clean_transcription"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


# def create_vocabulary(dataset: Dataset) -> Dict[str, int]:
#     """Create vocabulary dictionary from dataset.
    
#     Args:
#         dataset: Dataset containing text to build vocabulary from
        
#     Returns:
#         Dictionary mapping characters to integer indices
#     """
#     # Extract unique characters
#     vocab_list = dataset["vocab"][0]
#     vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    
#     # Handle special tokens
#     vocab_dict["|"] = vocab_dict[" "]
#     del vocab_dict[" "]
    
#     vocab_dict["[UNK]"] = len(vocab_dict)
#     vocab_dict["[PAD]"] = len(vocab_dict)
    
#     return vocab_dict


# def prepare_dataset(batch: Dict[str, Any], processor) -> Dict[str, Any]:
#     """Prepare dataset for training by processing audio and tokenizing text.
    
#     Args:
#         batch: Dictionary containing batch data
#         processor: 
#             Wav2Vec2Processor or Wav2Vec2BertProcessor for audio and text processing
        
#     Returns:
#         Processed batch ready for model input
#     """
#     # Process audio
#     audio = batch["audio"]

#     # check wheather the processor is Wav2Vec2BertProcessor
#     if isinstance(processor, Wav2Vec2BertProcessor):
#         batch["input_features"] = processor(
#             audio["array"], 
#             sampling_rate=audio["sampling_rate"]
#         ).input_features[0]

#         batch["input_length"] = len(batch["input_features"])

#     # for Wav2Vec2Processor or similar processors 
#     else:
#         batch["input_values"] = processor(
#             audio["array"], 
#             sampling_rate=audio["sampling_rate"]
#         ).input_values[0]

#         batch["input_length"] = len(batch["input_values"])
    
#     # Process text (updated approach)
#     batch["labels"] = processor(text=batch["clean_transcription"]).input_ids
    
#     return batch

def prepare_dataset(batch: Dict[str, Any], processor) -> Dict[str, Any]:
    """prepare dataset for training by processing audio and tokenizing text.
        Args:
        batch: Dictionary containing batch data
        processor: 
            Wav2Vec2Processor/Wav2Vec2BertProcessor for audio/text processing
        
    Returns:
        Processed batch ready for model input
    """

    # Process audio
    audio = batch["audio"]
    features = processor(audio["array"], sampling_rate=audio["sampling_rate"])
    
    # handle both processor types in one line
    is_w2vBERT = isinstance(processor, Wav2Vec2BertProcessor)

    if is_w2vBERT: 
        key = "input_features"
        features = features.input_features[0]
    else:
        # for Wav2Vec2Processor or similar processors
        key = "input_values"
        features = features.input_values[0]
    
    batch[key] = features
    batch["length"] = len(features)
    batch["labels"] = processor(text=batch["clean_transcription"]).input_ids
    
    return batch
