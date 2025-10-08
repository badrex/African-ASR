# src/data/preprocessing.py
import re
import unicodedata
from typing import Dict, List, Any
from datasets import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2BertProcessor


def clean_text(text):
    """clean single text string"""
    
    # apply NFC normalization
    text = unicodedata.normalize("NFC", text)
    
    # replace problematic chars with standard equivalents
    #text = text.replace('\xa0', ' ')
    #text = text.replace('，', ' ')  # remove chinese comma
    #text = text.replace('．', '.')
    #text = text.replace("’", "'")
    text = text.replace("’", "'").replace("ʼ", "'")

    # simple replacements for lowercase accented characters
    # this is only for Fulani
    replacements = {
        "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a", "å": "a", "ā": "a",
        "é": "e", "è": "e", "ê": "e", "ë": "e", "ē": "e",
        "í": "i", "ì": "i", "î": "i", "ï": "i", "ī": "i",
        "ó": "o", "ò": "o", "ô": "o", "ö": "o", "õ": "o", "ō": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u", "ū": "u",
        "ç": "c",
        "ñ": "n",
        "ÿ": "y",
    }

    for src, tgt in replacements.items():
        text = text.replace(src, tgt)

    # keep only allowed characters
    # allowed = "abcdefghijklmnopqrstuvwxyz \'" 
    #allowed = " abcdefghijklmnopqrstuvwxyz0123456789\'"
    #allowed = "abcdefghijklmnopqrstuvwxyzĩũ \'"
    
    # this is only for Fulani
    #allowed = "abcdefghijklmnopqrstuvwxyzɓɗƴŋɲ '"
    
    # for Zulu 
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789 -'"


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

# batch implementation
def prepare_dataset_batch(batch: Dict[str, List[Any]], processor) -> Dict[str, List[Any]]:
    """
    Prepare dataset batch for training by processing audio and tokenizing text.
    
    Args:
        batch: Dictionary containing batch data
        processor: 
            Wav2Vec2Processor/Wav2Vec2BertProcessor for audio/text processing
        
    Returns:
        Processed batch ready for model input
    """
    # process multiple samples at once
    audio_arrays = [audio["array"] for audio in batch["audio"]]
    sampling_rates = [audio["sampling_rate"] for audio in batch["audio"]]
    
    # process all audio at once if possible
    features = processor(audio_arrays, sampling_rate=sampling_rates[0])
    
    is_w2vBERT = isinstance(processor, Wav2Vec2BertProcessor)
    if is_w2vBERT:
        key = "input_features"
        batch[key] = features.input_features
    else:
        key = "input_values"
        batch[key] = features.input_values
    
    batch["length"] = [len(f) for f in batch[key]]
    
    # tokenize all texts at once
    labels = processor(text=batch["clean_transcription"]).input_ids
    batch["labels"] = labels
    
    return batch
