# src/models/factory.py
from typing import Optional
import torch
from transformers import Wav2Vec2ForCTC, AutoModelForCTC, Wav2Vec2Processor

from src.utils.config import ASRConfig


def create_asr_model(config: ASRConfig, 
                     processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:
    """
    Create and configure a Wav2Vec2 ASR model.
    
    Args:
        config: ASR configuration object
        processor: Wav2Vec2Processor with tokenizer information
        
    Returns:
        Configured Wav2Vec2ForCTC model
    """
    # Get pretrained model path
    pretrained_model_path = config.get_pretrained_model_path()
    
    # Initialize model
    model = AutoModelForCTC.from_pretrained(
        pretrained_model_path,
        attention_dropout=0.05,
        hidden_dropout=0.05,
        feat_proj_dropout=0.05,
        mask_time_prob=0.05,
        layerdrop=0.025,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    
    # Apply freezing configuration
    # if model is based on wav2vec2, freeze feature encoder if specified
    if hasattr(model, 'freeze_feature_encoder') and config.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # # If model is based on SeamlessM4T, freeze feature encoder if specified
    # elif hasattr(model, 'freeze_feature_extractor') and config.freeze_feature_encoder:
    #     #model.freeze_feature_extractor()
    #     pass
    
    return model


def get_device() -> torch.device:
    """
    Get the appropriate device for training.
    
    Returns:
        torch.device for training
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
