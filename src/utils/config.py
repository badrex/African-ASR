# src/utils/config.py
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import yaml
import datetime


@dataclass
class ASRConfig:
    """Configuration for ASR training and evaluation."""
    # Project settings
    project: str
    output_dir: str
    seed: int
    
    # Model settings
    pretrained_model: str
    freeze_feature_encoder: bool = True
    
    # Training settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 30
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    fp16: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = 400
    eval_steps: int = 400
    logging_steps: int = 10
    save_total_limit: int = 2

       
    # Step-based training settings
    max_steps: int = -1  # -1 means train for specified epochs, positive value means train for that many steps
    warmup_steps: int = 0  # Specific number of warmup steps (instead of ratio)
    
    # Data settings
    train_split: str = "train"
    eval_split: str = "validation"
    
    use_custom_dataset: bool = False
    dataset_path: Optional[str] = None
    sample: bool = False
    sample_size: int = 1000
    #chars_to_remove_regex: str = r'[\,\?\.\!\-\;\:\"\"\%\"\�\']'
    
    # Model mappings
    pretrained_model_map: Dict[str, str] = field(default_factory=lambda: {
        "xlsr-128": "facebook/wav2vec2-xls-r-300m",
        "xlsr-53": "facebook/wav2vec2-large-xlsr-53",
        "mHuBERT-147": "utter-project/mHuBERT-147",
        "w2v-BERT": "facebook/w2v-bert-2.0",
        "mms-300m": "facebook/mms-300m"
    })
    

    def get_pretrained_model_path(self) -> str:
        """Get the actual model path from the model name."""
        return self.pretrained_model_map.get(
            self.pretrained_model, 
            self.pretrained_model
        )
    
    def get_experiment_name(self) -> str:
        """Generate a consistent experiment name with timestamp."""
        timestamp = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")

        model_name_str = [
            self.pretrained_model,  
            timestamp
        ]

        return f"{'-'.join(model_name_str)}"


def load_config(config_path: str) -> ASRConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return ASRConfig(**config_dict)