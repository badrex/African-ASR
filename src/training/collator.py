# src/training/collator.py
from dataclasses import dataclass
from typing import Dict, List, Union
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2BertProcessor


# logging for debugging
# import logging
# logging.basicConfig(level=logging.INFO)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that dynamically pads inputs for CTC training.
    
    Args:
        processor: Wav2Vec2Processor for processing inputs
        padding: Padding strategy (True/False or 'longest')
    """
    processor: Union[Wav2Vec2Processor, Wav2Vec2BertProcessor]  # Processor for audio and text
    padding: Union[bool, str] = True

    def __call__(
        self, 
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate and pad a batch of examples.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Batch dictionary with padded tensors
        """

        if isinstance(self.processor, Wav2Vec2BertProcessor):
            input_key = "input_features"
        else:
            input_key = "input_values"


        input_features = [
            {input_key: feature[input_key]} for feature in features
        ]

        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]

        # Pad inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels using the tokenizer, not the processor
        tokenizer = self.processor.tokenizer

        labels_batch = tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch