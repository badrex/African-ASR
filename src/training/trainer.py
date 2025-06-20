# src/training/trainer.py
import os
from typing import Dict, Any, Optional
import torch
from transformers import Trainer, TrainingArguments, Wav2Vec2Processor
import evaluate

from src.utils.config import ASRConfig
from src.training.metrics import compute_metrics


def create_training_args(config: ASRConfig, experiment_name: str) -> TrainingArguments:
    """Create training arguments for the Trainer."""
    output_dir = os.path.join(config.output_dir, experiment_name)
    
    # Configure training based on max_steps or epochs
    if hasattr(config, 'max_steps') and config.max_steps > 0:
        # Step-based training - use a very large number of epochs
        # This way the trainer will stop based on steps, not epochs
        num_train_epochs = 9999
        max_steps = config.max_steps
    else:
        # Epoch-based training (existing approach)
        num_train_epochs = config.num_epochs
        max_steps = -1
    
    # Configure warmup based on steps or ratio
    if hasattr(config, 'warmup_steps') and config.warmup_steps > 0:
        # Step-based warmup
        warmup_steps = config.warmup_steps
        warmup_ratio = 0.0
    else:
        # Ratio-based warmup (existing approach)
        warmup_steps = 0
        warmup_ratio = config.warmup_ratio
    
    return TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="steps",
        num_train_epochs=num_train_epochs,  # Always a number, never None
        max_steps=max_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=config.fp16,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-08,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_strategy="best",
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        save_total_limit=config.save_total_limit,
        push_to_hub=False,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="score",
        greater_is_better=True,
    )  


def create_asr_trainer(
    model: torch.nn.Module,
    train_dataset,
    eval_dataset,
    data_collator,
    processor: Wav2Vec2Processor,
    config: ASRConfig
) -> Trainer:
    """
    Create a trainer for ASR model training.
    
    Args:
        model: ASR model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator for batching
        processor: Wav2Vec2Processor
        config: ASR configuration
        
    Returns:
        Configured Trainer object
    """
    # Create experiment name
    experiment_name = config.get_experiment_name()
    
    # Create training arguments
    training_args = create_training_args(config, experiment_name)

    # Load eval metrics
    # this was moved from /src/training/metrics.py to here because evaluation loop was too slow
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(
            pred, 
            processor, 
            wer_metric=wer_metric,
            cer_metric=cer_metric
        ),
    )
    
    return trainer