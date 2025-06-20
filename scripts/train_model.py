#!/usr/bin/env python
# scripts/train.py

import os
import sys
from pathlib import Path
import argparse
import json
import logging
import random
import numpy as np
import torch
import wandb

os.environ['NUMBA_CACHE_DIR' ] = '/tmp/'

# Disable setting seeds for huggingface because it causes issues with cache access
#from transformers import set_seed as huggingface_set_seed

# A function to load environment variables from .env file
def load_env_file(env_path='.env'):
    """Load environment variables from .env file"""
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        os.environ[key] = value.strip("'").strip('"')
                    except ValueError:
                        # Skip lines that don't have the format KEY=VALUE
                        continue
        return True
    else:
        print(f"Warning: .env file not found at {env_path}. "
              "Environment variables will not be loaded.")
        return False

# Try to load from .env file in project root
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
env_path = project_root / '.env'
load_env_file(env_path)

# Add project root to Python path
sys.path.insert(0, str(project_root))

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Local imports -- don't move these to the top 
from src.utils.config import load_config
from src.data.dataset import (
    load_datasets, 
    build_vocabulary, 
    create_processor, 
    prepare_datasets
)

from src.models.factory import create_asr_model
from src.training.collator import DataCollatorCTCWithPadding
from src.training.trainer import create_asr_trainer

# Verify HF_HOME is set and the directory exists
# hf_home = os.environ.get('HF_HOME')

# if hf_home:
#     os.makedirs(hf_home, exist_ok=True)
#     print(f"Using HF_HOME: {hf_home}")

# else:
#     print("Warning: HF_HOME not set in environment variables")

#     # Set a default if not found
#     default_hf_home = os.path.expanduser('~/.cache/huggingface')
#     os.environ['HF_HOME'] = default_hf_home
#     os.makedirs(default_hf_home, exist_ok=True)
#     print(f"Setting default HF_HOME to: {default_hf_home}")


# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def setup_environment():
    """Set up necessary environment variables."""
    # Get API keys from environment (assumes they're already set)
    wandb_key = os.environ.get("WANDB_API_KEY")
    hf_key = os.environ.get("HF_API_KEY")
    
    if not wandb_key:
        logging.warning("WANDB_API_KEY not found in environment variables")
    if not hf_key:
        logging.warning("HF_API_KEY not found in environment variables")


def setup_seed(seed: int):
    """Set up random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For reproducible operations on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ASR model')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to config YAML file'
    )
    return parser.parse_args()


def main():
    """Main training function."""
    
    # Setup logging and environment
    setup_logging()
    setup_environment()
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    logging.info(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Setup random seed for reproducibility
    setup_seed(config.seed)

    # disabled seed for huggingface because it causes issues with cache access
    # TODO: investigate why this is happening
    #huggingface_set_seed(config.seed)
    
    # Login to wandb
    logging.info("Logging in to Weights & Biases...")
    if os.environ.get("WANDB_API_KEY"):
        wandb.login()
        wandb.init(
            project=config.project,
            config={
                'learning_rate': config.learning_rate,
                'model': config.pretrained_model,
                'batch_size': config.batch_size,
                'epochs': config.num_epochs
            }
        )
    else:
        logging.warning("WANDB_API_KEY not found in environment variables. "
                        "Weights & Biases logging will be disabled.")
    
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load datasets
    logging.info(f"Loading datasets from {config.dataset_path}...")
    train_dataset, eval_dataset = load_datasets(config)
    logging.info(f"Loaded {len(train_dataset)} training samples "
                 f"and {len(eval_dataset)} test samples")
    
    # Build vocabulary
    logging.info("Building vocabulary...")
    
    vocab_path = os.path.join(project_root, "vocab")
    os.makedirs(vocab_path, exist_ok=True)

    vocab_file = os.path.join(vocab_path, "vocab.json")

    vocab_dict = build_vocabulary(train_dataset, eval_dataset, vocab_file)

    logging.info(f"Size of vocabulary created: {len(vocab_dict)}.")
    

    # save vocab_dict as JSON -- but this is handled by build_vocabulary
    # with open(vocab_file, 'w') as f:
    #     json.dump(vocab_dict, f, indent=4)

    
    # Create processor
    logging.info("Creating processor...")
    processor = create_processor(config, vocab_path)

    logging.info(f"Type of the processor: {type(processor)}")   
    
    # Prepare datasets
    logging.info("Preparing datasets...")
    train_dataset, eval_dataset = prepare_datasets(
        train_dataset, eval_dataset, processor
    )
    
    # Create model
    logging.info(f"Creating model from {config.pretrained_model}...")
    model = create_asr_model(config, processor)
    
    # Create data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, 
        padding=True
    )
    
    # Create and run trainer
    logging.info("Starting training model for ASR...")
    trainer = create_asr_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processor=processor,
        config=config
    )
    
    # Train model
    trainer.train()
    
    # Save model
    experiment_name = config.get_experiment_name()
    model_output_dir = output_dir / experiment_name
    model_output_dir.mkdir(exist_ok=True, parents=True)
    
    trainer.save_model(str(model_output_dir))
    processor.save_pretrained(str(model_output_dir))

    # push model to Hugging Face Hub as a private model
    # if os.environ.get("HF_API_KEY"):
    #     model.push_to_hub(
    #         repo_id=f"badrex/{experiment_name}",
    #         use_auth_token=os.environ["HF_API_KEY"],
    #         private=True
    #     )
    #     logging.info(f"Pushed model to Hugging Face Hub: badrex/{experiment_name}")
    
    # Final evaluation
    metrics = trainer.evaluate()
    logging.info(f"Final evaluation metrics: {metrics}")
    
    # Save metrics
    with open(model_output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    
    logging.info(f"Training completed. Model saved to {model_output_dir}")

    # add code to delete dataset cache
    train_dataset.cleanup_cache_files()
    eval_dataset.cleanup_cache_files()


if __name__ == "__main__":
    main()