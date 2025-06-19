#!/usr/bin/env python
# scr/data/data_stats.py
import os
import sys
import random
import logging
from pathlib import Path
import pandas as pd
#from datasets import load_dataset
from tqdm.notebook import tqdm

import logging


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
    return False

# Try to load from .env file in project root
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
env_path = project_root / '.env'
load_env_file(env_path)


# Add project root to Python path
sys.path.insert(0, str(project_root))

# Verify HF_HOME is set and the directory exists
hf_home = os.environ.get('HF_HOME')

if hf_home:
    os.makedirs(hf_home, exist_ok=True)
    print(f"Using HF_HOME: {hf_home}")

else:
    print("Warning: HF_HOME not set in environment variables")

    # Set a default if not found
    default_hf_home = os.path.expanduser('~/.cache/huggingface')
    os.environ['HF_HOME'] = default_hf_home
    os.makedirs(default_hf_home, exist_ok=True)
    print(f"Setting default HF_HOME to: {default_hf_home}")


# Verify TORCH_HOME is set and the directory exists
# torch_home = os.environ.get('TORCH_HOME')

# if torch_home:
#     os.makedirs(torch_home, exist_ok=True)
#     print(f"Using TORCH_HOME: {torch_home}")

# else:

#     print("TORCH_HOME not set in environment variables")

#     # Set a default if not found
#     default_torch_home = os.path.expanduser('~/.cache/torch')
#     os.environ['TORCH_HOME'] = default_torch_home
#     os.makedirs(default_torch_home, exist_ok=True)
#     print(f"Setting default TORCH_HOME to: {default_torch_home}")


# this has to be imported after setting the env variables
from datasets import load_dataset

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('audio_augmentation.log')
        ]
    )
    return logging.getLogger(__name__)



def get_dataset_stats_fast(dataset_name, language_code, split="train", batch_size=1000):
    """
    Calculate dataset stats efficiently using batching.
    """
    # Load the dataset
    try:
        dataset = load_dataset(dataset_name, language_code, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Error with trust_remote_code, trying without: {str(e)}")
        dataset = load_dataset(dataset_name, language_code, split=split)
    
    # Get number of samples
    num_samples = len(dataset)
    
    # Check for speaker information
    has_speaker_id = "client_id" in dataset.column_names
    has_speaker_metadata = "speaker_id" in dataset.column_names
    
    # Calculate total duration and collect speakers using batched operations
    total_seconds = 0
    speakers = set()
    
    # Function to process a batch
    def process_batch(examples):
        nonlocal total_seconds
        
        # Efficient duration calculation for the whole batch
        for audio in examples["audio"]:
            duration = len(audio["array"]) / audio["sampling_rate"]
            total_seconds += duration
        
        # Collect speaker information
        if has_speaker_id:
            speakers.update(examples["client_id"])
        elif has_speaker_metadata:
            speakers.update(examples["speaker_id"])
        
        return None  # No need to return anything in this case
    
    # Process dataset in batches with progress bar
    dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        desc=f"Processing {language_code} {split}"
    )
    
    # Convert to hours
    total_hours = total_seconds / 3600
    
    # Get number of unique speakers
    num_speakers = len(speakers) if speakers else None
    
    return num_samples, total_hours, num_speakers

# Define languages and splits to analyze
languages = ["hsb", "lij", "ga-IE"]  # Upper Sorbian, Maltese, Irish Gaelic
splits = ["train", "validation", "test"]
dataset_name = "mozilla-foundation/common_voice_17_0"

# Create a results table
results = []

# Analyze each language and split
for language in languages:
    for split in splits:
        print(f"\nAnalyzing {language} {split}...")
        try:
            samples, hours, speakers = get_dataset_stats_fast(dataset_name, language, split)
            results.append({
                "Language": language,
                "Split": split,
                "Samples": samples,
                "Hours": round(hours, 2),
                "Minutes": round(hours * 60, 2),
                "Speakers": speakers
            })
            print(f"  - {samples} samples, {hours:.2f} hours, {speakers} speakers")
        except Exception as e:
            print(f"Error analyzing {language} {split}: {e}")
            results.append({
                "Language": language,
                "Split": split,
                "Samples": "Error",
                "Hours": "Error",
                "Minutes": "Error",
                "Speakers": "Error"
            })

# Create a DataFrame for better visualization
results_df = pd.DataFrame(results)

# Pivot the table for a better view
pivot_table = results_df.pivot(index="Language", columns="Split")

# Display the results
print("\nDataset Statistics Summary:")
print(results_df)

print("\nPivot Table View:")
print(pivot_table)

# Save results to CSV
results_df.to_csv("dataset_statistics.csv", index=False)
pivot_table.to_csv("dataset_statistics_pivot.csv")

print("\nResults have been saved to dataset_statistics.csv and dataset_statistics_pivot.csv")