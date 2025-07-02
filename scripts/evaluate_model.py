#!/usr/bin/env python3

import os
import torch
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2ProcessorWithLM
from pyctcdecode import build_ctcdecoder
from datasets import load_from_disk, load_dataset, Audio, Dataset
import jiwer
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import evaluate
import re
from typing import Dict, List, Any, Tuple
import logging
import argparse


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def process_text(text: str) -> str:
    """
    Text normalization:
    1. Remove punctuation, except Kinayarwanda's apostrophe (')
    2. Remove digits
    3. Trim leading/trailing whitespace
    """

    # keep only allowed characters
    allowed = "abcdefghijklmnopqrstuvwxyz \'"
    text = ''.join(c for c in text if c.lower() in allowed)
    
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text.lower().strip()

# type alias for cleaner signature
ModelComponents = Tuple[AutoModelForCTC, AutoProcessor, torch.device]

def load_model(model_path: str) -> ModelComponents:
    
    """Load the trained ASR model and processor from local path"""
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCTC.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, processor, device


def transcribe_audio(audio_array: Audio, 
                     model: AutoModelForCTC, 
                     processor: AutoProcessor, 
                     device: torch.device) -> str:
    
    """Transcribe a single audio array"""
    # make sure audio is at 16kHz sampling rate
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(
        predicted_ids, 
        skip_special_tokens=True # this should ignore special tokens like [PAD]
    )

    return transcription[0].strip()


def evaluate_split(eval_dataset: Dataset, 
                   model: AutoModelForCTC, 
                   processor: AutoProcessor, 
                   device: torch.device,
                   text_column: str='transcription', 
                   split_name='validation') -> Dict[str, Any]:

    """Evaluate a dataset split and calculate WER/CER"""

    # check if eval_dataset is empty
    if not eval_dataset:
        logging.error(f"No data found for in the eval dataset. "
              f"Please check the dataset path and the split name.")
        return {}

    logging.info(f"Processing {len(eval_dataset)} samples...")

    # initialize metrics
    logging.info(f"Initializing metrics...")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    predictions = []
    references = []

    # resample dataset to 16kHz if needed
    if eval_dataset.features['audio'].sampling_rate != 16000:
        logging.info(f"Resampling speech to 16kHz...")

        eval_dataset = eval_dataset.cast_column(
            "audio", Audio(sampling_rate=16_000)
        )
    
    for i, sample in enumerate(tqdm(eval_dataset, desc=f"Transcribing speech.")):
        # Get audio array (should be at 16kHz)
        audio_array = sample['audio']['array']
        
        # Transcribe
        pred_text = transcribe_audio(audio_array, model, processor, device)

        # Use cleaned_text as reference
        ref_text = process_text(sample[text_column])

        print("-"*75)
        print(f"{i}  Reference: {ref_text}")
        print(f"{i} Prediction: {pred_text}")

        predictions.append(pred_text)
        references.append(ref_text)

    # print total number of predictions and references
    print(f"Total predictions: {len(predictions)}, "
          f"Total references: {len(references)}")

    # check for empty references. Empty references can be problematic for the metrics
    print(f"Checking for empty references...")
    for i in range(len(references)):
        if not references[i]:
            logging.warning(f"Empty Reference at index {i}: {references[i]}")
            logging.warning(f"Prediction: {predictions[i]}")


    # Filter out empty predictions and references
    filtered_predictions = []
    filtered_references = []

    for pred, ref in zip(predictions, references):
        if ref:
            filtered_predictions.append(pred)
            filtered_references.append(ref)
            # print(f"Reference: {ref}")
            # print(f"Prediction: {pred}")

    # print total number of filtered predictions and references
    logging.info(f"Filtered predictions: {len(filtered_predictions)}, "
          f"Filtered references: {len(filtered_references)}")

    # compute metrics
    wer = wer_metric.compute(
        predictions=filtered_predictions, 
        references=filtered_references
    )

    cer = cer_metric.compute(
        predictions=filtered_predictions, 
        references=filtered_references
    )

    error_rate = (0.4 * wer) + (0.6 * cer)
    score = (1 - error_rate) * 100

    print(f"{split_name} Results:")
    print(f"  WER: {wer:.4f} ({wer*100:.4f}%)")
    print(f"  CER: {cer:.4f} ({cer*100:.4f}%)")
    print(f"  Score: {score:.4f}%")

    
    return {
        'split': split_name,
        'wer': wer,
        'cer': cer,
        'score': score,
        'samples': len(eval_dataset),
        'predictions': predictions,
        'references': references
    }


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned ASR model with language model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the dataset directory (e.g., kinyarwanda_asr_dataset)')
    parser.add_argument('--split', type=str, required=True,
                       help='Split to evaluate on (e.g., validation, test)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.info("Starting evaluation script...")

    # Configuration
    model_path = args.model_path
    dataset_path = args.dataset_path
    split = args.split
    
    logging.info(f"Loading trained model from {model_path}...")
    model, processor, device = load_model(model_path)
    logging.info(f"Model loaded on device: {device}")


    text_column = 'transcription' 

    logging.info(f"Loading dataset {dataset_path}/{split}...")

    # check if dataset_path is a local path or a HF hub path
    # check if dataset_path is a local path
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path + f"/{split}")
    else:
        dataset = load_dataset(dataset_path, split=split)

    results = {}
    
    results[split] = evaluate_split(
        dataset, model, processor, device, text_column, split
    )

    
    # make a summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    summary_data = []
    for split_name, result in results.items():
        summary_data.append({
            'Split': split_name,
            'Samples': result['samples'],
            'WER (%)': f"{result['wer']*100:.4f}%",
            'CER (%)': f"{result['cer']*100:.4f}%",
            'Score (%)': f"{result['score']:.4f}%",
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    

if __name__ == "__main__":
    main()
