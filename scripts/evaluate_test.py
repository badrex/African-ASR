#!/usr/bin/env python3

import torch
from transformers import AutoProcessor, AutoModelForCTC
from datasets import load_from_disk, Audio
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from typing import Tuple
import logging

def setup_logging():
    """configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

# type alias for cleaner signature
ModelComponents = Tuple[AutoModelForCTC, AutoProcessor, torch.device]

def load_model(model_path: str) -> ModelComponents:
    """load the trained ASR model and processor from local path"""
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCTC.from_pretrained(model_path)
    
    # move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, processor, device

def transcribe_audio(audio_array: Audio, 
                     model: AutoModelForCTC, 
                     processor: AutoProcessor, 
                     device: torch.device) -> str:
    """transcribe a single audio array"""
    # make sure audio is at 16kHz sampling rate
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    
    # move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(
        predicted_ids, 
        skip_special_tokens=True
    )

    return transcription[0].strip()

def run_inference(dataset_path: str, 
                  model_path: str,
                  metadata_file: str, 
                  output_file: str):
    
    """run inference on test split and save predictions to CSV"""
    logging.info("Starting inference...")

    # load dataset
    logging.info(f"Loading dataset {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    # load metadata
    logging.info(f"Loading metadata from {metadata_file}...")
    try:
        with open(metadata_file, 'r') as f:
            metadata_dict = pd.read_json(f)
        logging.info(f"Metadata loaded with {len(metadata_dict)} entries.")
    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        return
    logging.info("Metadata loaded successfully.")

    # get mapping from audio_id to json key
    audio_id_to_key = {}
    for key, value in metadata_dict.items():
        audio_id = value['audio_path'].split('/')[-1].split('-')[0]  # extract timestamp
        audio_id_to_key[audio_id] = key
    
    print("Loading trained model...")
    model, processor, device = load_model(model_path)
    print(f"Model loaded on device: {device}")
    
    # get test split
    test_dataset = dataset['test'] #.select(range(10))
    print(f"Processing test split. {len(test_dataset)} samples found.")
    
    # resample dataset to 16kHz if needed
    # if test_dataset.features['audio'].sampling_rate != 16000:
    #     print("Resampling test split to 16kHz...")
    #     test_dataset = test_dataset.cast_column(
    #         "audio", Audio(sampling_rate=16_000)
    #     )
    
    audio_ids =  []
    predictions = []

    # print first sample for debugging
    print(f"First sample audio ID: {test_dataset[0]}")
    
    for i, sample in enumerate(tqdm(test_dataset, desc="Transcribing test samples")):
        # get audio array (should be at 16kHz)
        audio_array = sample['audio']['array']
        audio_id = audio_id_to_key.get(sample['audio_id'], "unknown")
        
        # transcribe
        pred_text = transcribe_audio(audio_array, model, processor, device)
        
        print(f"{i} -- Audio ID {audio_id}: {pred_text}")
        audio_ids.append(audio_id)
        predictions.append(pred_text)
    
    # save predictions to CSV
    df = pd.DataFrame({
        'id': audio_ids,
        'transcription': predictions
    })
    df.to_csv(output_file, index=False, header=True)

    print(f"\nPredictions saved to {output_file}")
    print(f"Total predictions: {len(predictions)}")

def main():
    # setup logging
    setup_logging()
    logging.info("Starting inference script...")

    # configuration
    model_path = "./inprogress/baseline/facebook/w2v-bert-2.0-22062025-210026" # "./inprogress/current_best/checkpoint-6800/"
    dataset_path = "./kinyarwanda-ASR/kinyarwanda_asr_dataset"
    metadata_file = "./kinyarwanda-ASR/metadata/test.json"
    output_file = "./kinyarwanda-ASR/predictions_2.csv"

    run_inference(dataset_path, model_path, metadata_file, output_file)

if __name__ == "__main__":
    main()


"""
# create mapping from audio_id to json key
audio_id_to_key = {}
for key, value in metadata.items():
    audio_id = value['audio_path'].split('/')[-1].split('-')[0]  # extract timestamp
    audio_id_to_key[audio_id] = key

# in your loop:
json_key = audio_id_to_key.get(sample['audio_id'], "unknown")
"""