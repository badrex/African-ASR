#!/usr/bin/env python3

import torch
from transformers import AutoProcessor, AutoModelForCTC
from datasets import load_from_disk, Audio, Dataset
import jiwer
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import evaluate
import re
from typing import Dict, List, Any, Tuple
import logging

def setup_logging():
    """configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def process_text(text: str) -> str:
    """
    text normalization:
    1. remove punctuation, except Kinayarwanda's apostrophe (')
    2. remove digits
    3. trim leading/trailing whitespace
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

    # get predicted token IDs
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # debug: print token info
    print(f"predicted_ids shape: {predicted_ids.shape}")
    print(f"first 20 predicted_ids: {predicted_ids[0][:20].tolist()}")
    print(f"unique predicted_ids: {torch.unique(predicted_ids[0]).tolist()}")
    
    # for character-based CTC models, we need to:
    # 1. remove padding tokens (29)
    # 2. remove blank tokens (0 = "|")
    # 3. remove consecutive duplicates (CTC collapse)
    
    predicted_ids = predicted_ids[0]  # remove batch dimension
    
    # remove padding tokens
    predicted_ids = predicted_ids[predicted_ids != 29]  # [PAD] token
    
    # CTC decoding: remove consecutive duplicates and blank tokens
    decoded_ids = []
    prev_id = None
    
    for token_id in predicted_ids:
        token_id = token_id.item()
        # skip blank token ("|" = 0) and consecutive duplicates
        if token_id != 0 and token_id != prev_id:
            decoded_ids.append(token_id)
        prev_id = token_id
    
    print(f"after CTC decoding: {decoded_ids}")
    
    # convert to characters and join
    vocab = processor.tokenizer.get_vocab()
    id_to_char = {v: k for k, v in vocab.items()}
    
    characters = []
    for token_id in decoded_ids:
        if token_id in id_to_char:
            char = id_to_char[token_id]
            characters.append(char)
    
    transcription = ''.join(characters)
    print(f"final transcription: '{transcription}'")
    
    return transcription.strip()

def debug_tokenizer(processor):
    """debug tokenizer information"""
    print("=== TOKENIZER DEBUG ===")
    print(f"tokenizer type: {type(processor.tokenizer)}")
    if hasattr(processor.tokenizer, 'vocab_size'):
        print(f"vocab size: {processor.tokenizer.vocab_size}")
    if hasattr(processor.tokenizer, 'pad_token_id'):
        print(f"pad_token_id: {processor.tokenizer.pad_token_id}")
    if hasattr(processor.tokenizer, 'get_vocab'):
        vocab = processor.tokenizer.get_vocab()
        print(f"sample vocab (first 10): {list(vocab.keys())[:10]}")
        # check for pad tokens
        pad_tokens = [k for k, v in vocab.items() if 'pad' in k.lower()]
        print(f"pad-related tokens: {pad_tokens[:5]}")
    print("=====================")

def evaluate_split(dataset: Dataset, 
                   model: AutoModelForCTC, 
                   processor: AutoProcessor, 
                   device: torch.device, 
                   split_name: str, 
                   text_column: str='transcription') -> Dict[str, Any]:
    """evaluate a dataset split and calculate WER/CER"""
    eval_dataset = dataset[split_name]

    if not eval_dataset:
        print(f"no data found for {split_name} split.")
        return {}

    print(f"processing {split_name} split. "
          f"{len(eval_dataset)} samples were found.")

    # debug tokenizer on first run
    if split_name == 'validation':
        debug_tokenizer(processor)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    predictions = []
    references = []

    # resample dataset to 16kHz if needed
    if eval_dataset.features['audio'].sampling_rate != 16000:
        print(f"resampling {split_name} split to 16kHz...")
        eval_dataset = eval_dataset.cast_column(
            "audio", Audio(sampling_rate=16_000)
        )
    
    # limit to first few samples for debugging
    eval_subset = eval_dataset.select(range(min(5, len(eval_dataset))))
    
    for i, sample in enumerate(tqdm(eval_subset, desc=f"transcribing {split_name}")):
        print(f"\n--- sample {i+1} ---")
        
        # get audio array (should be at 16kHz)
        audio_array = sample['audio']['array']
        print(f"audio shape: {len(audio_array)}")
        
        # transcribe
        pred_text = process_text(
            transcribe_audio(audio_array, model, processor, device)
        )
        
        # use cleaned_text as reference
        ref_text = process_text(sample[text_column])

        print(f"reference: {ref_text}")
        print(f"prediction: {pred_text}")

        predictions.append(pred_text)
        references.append(ref_text)

    # filter out empty predictions and references
    filtered_predictions = []
    filtered_references = []

    for pred, ref in zip(predictions, references):
        if ref:
            filtered_predictions.append(pred)
            filtered_references.append(ref)

    print(f"filtered predictions: {len(filtered_predictions)}, filtered references: {len(filtered_references)}")

    if not filtered_predictions:
        print("no valid predictions found!")
        return {
            'split': split_name,
            'wer': 1.0,
            'cer': 1.0,
            'score': 0.0,
            'samples': len(eval_dataset),
            'predictions': predictions,
            'references': references
        }

    wer = wer_metric.compute(predictions=filtered_predictions, references=filtered_references)
    cer = cer_metric.compute(predictions=filtered_predictions, references=filtered_references)

    error_rate = (0.4 * wer) + (0.6 * cer)
    score = (1 - error_rate) * 100

    print(f"{split_name} results:")
    print(f"  WER: {wer:.4f} ({wer*100:.4f}%)")
    print(f"  CER: {cer:.4f} ({cer*100:.4f}%)")
    print(f"  score: {score:.4f}%")

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
    # setup logging
    setup_logging()
    logging.info("starting evaluation script...")

    # configuration
    model_path = "./inprogress/baseline/facebook/w2v-bert-2.0-20062025-203510/checkpoint-2400"  
    dataset_path = "./kinyarwanda-ASR/kinyarwanda_asr_dataset"
    text_column = 'transcription' 

    logging.info(f"loading dataset {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    print("loading trained model...")
    model, processor, device = load_model(model_path)
    print(f"model loaded on device: {device}")
    
    results = {}
    
    results['validation'] = evaluate_split(
        dataset, model, processor, device, 'validation', text_column
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