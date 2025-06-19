#!/usr/bin/env python3

import torch
from transformers import AutoProcessor, AutoModelForCTC
from datasets import load_dataset, Audio
import jiwer
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import evaluate
import re

def normalize_arabic_text(text):
    """
    Arabic text normalization:
    1. Remove punctuation
    2. Remove diacritics
    3. Eastern Arabic numerals to Western Arabic numerals

    Arguments
    ---------
    text: str
        text to normalize
    Output
    ---------
    normalized text
    """
    # Remove punctuation
    #punctuation = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟\,\?\.\!\-\;\:\"\"\%\'\"\�\']') #r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،؛؟]'

    CHARS_TO_IGNORE = [
        ",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", 
        "ʿ", "·", "჻", "~", "՞", "؟", "،", "।", "॥", "«", "»", "„", "“", "”", 
        "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]", "{", "}", "=", "`",
        "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", 
        "→", "。", "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）",
        "［", "］", "【", "】", "‥", "〽", "『", "』", "〝", "〟", "⟨", "⟩", "〜",
        "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "'", "ʻ", "ˆ"
    ]

    chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"


    text = re.sub(chars_to_ignore_regex, '', text)

    # Remove diacritics
    diacritics = re.compile(r'[\u064B-\u0652]')  # Arabic diacritical marks (Fatha, Damma, etc.)
    text = re.sub(diacritics, '', text)

    # Remove latin characters
    #text = re.sub(r'[a-zA-Z]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Normalize Hamzas and Maddas
    text = re.sub('پ', 'ب', text)
    text = re.sub('ڤ', 'ف', text)
    text = re.sub(r'[آ]', 'ا', text)
    text = re.sub(r'[أإ]', 'ا', text)
    text = re.sub(r'[ؤ]', 'ء', text)
    text = re.sub(r'[ئ]', 'ء', text)
    # text = re.sub(r'[ء]', '', text)   
    text = re.sub(r'[ى]', 'ي', text)
    text = re.sub(r'[ة]', 'ه', text)
 
    # Replace the phrase "غيرواضح" with empty string
    #text = text.replace("غيرواضح", "")

    # Transliterate Eastern Arabic numerals to Western Arabic numerals
    eastern_to_western_numerals = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for eastern, western in eastern_to_western_numerals.items():
        text = text.replace(eastern, western)

    # pattern = re.compile(r'[\,\?\.\!\-\;\:\"\"\%\'\"\�\']')

    # text = re.sub(pattern, '', text)

    return text.strip()


def load_model(model_path):
    """Load the trained ASR model and processor from local path"""
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCTC.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, processor, device


def transcribe_audio(audio_array, model, processor, device):
    """Transcribe a single audio array"""
    # Ensure audio is at 16kHz sampling rate
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits

    generated_ids = torch.argmax(logits, dim=-1)

    #generated_ids.label_ids[generated_ids.label_ids == -100] = processor.tokenizer.pad_token_id

    
    transcription = processor.batch_decode(generated_ids)[0] #, skip_special_tokens=True
    return transcription.strip()


def evaluate_split(dataset_split, model, processor, device, split_name, text_column):

    """Evaluate a dataset split and calculate WER/CER"""
    print(f"\nProcessing {split_name} split ({len(dataset_split)} samples)...")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    predictions = []
    references = []

    # resample dataset to 16kHz if needed
    if dataset_split.features['audio'].sampling_rate != 16000:
        print(f"Resampling {split_name} split to 16kHz...")

        dataset_split = dataset_split.cast_column(
            "audio", Audio(sampling_rate=16_000)
        )
    
    for sample in tqdm(dataset_split, desc=f"Transcribing {split_name}"):
        # Get audio array (should be at 16kHz)
        audio_array = sample['audio']['array']
        
        # Transcribe
        pred_text = normalize_arabic_text(
            transcribe_audio(audio_array, model, processor, device)
        )
        
        # Use cleaned_text as reference
        ref_text = normalize_arabic_text(sample[text_column])

        print(f"Reference: {ref_text}")
        print(f"Prediction: {pred_text}")

        
        predictions.append(pred_text)
        references.append(ref_text)



    # print total number of predictions and references
    print(f"Total predictions: {len(predictions)}, Total references: {len(references)}")

    for i in range(len(references)):
        if not references[i]:
            print(f"Empty Reference at index {i}: {references[i]}")
            print(f"Prediction: {predictions[i]}")


    # Filter out empty predictions and references
    filtered_predictions = []
    filtered_references = []

    for pred, ref in zip(predictions, references):
        if ref:
            filtered_predictions.append(pred)
            filtered_references.append(ref)
            #print(f"Reference: {ref}")
            #print(f"Prediction: {pred}")

    # print total number of filtered predictions and references
    print(f"Filtered predictions: {len(filtered_predictions)}, Filtered references: {len(filtered_references)}")

    
    # Calculate WER and CER
    #wer = jiwer.wer(filtered_references, filtered_predictions)
    #cer = jiwer.cer(filtered_references, filtered_predictions)

    wer = wer_metric.compute(predictions=filtered_predictions, references=filtered_references)
    cer = cer_metric.compute(predictions=filtered_predictions, references=filtered_references)
    
    print(f"{split_name} Results:")
    print(f"  WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"  CER: {cer:.4f} ({cer*100:.2f}%)")
    
    return {
        'split': split_name,
        'wer': wer,
        'cer': cer,
        'samples': len(dataset_split),
        'predictions': predictions,
        'references': references
    }


def main():
    # Configuration
    #MODEL_PATH = "./outputs/baseline/mms-300m-arabic-22052025-045030"  # Update this path
    #MODEL_PATH = "./outputs/baseline/facebook/mms-300m-arabic-23052025-094428/checkpoint-42000/"
    #MODEL_PATH = "./outputs/current_best_model/"  
    #MODEL_PATH = "./outputs/baseline/facebook/wav2vec2-xls-r-300m-arabic-26052025-123311/checkpoint-33000"  con
    #MODEL_PATH = "outputs/baseline/facebook/wav2vec2-xls-r-300m-arabic-29052025-230536"  # Update this path
    #MODEL_PATH = "./outputs/baseline/facebook/mms-300m-arabic-26052025-221445"  # Update this path
    # MODEL_PATH = "./outputs/baseline/facebook/mms-1b-arabic-01062025-094043" 
    #MODEL_PATH = "./outputs/baseline/facebook/mms-300m-arabic-02062025-023818"  # MMS 300M No freeze
    MODEL_PATH = "./outputs/baseline/facebook/mms-300m-arabic-26052025-221445"  # MMS 300M frozen
    #MODEL_PATH = "./outputs/baseline/facebook/w2v-bert-2.0-arabic-04062025-002805"  # w2v-bert-2.0


    DATASET_NAME = "SADA22"
    #DATASET_NAME = "CommonVoice"  # Uncomment to use Common Voice dataset
    #DATASET_NAME = "Casablanca" 

    print("Loading dataset...")
    if DATASET_NAME == "SADA22":
        dataset = load_dataset("MohamedRashad/SADA22")
        text_column = 'cleaned_text'  # Use cleaned_text for SADA22

    elif DATASET_NAME == "CommonVoice":
        dataset = load_dataset("mozilla-foundation/common_voice_17_0", 'ar')
        text_column = 'sentence'  # Use sentence for Common Voice

    elif DATASET_NAME == "Casablanca":
        dataset = load_dataset("UBC-NLP/Casablanca", "UAE")
        text_column = 'transcription'
    
    print("Loading trained model...")
    model, processor, device = load_model(MODEL_PATH)
    print(f"Model loaded on device: {device}")
    
    results = {}
    
    # Evaluate validation split (if exists)
    if 'validation' in dataset:
        results['validation'] = evaluate_split(
            dataset['validation'], model, processor, device, 'validation', text_column
        )
    
    # Evaluate test split (if exists)
    if 'test' in dataset:
        results['test'] = evaluate_split(
            dataset['test'], model, processor, device, 'test', text_column
        )
    
    # If no validation/test splits, evaluate a subset of train split
    if not results:
        print("No validation/test splits found. Evaluating subset of train split...")
        train_subset = dataset['train'].select(range(min(1000, len(dataset['train']))))
        results['train_subset'] = evaluate_split(
            train_subset, model, processor, device, 'train_subset'
        )
    
    # Summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    summary_data = []
    for split_name, result in results.items():
        summary_data.append({
            'Split': split_name,
            'Samples': result['samples'],
            'WER': f"{result['wer']:.4f}",
            'WER (%)': f"{result['wer']*100:.2f}%",
            'CER': f"{result['cer']:.4f}",
            'CER (%)': f"{result['cer']*100:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save detailed results
    for split_name, result in results.items():
        results_df = pd.DataFrame({
            'reference': result['references'],
            'prediction': result['predictions']
        })
        results_df.to_csv(f"{split_name}_results.tsv", sep='\t', index=False)
        print(f"\nDetailed results saved to {split_name}_results.csv")

if __name__ == "__main__":
    main()