#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio
import soundfile as sf
from tqdm import tqdm

def extract_audio_id(audio_path):
    # extract audio id from path 
    # for example, 'audio/1738233033-UzeE5AwhAVX6wUwsMru1Aymkyah2'
    # returns '1738233033'
    
    filename = Path(audio_path).name
    return filename.split('-')[0]

def load_json_data(json_path):
    """load and parse json file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_split(json_path, audio_dir, split_name):
    """process one data split"""
    data = load_json_data(json_path)
    processed_records = []
    missing_audio_ids = []
    
    print(f"\nprocessing {split_name} split...")
    for record_id, record in tqdm(data.items(), desc=f"{split_name}"):
        try:
            # extract audio id and build file path
            audio_id = extract_audio_id(record['audio_path'])
            audio_filename = f"{audio_id}-{record['voice_creator_id']}.wav"
            audio_file_path = os.path.join(audio_dir, audio_filename)
            
            # check if audio file exists
            if not os.path.exists(audio_file_path):
                missing_audio_ids.append(audio_id)
                continue
                
            # get audio duration
            try:
                audio_info = sf.info(audio_file_path)
                duration = audio_info.duration
            except:
                duration = record.get('duration', 0)
            
            # create record
            processed_record = {
                'audio_id': audio_id,
                'audio': audio_file_path,
                'audio_duration': duration,
                'gender': record['gender'],
                'age_group': record['age_group'],
                'category': record['image_category'],
                'transcription': record.get('transcription', '')
            }
            
            processed_records.append(processed_record)
            
        except Exception as e:
            print(f"error processing {record_id}: {e}")
            missing_audio_ids.append(extract_audio_id(record['audio_path']))
    
    # save missing audio ids
    if missing_audio_ids:
        with open(f'missing_audio_{split_name}.txt', 'w') as f:
            f.write('\n'.join(missing_audio_ids))
        print(f"{split_name}: {len(missing_audio_ids)} missing audio files saved to missing_audio_{split_name}.txt")
    
    print(f"{split_name}: processed {len(processed_records)} records")
    return processed_records

def create_hf_dataset():
    """create hugging face dataset"""
    metadata_dir = './metadata'
    audio_dir = './data/converted_audio'
    
    # process each split
    splits = {
        'train': process_split(f'{metadata_dir}/train.json', audio_dir, 'train'),
        'validation': process_split(f'{metadata_dir}/dev_test.json', audio_dir, 'validation'),
        'test': process_split(f'{metadata_dir}/test.json', audio_dir, 'test')
    }
    
    # create datasets
    dataset_dict = {}
    for split_name, records in splits.items():
        if records:
            dataset = Dataset.from_list(records)
            # cast audio column to Audio feature with 16kHz sampling rate
            dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
            dataset_dict[split_name] = dataset
    
    # create dataset dict
    hf_dataset = DatasetDict(dataset_dict)
    
    print(f"\ndataset created successfully!")
    print(f"train: {len(hf_dataset.get('train', []))} samples")
    print(f"validation: {len(hf_dataset.get('validation', []))} samples")
    print(f"test: {len(hf_dataset.get('test', []))} samples")
    
    return hf_dataset

if __name__ == "__main__":
    dataset = create_hf_dataset()
    
    # save dataset
    dataset.save_to_disk('./kinyarwanda_asr_dataset')
    print("dataset saved to ./kinyarwanda_asr_dataset")

