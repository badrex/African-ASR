# African-ASR

Develop Automatic Speech Recognition (ASR) Systems for African languages by fine-tuning multilingual speech models using CTC loss .  
This repository provides training scripts and configurations to build ASR models using [🤗 Hugging Face Transformers](https://huggingface.co/transformers/) and related libraries.

---

## 🚀 Features
- Training pipeline for ASR models
- Configurable YAML files for flexible experiments
- Language support for Kinyarwanda (initially) and extendable to other African languages
- Support for wav2vec-BERT-2.0 and other multilingual models (XLSR, MMS, etc.)
---

## 📦 Requirements

- Python 3.8+
- PyTorch (with GPU support recommended)
- Hugging Face Transformers, Datasets, and Tokenizers
- Other dependencies listed in `requirements.txt` 


Install the dependencies:
```bash
pip install -r requirements.txt
```


## ⚙️ Usage

1. Set up Hugging Face cache (optional): 
If you want to store downloaded models/datasets in a custom directory

```bash
export HF_HOME="/path/to/huggingface/cache"
```

2. Train the model

Run the training script with a configuration file:

```bash
python3 kinyarwanda-ASR/scripts/train_model.py \
    --config kinyarwanda-ASR/config_files/ASR_train_config_sample.yaml
```

3. In the configuration file, you should specifiy the base model (e.g., w2v-BERT-2.0), directory where the model will be saved, the training and validation datasets, as well as other hyperparameters such as the learning rate 

```yaml
# Project settings
project: "Kinyarwanda-ASR"
output_dir: "inprogress/kinyarwanda-ASR"
seed: 42

# Model settings
pretrained_model: "facebook/w2v-bert-2.0"  # or "facebook/mms-300m"  
freeze_feature_encoder: true

# Training settings
batch_size: 4
gradient_accumulation_steps: 8
num_epochs: 25
max_steps: 16000
learning_rate: 0.00005
warmup_ratio: 0.1
fp16: true
gradient_checkpointing: true
save_steps: 800
eval_steps: 800
logging_steps: 5
save_total_limit: 2

# Data settings
# if use_custom_dataset is true, then dataset_path is the path to the custom dataset on disk
# if use_custom_dataset is false, then dataset_path is the dataset repo name on the HF hub
use_custom_dataset: false
# if from HF hub, use the repo name such as badrex/kinyarwanda-speech-500h
dataset_path: "badrex/kinyarwanda-speech-1000h" 
train_split: "train"
eval_split: "validation"

# Data sampling settings (for debugging purposes)
sample: false
sample_size: 1800 

```

