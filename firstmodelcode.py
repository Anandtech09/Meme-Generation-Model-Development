!pip install transformers datasets
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

import pandas as pd
from datasets import Dataset
import numpy as np

# Load the dataset
df = pd.read_csv('meme_custom_dataset.csv') # create your own dataset for meme creation model dvelopment

# For simplicity, assume 'Full Text' column contains the meme text
captions_list = df['Full Text'].tolist()

# Create a Hugging Face Dataset
dataset = Dataset.from_dict({"text": captions_list})

# Split dataset into train and validation
dataset = dataset.train_test_split(test_size=0.1)


from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Tokenize the data with labels
def tokenize_function_with_labels(examples):
    tokenized_text = tokenizer(examples['text'], padding='max_length', truncation=True)
    tokenized_text["labels"] = tokenized_text["input_ids"].copy()
    return tokenized_text

tokenized_datasets = dataset.map(tokenize_function_with_labels, batched=True)


from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

model = GPT2LMHeadModel.from_pretrained('gpt2')

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',  # Update this to avoid the deprecation warning
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

trainer.train()


model.save_pretrained("./meme-generator-model")
tokenizer.save_pretrained("./meme-generator-tokenizer")

# Load the trained model
model = GPT2LMHeadModel.from_pretrained('./meme-generator-model')
tokenizer = GPT2Tokenizer.from_pretrained('./meme-generator-tokenizer')

# Generate meme text
input_text = "When you realize"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

generated_texts = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
for i, text in enumerate(generated_texts):
    print(f"Meme {i+1}: {text}")