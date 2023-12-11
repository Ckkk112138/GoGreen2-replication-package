from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd
import numpy as np
import time
from datasets import load_dataset
import sys

# Record the start time
start_time = time.time()

data = pd.read_csv("./SST-2/dev.tsv", sep='\t')

sentence = data['sentence'].to_numpy()
sentence = sentence[:len(sentence) // 2]

# Load the GPT-2 model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Store sentiment classification results in a list
sentiment_labels = []

for sen in sentence:
    # Use GPT-2 to generate text
    input_ids = gpt2_tokenizer.encode(sen, return_tensors="pt")
    generated_text = gpt2_model.generate(
        input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = gpt2_tokenizer.decode(generated_text[0])

    # Look for sentiment labels in the generated text
    if "positive" in generated_text:
        sentiment_labels.append(1)
    else:
        sentiment_labels.append(0)

# Record the end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Record the start time
start_time = time.time()

# Load the STS-B dataset
dataset = load_dataset("glue", "stsb")
# Initialize the GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = "<pad>"
gpt2_model = GPT2ForSequenceClassification.from_pretrained("gpt2")

# Loop through examples in the STS-B dataset
for example in dataset["train"]:
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]
    similarity_score = example["label"]

    # Use GPT-2 for inference
    inputs = gpt2_tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)
    outputs = gpt2_model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    predicted_similarity_score_gpt2 = predicted_class

# Record the end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Record the start time
start_time = time.time()

# Load the MNLI dataset
dataset = load_dataset("multi_nli", split="validation_matched")
desired_fraction = 0.8
split = dataset.train_test_split(test_size=desired_fraction)

# Get the "train" subset with the specified fraction
dataset = split["train"]

# Initialize the GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2ForSequenceClassification.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = "<pad>"

# Loop through examples in the MNLI dataset
for example in dataset:
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label = example["label"]

    # Use GPT-2 for inference
    inputs = gpt2_tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True)
    outputs = gpt2_model(**inputs)
    predicted_label_gpt2 = outputs.logits.argmax().item()

# Record the end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Exit the script
sys.exit()
