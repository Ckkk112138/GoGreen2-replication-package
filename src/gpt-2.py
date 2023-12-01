from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd
import numpy as np
import time

''' SST-2 Dataset '''
data = pd.read_csv("./SST-2/train.tsv", sep='\t')

sentence = data['sentence'].to_numpy()

# Load GPT-2 Model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Initilize sentiment labels
sentiment_labels = []

for sen in sentence:
    # Generate through GPT-2
    input_ids = gpt2_tokenizer.encode(sen, return_tensors="pt")
    generated_text = gpt2_model.generate(
        input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = gpt2_tokenizer.decode(generated_text[0])

    # Find label in generated text
    if "positive" in generated_text:
        sentiment_labels.append(1)
    else:
        sentiment_labels.append(0)

# Output classification result
for i, label in enumerate(sentiment_labels):
    print(f"Sentence {i + 1}: Sentiment Label {label}")


# Load DistilGPT-2 Model
from transformers import AutoTokenizer, AutoModelForCausalLM

distil_gpt2_tokenizer = AutoTokenizer.from_pretrained("abhinema/distillgpt2")
distil_gpt2_model = AutoModelForCausalLM.from_pretrained("abhinema/distillgpt2")

# Initilize sentiment labels
sentiment_labels = []

for sen in sentence:
    # Generate through DistilGPT-2
    input_ids = distil_gpt2_tokenizer.encode(
        sen, return_tensors="pt", padding=True, truncation=True, max_length=100)
    generated_text = distil_gpt2_model.generate(
        input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = distil_gpt2_tokenizer.decode(generated_text[0])

    # Find label in generated text
    if "positive" in generated_text:
        sentiment_labels.append("positive")
    else:
        sentiment_labels.append("negative")

# Output classification result
for i, label in enumerate(sentiment_labels):
    print(f"SST-2 Sentence {i + 1}: Sentiment Label {label}")

time.sleep(60)

''' MNLI Dataset '''
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

distil_gpt2_tokenizer = AutoTokenizer.from_pretrained("abhinema/distillgpt2")
distil_gpt2_model = AutoModelForCausalLM.from_pretrained("abhinema/distillgpt2")

# Load MNLI dataset
dataset = load_dataset("multi_nli")

# Load GPT-2 Model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2ForSequenceClassification.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = "<pad>"

distil_gpt2_tokenizer.pad_token = "<pad>"

# Run all examples in MNLI dataset
for example in dataset["train"]:
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label = example["label"]

    # Inference through GPT-2
    inputs = gpt2_tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True)
    outputs = gpt2_model(**inputs)
    predicted_label_gpt2 = outputs.logits.argmax().item()
    
    # Inference through DistilGPT-2
    inputs = distil_gpt2_tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True)
    outputs = distil_gpt2_model(**inputs)
    predicted_label_distil_gpt2 = outputs.logits.argmax().item()

    # Export result
    print(f"Premise: {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"True Label: {label}")
    print(f"GPT-2 Predicted Label: {predicted_label_gpt2}")
    print(f"DistilGPT-2 Predicted Label: {predicted_label_distil_gpt2}")
    print("=" * 50)

time.sleep(60)

''' STS-B Dataset '''
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from datasets import load_dataset

dataset = load_dataset("glue", "stsb")

# Load GPT-2 Model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = "<pad>"
gpt2_model = GPT2ForSequenceClassification.from_pretrained("gpt2")

# Load DistilGPT-2 Model and tokenizer
distil_gpt2_tokenizer.pad_token = "<pad>"

# Run all examplse in STS-B dataset
for example in dataset["train"]:
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]
    similarity_score = example["label"]

    # Inference through GPT-2
    inputs = gpt2_tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)
    outputs = gpt2_model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    predicted_similarity_score_gpt2 = predicted_class
    
    # Inference through DistilGPT-2
    inputs = distil_gpt2_tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)
    outputs = distil_gpt2_model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    predicted_similarity_score_distil_gpt2 = predicted_class

    # Export result
    print(f"Sentence 1: {sentence1}")
    print(f"Sentence 2: {sentence2}")
    print(f"True Similarity Score: {similarity_score}")
    print(f"GPT-2 Predicted Similarity Score: {predicted_similarity_score_gpt2}")
    print(f"DistilGPT-2 Predicted Similarity Score: {predicted_similarity_score_distil_gpt2}")
    print("=" * 50)
