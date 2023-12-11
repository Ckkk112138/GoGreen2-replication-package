import random

# List of model names
models = ["BERT", "DistilBERT", "GPT", "DistilGPT"]

# Randomly pick a model
selected_model = random.choice(models)

print(f"The randomly selected model is: {selected_model}")