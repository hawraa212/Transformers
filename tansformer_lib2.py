# Import necessary classes from Hugging Face Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoConfig
import torch
# Specify the model name (Hugging Face Hub ID)
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load the tokenizer , model for sequence classification,configuration
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# Prepare the input text for classification
text = "I love programming with Python!"
inputs = tokenizer(text, return_tensors="pt")#return_tensors="pt" converts the input to PyTorch tensors

# get predictions no need to compute gradients because we are not training the model
with torch.no_grad():
    output = model(**inputs)

# Extract logits are the score 
scores = output.logits[0]
probabilities = torch.softmax(scores, dim=0) #use softmax to make scores between 0 and 1

# Print the probability for each label defined in the model config
for i, probability in enumerate(probabilities):
    label = config.id2label[i]
    print(f"{i+1} {label}: {probability:.4f}")
