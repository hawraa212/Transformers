# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
text = " I love programming with Python! "
result = pipe(text)
print(result)  # Display the result of the sentiment analysis