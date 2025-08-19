# 📝 Sentiment Analysis with Hugging Face Transformers

This repository contains examples of how to perform **sentiment analysis** using a pre-trained model from the Hugging Face Transformers library.  
We'll explore two main approaches: a **detailed, manual method** and a **streamlined, high-level pipeline method**.

---

## 1️⃣ Manual Approach

The manual approach gives you **full control** over the process, allowing you to see each step of the classification.  
It's great for understanding **exactly what's happening under the hood**.

### Key Steps:

- **Loading the Model and Tokenizer**  
  The `AutoTokenizer` and `AutoModelForSequenceClassification` classes automatically load the correct tokenizer and model .  
  The tokenizer converts your text into a format the model can understand (numerical tokens).

- **Preparing the Input**  
  The tokenizer processes the raw text.

- **Getting Predictions**  
  The tokenized input is passed to the model. The output contains logits, which are raw, unnormalized scores for each sentiment class (e.g., positive, negative, neutral).

- **Calculating Probabilities**  
  To turn logits into a meaningful probability distribution, we apply the softmax function. Softmax ensures all scores are between 0 and 1 and sum to 1, effectively giving the probability for each sentiment.

- **Interpreting the Results**  
  We use the model's configuration (`config.id2label`) to map the numerical index of the output back to its corresponding label (e.g., "negative", "neutral", "positive").
  
The pipeline reproduces the **three core steps** of the manual approach:  
1. Preprocessing with tokenizers  
2. Passing inputs through the model  
3. Postprocessing the output  
---

## 2️⃣ Pipeline Approach

The most basic object in the 🤗 Transformers library is the **`pipeline()`** function.  
It connects a model with all the necessary preprocessing and postprocessing steps, allowing us to **directly input text** and get an intelligible answer.


---

## 📝 Which Approach to Use?

- **Manual Approach**  
  Use it if you need **fine-grained control**, want to integrate the model into a larger system, or want to **learn how each component works**.

- **Pipeline Approach**  
  Use it for **quick prototyping**, simple applications, or when you just want results **without extra code**.
