Working with Transformers Library 
The most basic object in the 🤗 Transformers library is the pipeline() function. It connects a model with its necessary preprocessing and postprocessing steps, 
allowing us to directly input any text and get an intelligible answer
Behind the pipeline:
Transformer models can’t process raw text directly, so the first step of our pipeline is to convert the text inputs into numbers that the model can make sense of.To do this, we use the AutoTokenizer class and
its from_pretrained() method.
We can download our pretrained model the same way we did with our tokenizer. 🤗 Transformers provides an AutoModel class which also has a from_pretrained() method
