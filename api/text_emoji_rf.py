from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import gensim
import pickle
from transformers import BertTokenizer, BertModel

# Define API app
app = FastAPI()

# Define Input Schema
class SentimentInput(BaseModel):
    text: str
    emoji: str

# Load Emoji2Vec Model
try:
    emoji2vec = gensim.models.KeyedVectors.load_word2vec_format("emoji2vec/pre-trained/emoji2vec.bin", binary=True)
except FileNotFoundError:
    raise Exception("Error: 'emoji2vec.bin' file not found. Make sure it's in the correct path.")

# Load BERT Model & Tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# Load Ensemble Model (Trained & Saved)
try:
    with open("model/emoji_sentiment_RFR.pkl", "rb") as f:
        ensemble_model = pickle.load(f)
except FileNotFoundError:
    raise Exception("Error: 'ensemble_model.pkl' file not found. Make sure it's in the correct path.")

def get_emoji_embedding(emoji):
    """Fetch Emoji2Vec embedding; return zero vector if not found."""
    try:
        return emoji2vec[emoji]
    except KeyError:
        return np.zeros((300,))  # Assuming 300-dimension vectors

def get_text_embedding(text):
    """Generate BERT embedding for input text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=50)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS Token Representation

@app.post("/predict/")
async def predict(data: SentimentInput):
    """Predict sentiment score using Emoji2Vec and BERT embeddings."""
    text_embedding = get_text_embedding(data.text)
    emoji_embedding = get_emoji_embedding(data.emoji)

    # Concatenate emoji and text embeddings
    combined_embedding = np.concatenate([text_embedding, emoji_embedding]).reshape(1, -1)

    # Make prediction using ensemble model
    sentiment_score = ensemble_model.predict(combined_embedding)[0]

    return {"text": data.text, "emoji": data.emoji, "sentiment_score": sentiment_score}

@app.get("/")
async def root():
    return {"message": "Emoji + Text Sentiment Analysis API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
