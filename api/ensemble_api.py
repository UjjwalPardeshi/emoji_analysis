from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import gensim
import numpy as np
import uvicorn

# Define Model Input Schema
class EmojiInput(BaseModel):
    emoji: str

# Load Emoji2Vec Model
try:
    emoji2vec = gensim.models.KeyedVectors.load_word2vec_format("emoji2vec/pre-trained/emoji2vec.bin", binary=True)
except FileNotFoundError:
    raise Exception("Error: 'emoji2vec.bin' file not found. Make sure it's in the correct path.")

# Load Trained Ensemble Model
try:
    ensemble_model = joblib.load("model/stacking_ensemble.pkl")  # Load ensemble model
except FileNotFoundError:
    raise Exception("Error: 'ensemble_model.pkl' file not found. Make sure it's in the correct path.")

app = FastAPI()

@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {"message": "Ensemble Emoji Sentiment API is running!"}

def get_emoji_embedding(emoji):
    """Fetch Emoji2Vec embedding, return zero vector if not found."""
    try:
        return emoji2vec[emoji]
    except KeyError:
        return np.zeros((300,))  # Assuming 300-dimension vectors

@app.post("/predict/")
async def predict(data: EmojiInput):
    """Predict sentiment score based on emoji input using the ensemble model."""
    emoji = data.emoji
    if not emoji:
        raise HTTPException(status_code=400, detail="Emoji input is required")
    
    embedding = np.array(get_emoji_embedding(emoji)).reshape(1, -1)
    
    sentiment_score = ensemble_model.predict(embedding)[0]  # Predict using ensemble model
    
    return {"emoji": emoji, "sentiment_score": sentiment_score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
