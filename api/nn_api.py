from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import gensim
import numpy as np
import uvicorn

# Define Model Input Schema
class EmojiInput(BaseModel):
    emoji: str

# Define Model Architecture
class SentimentModel(nn.Module):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.fc1 = nn.Linear(300, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for regression
        return x

# Load Emoji2Vec Model
try:
    emoji2vec = gensim.models.KeyedVectors.load_word2vec_format("emoji2vec/pre-trained/emoji2vec.bin", binary=True)
except FileNotFoundError:
    raise Exception("Error: 'emoji2vec.bin' file not found. Make sure it's in the correct path.")

# Load Trained Model
model = SentimentModel()
model.load_state_dict(torch.load("model/only_emoji_NN_200.pth", map_location=torch.device("cpu"), weights_only=True))
model.eval()  # Set to evaluation mode

app = FastAPI()

@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {"message": "Emoji Sentiment API is running!"}

def get_emoji_embedding(emoji):
    """Fetch Emoji2Vec embedding, return zero vector if not found."""
    try:
        return emoji2vec[emoji]
    except KeyError:
        return np.zeros((300,))  # Assuming 300-dimension vectors

@app.post("/predict/")
async def predict(data: EmojiInput):
    """Predict sentiment score based on emoji input."""
    emoji = data.emoji
    if not emoji:
        raise HTTPException(status_code=400, detail="Emoji input is required")

    embedding = np.array(get_emoji_embedding(emoji)).reshape(1, -1)
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)

    with torch.no_grad():  # Ensure no gradient computation
        sentiment_score = model(embedding_tensor).item()

    return {"emoji": emoji, "sentiment_score": sentiment_score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
