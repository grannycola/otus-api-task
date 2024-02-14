import pandas as pd
import numpy as np
import pickle
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from pydantic import BaseModel, Field, field_validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = FastAPI()
model_path = 'model.pkl'


def train_and_save_model(data_path, model_path):
    heart_df = pd.read_csv(data_path)
    X, y = heart_df.iloc[:, :-1], heart_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model = RandomForestClassifier().fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {acc}")

    with open(model_path, 'wb') as file:
        pickle.dump((model, acc), file)

    return model, acc


def load_model(model_path):
    with open(model_path, 'rb') as file:
        model, acc = pickle.load(file)
    return model, acc


if os.path.exists(model_path):
    print("Loading model...")
    model, acc = load_model(model_path)
else:
    print("Model training...")
    model, acc = train_and_save_model('data/raw/heart.csv', model_path)


class Item(BaseModel):
    features: list = Field(..., example=[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1])

    @field_validator('features')
    @classmethod
    def check_features_length(cls, v):
        if len(v) != 13:
            raise ValueError('Error message: The length of the features list must be 13')
        return v


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
    with open(html_file_path, 'r', encoding='utf-8') as html_file:
        return HTMLResponse(content=html_file.read())


@app.get("/model", response_class=JSONResponse)
async def get_model():
    """
    Get model information
    """
    return {"name": "Random Forest Classifier", "accuracy": acc}


@app.post("/predict/", summary="Make a prediction")
async def make_prediction(item: Item):
    """
    Make a prediction with the RandomForestClassifier model

    - **features**: a list of features required by the model
    """
    try:
        features = np.array(item.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
