from fastapi import FastAPI
import joblib
import os
from pydantic import BaseModel

app = FastAPI() # app instance from fastapi
@app.get("/")
def root():
 return "hello"

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

unsupervised_dir = os.path.join(parent_dir, "Unsupervised Learning")

# load the unsupervised models
kmeans = joblib.load(os.path.join(unsupervised_dir, "kmeans_model.joblib"))
scaler = joblib.load(os.path.join(unsupervised_dir, "scaler.joblib"))

class InputFeatures(BaseModel):
 appearance: float
 goals: float
 assists: float
 minutes_played: float

def preprocessing(input_features: InputFeatures):
 dict_f = {
 'appearance': input_features.appearance,
 'goals': input_features.goals,
 'assists': input_features.assists,
 'minutes_played': input_features.minutes_played,
 }
 # Convert dictionary values to a list in the correct order
 features_list = [dict_f[key] for key in sorted(dict_f)]
 # Scale the input features
 scaled_features = scaler.transform([list(dict_f.values
 ())])
 return scaled_features

@app.get("/predict")
def predict(input_features: InputFeatures):
  return preprocessing(input_features)

@app.post("/predict")
async def predict(input_features: InputFeatures):
 data = preprocessing(input_features)
 y_pred = kmeans.predict(data)
 return {"predition": y_pred.tolist()[0]}