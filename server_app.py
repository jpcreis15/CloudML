import pickle
from sklearn.tree import DecisionTreeClassifier
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request, Form
import json

# Opening the file might also be once the inference endpoint is called
# Model trained using the Iris dataset
with open("./model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthcheck")
async def health():
    return {"message":"all good"}

@app.post("/inference")
async def predict(input: list):
    features = list(input)
    
    print("features: ", features)

    pred = model.predict([features])
    print("prediction: ", pred)
    return json.dumps({"result": pred[0]})