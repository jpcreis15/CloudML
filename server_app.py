import pickle
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
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

# Health check service
@app.get("/healthcheck")
async def health():
    """
    Health Check service
    """

    return {"message":"all good"}

# Inference service
@app.post("/inference")
async def predict(input: list):
    """
    Inference Service
    """

    features = list(input)
    print("features: ", features)

    pred = model.predict([features])
    print("prediction: ", pred)

    return json.dumps({"result": pred[0]})