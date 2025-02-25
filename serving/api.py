from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

class Request(BaseModel):
    data: dict

def drop_columns(X):
    columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    return X[columns]

# Load pipeline
pipeline = joblib.load("fixed_pipeline.joblib")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def perdict(request: Request):

    try:
        if not isinstance(request.data, dict):
            raise ValueError("Invalid input data")

        # List of required columns (update this based on your model's requirements)
        required_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        if not all(col in request.data for col in required_columns):
            raise ValueError(f"Input data must contain the following columns: {required_columns}")

        input_data = pd.DataFrame([request.data])

        input_data = drop_columns(input_data)
        prediction = pipeline.predict(input_data)
    except Exception as e:
        return {"error": str(e)}

    return {"prediction": prediction[0]}