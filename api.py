from fastapi import FastAPI
from pydantic import BaseModel

class Request(BaseModel):
    content: str


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def perdict(request: Request):
    return {"message": "Predict"}
