from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from modules.text_classification.predictor import predict

app = FastAPI(
    title="MiniGrok Text Classification API",
    description="API for text classification into pre-defined categories using a fined-tuned DistilBERT model.",
    version="1.0.0",
)


class TextInput(BaseModel):
    """
    Model for the input text to be classified.
    """
    text: str


@app.get("/")
async def root() -> dict[str, str]:
    """
    Root endpoint for the API.
    Returns a simple message indicating the API is running.
    """
    return {"message": "Welcome to the MiniGrok Text Classification API!"}


@app.post("/predict", response_model=dict[str, str])
async def classify_text(input_data: TextInput):
    """
    Endpoint to classify the input text into one of the pre-defined categories.
    Args:
        input_data: The input data containing the text to classify.
    Returns:
        A dictionary containing the predicted category.
    """
    text = input_data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    try:
        prediction = predict(text)
        return {"predicted_category": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {repr(e)}")
