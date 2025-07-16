from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from modules.config import IMAGE_GENERATION_IMAGES_DIR as IMAGES_DIR
from modules.config import LLM_GENERATION_MODEL_NAME as LLM_MODEL_NAME
from modules.config import RAG_CHUNKS_FILE as CHUNKS_FILE
from modules.config import RAG_EMBEDDING_MODEL_NAME as EMBEDDING_MODEL_NAME
from modules.config import RAG_INDEX_FILE as INDEX_FILE
from modules.config import TEXT_CLASSIFICATION_MODEL_NAME as TXT_MODEL_NAME
from modules.config import TEXT_CLASSIFICATION_MODEL_WEIGHTS_FILE as TXT_MODEL_PATH
from modules.image_generation.generator import generate_image
from modules.llm_generation.generator import generate_answer
from modules.llm_generation.model_loader import load_model as load_llm_model
from modules.rag_retrieval.retriever import DocumentRetriever
from modules.text_classification.model_loader import load_model as load_class_model
from modules.text_classification.predictor import predict_query_class

# Load the text classification model and tokenizer.
classifier, cls_tokenizer = load_class_model(
    model_name=TXT_MODEL_NAME,
    model_path=TXT_MODEL_PATH,
    num_labels=3,
    device="cpu",
)
# Load the large language model and tokenizer.
llm, llm_tokenizer = load_llm_model(
    model_name=LLM_MODEL_NAME,
)
# Load the diffusion/dummy model.
img_gen_model = "dummy"
# Load the RAG retriever.
rag_retriever = DocumentRetriever(INDEX_FILE, CHUNKS_FILE, EMBEDDING_MODEL_NAME)

# Construct the API.
app = FastAPI(
    title="MiniGrok assistant API.",
    description="API for querying MiniGrok.",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    """
    Model for the input query to be classified and later answered by the appropriate module.
    """
    text: str


class GenerateResponse(BaseModel):
    """
    Model for the generated response containing the requested data: text answer, context-informed text answer,
    or path to image produced.
    """
    query_type: str
    answer: str | None = None
    image_path: str | None = None
    retrieved_chunks: list[dict] | None = None


@app.get("/")
async def root() -> dict[str, str]:
    """
    Root endpoint for the API.
    Returns a simple message indicating the API is running.
    """
    return {"message": "Welcome to the MiniGrok Assistant API!"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(input_query: GenerateRequest):
    """
    Endpoint to process the query and generate the appropriate answer.

    First, the query is classified into 'llm_generation', 'image_generation', or 'rag_retrieval'.
    Thereafter, the corresponding module is called to process the query and generate the correct response.

    Args:
        input_query: The input data containing the text to classify and process.
    Returns:
        A GeneratedResponse instance with the information related to the model output.
    """
    logger.info("You are now generating!")
    text = input_query.text.strip()
    logger.info(f"Received text: {text}")
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # Determine type of query.
        query_type = predict_query_class(text=text, model=classifier, tokenizer=cls_tokenizer)
        # Select appropriate generator and configuration.
        if query_type in ["rag_retrieval", "llm_generation"]:
            retriever = rag_retriever if query_type == "rag_retrieval" else None
            generator = generate_answer
            generator_args = {
                "query": text,
                "query_type": query_type,
                "llm_model": llm,
                "tokenizer": llm_tokenizer,
                "retriever": retriever,
                "k": 1,
                "print_answer": False,
            }
        elif query_type == "image_generation":
            generator = generate_image
            generator_args = {
                "model": img_gen_model,
                "prompt": text,
                "output_dir": IMAGES_DIR,
            }
        # Get the model output.
        output = generator(**generator_args)
        base_response = {"query_type": query_type}
        base_response.update(output)
        base_response["image_path"] = str(base_response.get("image_path", ""))

        # Create the response and return it.
        response = GenerateResponse(**base_response)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {repr(e)}")
