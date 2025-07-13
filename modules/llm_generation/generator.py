import logging

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from modules.config import LLM_GENERATION_MODEL_NAME as GENERATION_MODEL_NAME
from modules.config import RAG_CHUNKS_FILE as CHUNKS_FILE
from modules.config import RAG_EMBEDDING_MODEL_NAME as EMBEDDING_MODEL_NAME
from modules.config import RAG_INDEX_FILE as INDEX_FILE
from modules.rag_retrieval.retriever import DocumentRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def generate_answer(
        *,
        query: str,
        retriever: DocumentRetriever,
        llm_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        k: int = 3,
        print_answer: bool = False,
) -> dict[str, str | list[dict]]:
    # Retrieve top-k relevant chunks based on query. Use the context to create a prompt for the LLM.
    retrieved_chunks = retriever.retrieve(query, k=k)
    context = "\n".join([chunk["text"] for chunk in retrieved_chunks])
    prompt = (
        "You are a helpful assistant that answers questions based on context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    # Tokenize the prompt and generate an answer using the LLM.
    encodings = tokenizer(
        prompt,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    output = llm_model.generate(
        input_ids=encodings["input_ids"],
        attention_mask=encodings["attention_mask"],
        max_new_tokens=200,
        num_beams=8,
        early_stopping=False,
    )
    # Decode the generated output and return the answer.
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    if print_answer:
        logging.info(
            f"Query: {query}\n"
            f"Generated answer: {answer}"
        )

    return {"answer": answer, "retrieved_chunks": retrieved_chunks}


if __name__ == "__main__":
    # Example usage of the LLM generation module with RAG retrieval.
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # set pad token for generation compatibility
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)
    retriever = DocumentRetriever(
        index_file=INDEX_FILE,
        chunks_file=CHUNKS_FILE,
        embedder_model_name=EMBEDDING_MODEL_NAME,
    )
    query = "What is a neural network?"
    answer = generate_answer(
        query=query,
        retriever=retriever,
        llm_model=model,
        tokenizer=tokenizer,
        k=3,
        print_answer=True
    )
