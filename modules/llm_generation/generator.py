import logging

from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from modules.config import LLM_GENERATION_MODEL_NAME as MODEL_NAME
from modules.config import RAG_CHUNKS_FILE as CHUNKS_FILE
from modules.config import RAG_EMBEDDING_MODEL_NAME as EMBEDDING_MODEL_NAME
from modules.config import RAG_INDEX_FILE as INDEX_FILE
from modules.llm_generation.model_loader import load_model
from modules.rag_retrieval.retriever import DocumentRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def generate_answer(
        *,
        query: str,
        query_type: str,
        llm_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        retriever: DocumentRetriever | None,
        k: int = 3,
        print_answer: bool = False,
) -> dict[str, str | list[dict]]:
    """
    Generates an answer to a query using the provided LLM model.

    The prompt and generation configuration are adapted based on the query type. Moreover, the RAG retriever is called
    to provide additional context to the prompt if the type is 'rag_retrieval'.

    Args:
        query: The query for the LLM model.
        query_type: Label assigned to the query by the text_classification module.
        llm_model: The LLM model to generate the answer.
        tokenizer: The tokenizer for the query.
        retriever: The retriever from the rag_retrieval module to gather context from the document corpus.
        k: The number of most relevant chunks to retrieved for the query.
        print_answer: If True, the query and generated answer are displayed on console.

    Returns:
        A dictionary with the generated answer and corresponding retrieved chunks, if any.
    """
    if query_type == "rag_retrieval":
        # Retrieve top-k relevant chunks based on query. Use the context to create a prompt for the LLM.
        retrieved_chunks = retriever.retrieve(query, k=k)
        context = "\n".join([chunk["text"] for chunk in retrieved_chunks])
        prompt = (
            "You are a helpful assistant that answers questions based on context.\n\n"
            f"Context: \n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        generation_args = {
            "max_new_tokens": 200,
            "num_beams": 8,
            "early_stopping": False,
        }
    else:
        retrieved_chunks = None
        prompt = f"Write a creative and short response to this prompt: \n{query}"
        generation_args = {
            "max_new_tokens": 150,
            "repetition_penalty": 1.3,
            "length_penalty": 1.2,
            "num_beams": 8,
            "early_stopping": True,
        }

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
        **generation_args,
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
    # Example usage of the LLM generation module with and without RAG retrieval.
    model, tokenizer = load_model(model_name=MODEL_NAME)
    retriever = DocumentRetriever(
        index_file=INDEX_FILE,
        chunks_file=CHUNKS_FILE,
        embedder_model_name=EMBEDDING_MODEL_NAME,
    )
    queries = [
        ("What is a neural network?", "rag_retrieval"),
        ("Tell me a riddle.", "llm_generation"),
    ]
    for query, qtype in queries:
        answer = generate_answer(
            query=query,
            query_type=qtype,
            retriever=retriever,
            llm_model=model,
            tokenizer=tokenizer,
            k=3,
            print_answer=True,
        )
