# MiniGrok

**MiniGrok** is a minimalist Retrieval-Augmented Generation (RAG) agent with modular components for query classification, retrieval, language generation, and (dummy) image generation. It is designed as a lightweight prototype for experimenting with multi-modal, multi-intent AI agents.

## Features

- Query Classification: Classifies input into:
  - `rag_retrieval`: factual queries answered via retrieval + generation.
  - `llm_generation`: open-ended text generation (e.g. storytelling).
  - `image_generation`: image generation from prompt (dummy model only).
- RAG Pipeline: semantic search over embedded documents (FAISS-based) + language generation with context.
- Text Generation: simple generation with a pretrained language model (e.g. DistilGPT2).
- Dummy Image Generation: returns a placeholder image (due to download failures with diffusion models).
- FastAPI Endpoint: exposes a `/query` POST route for interactive use.

## Installation

```bash
git clone https://github.com/yourusername/minigrok.git
cd minigrok
pip install -r requirements.txt
```

If you encounter issues downloading large models (e.g., `torch`, `transformers`, or `diffusers`), consider pre-downloading them manually, using a more robust internet connection, or disabling those modules. The image_generation module is set to the dummy model by default.

## Required Step – Train the Classifier

Before launching the MiniGrok agent, you **must train the query classifier**. Without it, the agent cannot route user inputs and will not work. From the root directory, run:

```bash
python modules/text_classification/src/run.py
```

This script:
- Tokenizes synthetic training queries for the supported classes.
- Fine-tunes a transformer classifier (e.g. DistilBERT).
- Saves the model and tokenizer under `modules/text_classification/model_weights`.

If this step is skipped, the API will raise an error when classifying queries.

## Running the API Server

Once the classifier is trained, start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

Then POST a query to:

```
http://localhost:8000/generate
```

Example request:

```json
{
  "text": "What is the theory of relativity?"
}
```

The response will vary depending on the classification result:
- For `rag_retrieval`, the system performs semantic retrieval + generation.
- For `llm_generation`, it produces creative output from a language model.
- For `image_generation`, it returns a placeholder image.

## What Works

- Classifier to route queries based on intent
- Fully functional RAG pipeline for text-based QA and storytelling
- Local FastAPI server with structured response handling
- Modular and easy to extend for other tasks

## What Does Not or Might Not Work

- Docker support was not fully tested due to persistent build issues (package conflicts, model downloads, network timeouts). The files are included; however.
- Image generation is currently a dummy module — model downloads from HuggingFace (e.g. Stable Diffusion) failed due to broken connections or incomplete files
- Not production-ready — meant as a self-contained prototype for experimentation

## License

MIT License

## Final Note

MiniGrok is not production-grade, but it is a working proof-of-concept showing how a single agent can combine retrieval, classification, generation, and multimodal behavior in a clean and extensible way. With more time and infrastructure, image generation and Docker support could be revisited, along with enhanced multi-modal capabilities.
