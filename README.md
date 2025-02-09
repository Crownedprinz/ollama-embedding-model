# Ollama Embedding Model

This project demonstrates building a Retrieval Augmented Generation (RAG) application using Ollama and embedding models.

## Overview

The application uses the `chromadb` library to store documents in a vector embedding database and the `ollama` library to generate embeddings and responses.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ollama-embedding-model.git
    ```
2. Navigate to the project directory:
    ```sh
    cd ollama-embedding-model
    ```

3. Download and run the Ollama model:
    ```sh
    ollama download model-name
    ollama run model-name
    ```

## Experimentation

This project also includes experimentation with embedding models as described in [this blog post](https://ollama.com/blog/embedding-models). Specifically, the `mxbai-embed-large` model was used:

```sh
ollama pull mxbai-embed-large
```

