# gba-rag
This is a demonstration project to provide question answering over the website and documents of German [Federal Joint Committee (G-BA)](https://www.g-ba.de/). The G-BA is the highest decision-making body of the joint self-government of physicians, dentists, hospitals and health insurance funds in Germany.

This project uses the following technologies:
 - [LLamaIndex](https://www.llamaindex.ai/) framework
 - [Ollama](https://ollama.com/) to serve up an LLM (namely LLama3)
 - [Fastembed](https://github.com/qdrant/fastembed) to embed text with lightweight embedding models
 - [Qdrant](https://qdrant.tech/) vector store to store and retrieve embeddings
 
It also assumes you have a copy of the website.


## Deployment
### prepare the local environment
```bash
python -m venv .venv
```

### activate the local environment
```bash
source .venv/bin/activate
```

### Deploy the vector database

The `docker-compose.yml` file contains the basic setup for running the Qdrant vector store. You can modify the basic settings there and in the `qdrant_config.yml` file (including the password). Currently, you'll need to modify the notebook if you changed the default settings. 

```bash
cd vectordb
docker compose up -d
```

### Open and run the ipython notebook 
The `notebook/gba-llamaindex.ipynb` notebook contains the code to run through the demo project.

