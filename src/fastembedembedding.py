from typing import Any, List, Literal, Optional

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from fastembed import TextEmbedding

class FastEmbedEmbedding(BaseEmbedding):
    model_name: str = Field(
        "BAAI/bge-small-en-v1.5",
        description="Name of the FastEmbedding model to use.\n"
        "Defaults to 'BAAI/bge-small-en-v1.5'.\n"
        "Find the list of supported models at "
        "https://qdrant.github.io/fastembed/examples/Supported_Models/",
    )

    max_length: int = Field(
        512,
        description="The maximum number of tokens. Defaults to 512.\n"
        "Unknown behavior for values > 512.",
    )

    cache_dir: Optional[str] = Field(
        None,
        description="The path to the cache directory.\n"
        "Defaults to `local_cache` in the parent directory",
    )

    threads: Optional[int] = Field(
        None,
        description="The number of threads single onnxruntime session can use.\n"
        "Defaults to None",
    )

    doc_embed_type: Literal["default", "passage"] = Field(
        "default",
        description="Type of embedding method to use for documents.\n"
        "Available options are 'default' and 'passage'.",
    )

    _model: Any = PrivateAttr()

    @classmethod
    def class_name(self) -> str:
        return "FastEmbedEmbedding"

    def __init__(
        self,
        model_name: Optional[str] = "BAAI/bge-small-en-v1.5",
        max_length: Optional[int] = 512,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        doc_embed_type: Literal["default", "passage"] = "default",
        device: str = "cpu",  # Add device parameter with default "cpu"
    ):
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            threads=threads,
            doc_embed_type=doc_embed_type,
        )

        self._model = TextEmbedding(
            model_name=model_name,
            max_length=max_length,
            cache_dir=cache_dir,
            threads=threads,
            device=device,  # Pass the device parameter to TextEmbedding
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings: List[np.ndarray]
        if self.doc_embed_type == "passage":
            embeddings = list(self._model.passage_embed(text))
        else:
            embeddings = list(self._model.embed(text))
        return embeddings[0].tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        query_embeddings: np.ndarray = next(self._model.query_embed(query))
        return query_embeddings.tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)