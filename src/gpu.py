import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')



from typing import List
import numpy as np
from fastembed import TextEmbedding

embedding_model_gpu = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5", 
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
print(embedding_model_gpu.model.model.get_providers())

documents: List[str] = list(np.repeat("Demonstrating GPU acceleration in fastembed", 500))
embedding_model_cpu = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

import timeit
time = timeit.timeit(lambda: list(embedding_model_gpu.embed(documents)),number=1)
print(f'gpu: {time}') 


time = timeit.timeit(lambda: list(embedding_model_cpu.embed(documents)),number=1)
print(f'cpu: {time}') 