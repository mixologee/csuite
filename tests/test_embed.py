from langchain_ollama import OllamaEmbeddings
 
embedder = OllamaEmbeddings(model='nomic-embed-text')
result = embedder.embed_query('test embedding')
print(f'Embedding dimensions: {len(result)}')
