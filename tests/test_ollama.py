from langchain_ollama import OllamaLLM
 
llm = OllamaLLM(model='qwen2.5:32b-instruct-q4_K_M')
response = llm.invoke('Reply with only the words: connection confirmed')
print(response)
