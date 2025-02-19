import ollama

client = ollama.Client()

model = "llama2"
prompt = input("Enter your prompt here: ")

response = client.generate(model=model, prompt=prompt)

print("Response from Ollama:")
print(response.response)