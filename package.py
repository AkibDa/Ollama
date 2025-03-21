import ollama

client = ollama.Client()

model = "llama3.2"

while(True):
  
  prompt = input("Enter your prompt here: ")

  if prompt == 'bye' or prompt == 'exit':
    break
  else:
    response = client.generate(model=model, prompt=prompt)

    print("Response from Ollama:")
    print(response.response)