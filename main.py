from transformers import pipeline
# import torch

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

model = pipeline(task="summarization", model="facebook/bart-large-cnn")
reponse = model("text to summarize")
print(reponse)