from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

model = pipeline(task="text-generation",
                 model="mistralai/Mistral-7B-Instruct-v0.2",
                 max_length = 256,
                 truncation=True,
                 )

llm = HuggingFacePipeline(pipeline=model)

template = PromptTemplate.from_template("Explain {topic} in detail for a {age} year old.")

chain = template | llm
topic = input("Topic: ")
age = input("Age: ")

response = chain.invoke({"topic": topic, "age": age})
print(response)