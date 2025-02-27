from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

summarization_pipeline = pipeline(task="summarization",model="facebook/bart-large-cnn",)
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

refinement_pipeline = pipeline(task="summarization",model="facebook/bart-large",)
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

qa_pipeline = pipeline(task="question-answering", model="deepset/roberta-base-squad2",)

summery_template = PromptTemplate.from_template("Summarize the following text in a {length} way:\n\n{text}")

summarization_chain = summery_template | summarizer | refiner

text_to_summarize = input("\nEnter text to summarize:\n")

length = input("\nEnter the length (short/medium/long): ")
length_map = {"short": 50, "medium": 150, "long": 300}
max_length = length_map.get(length.lower(),150)

summary = summarization_chain.invoke({"text": text_to_summarize, "length": max_length})

print("\n 🔹 **Genrated Summary:**")
print(summary)