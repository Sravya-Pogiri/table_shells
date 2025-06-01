import unstructured.partition
from unstructured.partition.pdf import partition_pdf
from pdfminer.pdfdocument import PDFDocument
import json 
from typing import List, Dict
from langchain_ollama.llms import OllamaLLM
import textwrap

model = OllamaLLM(model = "llama3.2")

def process_pdf(file_path: str, chunk_length: int = 500, chunk_overlap: int = 80) -> List[Dict]:
    elements = partition_pdf(filename = file_path)  # Partitions PDF into elements
    
    elements_dict = []
    for element in elements:
        elements_dict.append({
            "type": getattr(element, "category", ""), 
            "text": getattr(element, "text", ""), 
        })


    element_type = set()
    for element in elements_dict:
        element_type.add(element["type"])

    fullText = "\n\n".join([element["text"] for element in elements_dict if "text" in element])

    chunks = []
    start = 0 
    while start < len(fullText):
        end = min(start + chunk_length, len(fullText))
        chunks.append(fullText[start:end])
        start += chunk_length - chunk_overlap

    return chunks

def analyzeLLM(chunks: List[str], query: str) -> str:
    results = []
    for i, chunk in enumerate(chunks, 1):
        print(f"\nProcessing chunk {i}/{len(chunks)}...")
        if query:
            prompt = f"""Answer the following question based solely on the given text, discard any outside information, if there is no explicit mention, respond with DISCARD<VOID>" :
            Question: {query}
            Text: {chunk}
            Answer:"""
        else:
            prompt = f"""Analyze the given text and discard any outside information and only give the information found in the text, if there is no explicit mention, respond with DISCARD<VOID>:
            {chunk}
            Summary"""

        response = model.invoke(prompt)
        print(f"Chunk {i} is processed")
        results.append(response)
    return results 

def print_formatted(text: str, width: int = 80):
    for paragraph in text.split("\n"):
        print("\n".join(textwrap.wrap(paragraph, width=width)))
        print()

file_path = "./SAP.pdf"
pdf_chunks = process_pdf(file_path, chunk_length = 300, chunk_overlap = 80)

while True: 
    question = input("Please enter your question (type q to exit): ")
    if question== "q":
        break
    summary = analyzeLLM(pdf_chunks, question)

    new_summary = []
    for i in summary:
        if i != "DISCARD<VOID>":
            new_summary.append(i)

    summary_text = "\n".join(new_summary)
    answer = print_formatted(summary_text)







