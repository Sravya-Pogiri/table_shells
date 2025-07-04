from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import unstructured.partition
from unstructured.partition.pdf import partition_pdf

from langchain_ollama.llms import OllamaLLM

model = SentenceTransformer('all-MiniLM-L6-v2')     # Model of chunking
llmModel = OllamaLLM(model = "llama3.2")            # Model for analyzing

def embed_chunks(chunks):
    embeddings = model.encode(chunks, convert_to_numpy = True)      # Converts to vector embeddings
    return embeddings       

def build(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve(query, chunks, embeddings, index, top: int = 1):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    queryVector = model.encode([query], convert_to_numpy = True)
    distance, ids = index.search(queryVector, top)
    goodChunks = [chunks[i] for i in ids[0]]
    return goodChunks

def analyzeLLM(chunks, query):
    embeddings = embed_chunks(chunks)
    index = build(embeddings)

    best_chunks = retrieve(query, chunks, embeddings, index, top = 1)

    results = []
    for chunk in best_chunks: 
        prompt = f"""Answer the question using the given chunk (no outside information)
        Question: {query}
        Text: {chunk}
        Answer:"""

        response = llmModel.invoke(prompt)
        results.append(response)
    return results

def process_pdf(file_path: str, chunk_length: int = 300, chunk_overlap: int = 80):
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

pdf_chunks = process_pdf("./SAP.pdf", chunk_length=300, chunk_overlap=80)
while True:
    demo_question = "What are the variables in the demographics section of the pdf file and clasiffy them as numerical or categorical"
    question = input("Please enter your question (press q to quit): ")
    if question == 'q':
        break 

    results = analyzeLLM(pdf_chunks, question)
    for answer in results: 
        print(answer)


