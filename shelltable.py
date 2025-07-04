import requests 
import json
import docx
import os
import pandas as pd
import json
from tabulate import tabulate
import signal
from contextlib import contextmanager
#from langchain_ollama.llms import OllamaLLM
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.response_synthesizers import CompactAndRefine 
from llama_index.core.prompts import PromptTemplate
from unstructured.partition.pdf import partition_pdf

def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except FileNotFoundError:
        print(f"Error: DOCX file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None
    
def read_document_content(file_path):
    """
    Reads text from a PDF or DOCX file based on its extension.
    Returns the extracted text as a string.
    """
    if file_path.lower().endswith('.docx'):   #.docx
        return extract_text_from_docx(file_path)
    else:
        print("Unsupported file format. Please provide a .pdf or .docx file.")
        return None
    
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.1", request_timeout=1000.0)

def timeout_context(seconds):
    """Context manager for timeout handling"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler and alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

if __name__ == '__main__':
    base_directory_path = "/Users/arnav/Documents/Python Projects/IDSWG_TableShells/table_shells"
    file_name = "Table_shell_standard.docx" # Make sure this matches your file

    file_path = os.path.join(base_directory_path, file_name)

    if not os.path.exists(file_path):
        print(f"Error: Document file not found at {file_path}. Please ensure the file exists and the 'file_name' variable is correct.")
        
    print(f"Reading document: {file_path}")
    document_content_text = read_document_content(file_path)
    if document_content_text:
        print("\n--- Document Content Extracted (partial view) ---")
        print(document_content_text[:500] + "..." if len(document_content_text) > 500 else document_content_text)
        print("--------------------------------------------------\n")

        print("--- Setting up RAG Pipeline ---")

        document_for_rag = Document(text=document_content_text)
        print("Created LlamaIndex Document object.")

        print("Creating VectorStoreIndex (this involves embedding the document)...")
        index = VectorStoreIndex.from_documents([document_for_rag], embed_model=Settings.embed_model)
        print("VectorStoreIndex created.")

        print("Creating QueryEngine...")
        
        # --- ADJUSTED CUSTOM_QA_TEMPLATE ---
        # Define the template string
        custom_qa_template_str = """Context information is below.
---------------------
{context_str}
---------------------
Given the context information, answer the query.

If the query asks for a general characteristic (e.g., "height", "weight", "ethnicity"), provide all associated metrics (e.g., 'n', 'Mean', 'Standard Deviation', 'Median', 'Min, Max' for height/weight) and their values across all relevant columns ('Treatment A', 'Treatment B', 'Total'), strictly in JSON format. Organize the JSON by the characteristic, then by metric, then by column.

If the query asks for a specific metric or cell value (e.g., "Mean height for Treatment A", "Q1 Sales for Product A"), provide only that specific value in JSON.

If you cannot find the information, return an empty JSON object {{}}.

Example JSON for a general characteristic query (e.g., "height"):
{{
  "Height (cm)": {{
    "n": {{ "Treatment A": "xx", "Treatment B": "xx", "Total": "xx" }},
    "Mean": {{ "Treatment A": "xxx.xx", "Treatment B": "xxx.xx", "Total": "xxx.xx" }},
    "Standard Deviation": {{ "Treatment A": "xxx.xxx", "Treatment B": "xxx.xxx", "Total": "xxx.xxx" }},
    "Median": {{ "Treatment A": "xxx.xx", "Treatment B": "xxx.xx", "Total": "xxx.xx" }},
    "Min, Max": {{ "Treatment A": "xxx, xxx", "Treatment B": "xxx, xxx", "Total": "xxx, xxx" }}
  }}
}}

Example JSON for a specific metric query: {{"Mean height for Treatment A": "xxx.xx"}}

Query: {query_str}
"""
        # Wrap the string in a PromptTemplate object
        custom_qa_template = PromptTemplate(custom_qa_template_str)
        
        query_engine = index.as_query_engine(
            llm=Settings.llm,
            response_synthesizer=CompactAndRefine(text_qa_template=custom_qa_template)
        )
        print("QueryEngine created.")

        # --- Example Queries (Adjusted for the new output expectation) ---
        queries = [
            "Give me the height table", # General characteristic query
        ]

        print("\n--- Running Queries with RAG ---")
        for i, query_text in enumerate(queries):
            print(f"--- Query {i+1}: {query_text} ---")
            
            try:
                response = query_engine.query(query_text)
                llm_response_text = response.response
                
                print("LLM Response (attempted JSON):")
                try:
                    parsed_json = json.loads(llm_response_text)
                    print(json.dumps(parsed_json, indent=2))
                except json.JSONDecodeError:
                    print("Could not parse as JSON. Raw response:")
                    print(llm_response_text)
                    
            except Exception as e:
                print(f"Error during query {i+1}: {e}")
            print("------------------------------------------\n")

