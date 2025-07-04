import requests
import json
import docx
import pypdf
import os
import pandas as pd
from tabulate import tabulate
import signal
from contextlib import contextmanager

# LlamaIndex core imports for persistence
from llama_index.core import StorageContext, load_index_from_storage

# LlamaIndex and Unstructured imports
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.prompts import PromptTemplate
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx # Import for DOCX

def extract_text_from_pdf_simple(file_path):
    """
    Extracts text from a PDF file page by page.
    Note: This is a simple text extraction and might not preserve complex layouts
    or table structures as effectively as 'unstructured'.
    """
    text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n" # Add newline between pages
        return text
    except FileNotFoundError:
        print(f"Error: PDF file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

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
    Reads text from a PDF or DOCX file based on its extension using unstructured.
    Returns the extracted text as a string.
    """
    elements = []
    try:
        if file_path.lower().endswith('.docx'):
            print("Using unstructured.partition_docx for DOCX file...")
            elements = partition_docx(
                filename=file_path,
                infer_table_structure=True,
            )
        elif file_path.lower().endswith('.pdf'):
            print("Using unstructured.partition_pdf for PDF file...")
            elements = partition_pdf(
                filename=file_path,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000,
            )
        else:
            print("Unsupported file format. Please provide a .pdf or .docx file.")
            return None

        full_text = "\n\n".join([str(el.text) for el in elements if el.text])
        return full_text

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error extracting content from {file_path}: {e}")
        return None

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="mistral", request_timeout=200.0, format="json")

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout handling"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

if __name__ == '__main__':
    base_directory_path = "/Users/Sravya/Desktop/AI_model_table_shells"
    file_name = "Table shell standard.pdf" # Make sure this matches your file
    file_path = os.path.join(base_directory_path, file_name)

    # --- Index Storage Setup ---
    PERSIST_DIR = "./storage" # Define the directory to store the index

    index = None # Initialize index variable

    if not os.path.exists(file_path):
        print(f"Error: Document file not found at {file_path}. Please ensure the file exists and the 'file_name' variable is correct.")
        exit() # Exit if source document not found

    # Check if the index already exists in storage
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        print("Index not found in storage. Creating a new index...")
        document_content_text = read_document_content(file_path)

        if not document_content_text:
            print("No content extracted from the document. Exiting RAG pipeline setup.")
            exit() # Exit if no documents were loaded for indexing

        print(f"\n--- Document Content Extracted (partial view) ---")
        print(document_content_text[:500] + "..." if len(document_content_text) > 500 else document_content_text)
        print("--------------------------------------------------\n")

        print("Creating VectorStoreIndex (this involves embedding the document)...")
        document_for_rag = Document(text=document_content_text)
        index = VectorStoreIndex.from_documents([document_for_rag], embed_model=Settings.embed_model)
        print("VectorStoreIndex created. Persisting index to disk...")
        index.storage_context.persist(persist_dir=PERSIST_DIR) # Persist the index
        print(f"Index persisted to {PERSIST_DIR}")
    else:
        print(f"Loading index from {PERSIST_DIR}...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context) # Load the index from storage
        print("Index loaded.")

    print("Creating QueryEngine...")

    custom_qa_template_str = """Context information is below.
---------------------
{context_str}
---------------------
Given the context information, answer the query.

**IMPORTANT: Your response MUST be valid JSON and contain ONLY the JSON object. Do NOT include any conversational text, explanations, or markdown fences (```json).**

If the query asks for a general characteristic (e.g., "height", "weight", "ethnicity"), provide all associated metrics (e.g., 'n', 'Mean', 'Standard Deviation', 'Median', 'Min, Max' for height/weight) and their values across all relevant columns ('Treatment A', 'Treatment B', 'Total'), strictly in JSON format. Organize the JSON by the characteristic, then by metric, then by column.

If the query asks for a specific metric or cell value (e.g., "Mean height for Treatment A", "Q1 Sales for Product A"), provide only that specific value in JSON.

If you cannot find the information, return an empty JSON object {{}}.

Example JSON for a single general characteristic query (e.g., "height"):
{{
  "Height (cm)": {{
    "n": {{ "Treatment A": "xx", "Treatment B": "xx", "Total": "xx" }},
    "Mean": {{ "Treatment A": "xxx.xx", "Treatment B": "xxx.xx", "Total": "xxx.xx" }},
    "Standard Deviation": {{ "Treatment A": "xxx.xxx", "Treatment B": "xxx.xxx", "Total": "xxx.xxx" }},
    "Median": {{ "Treatment A": "xxx.xx", "Treatment B": "xxx.xx", "Total": "xxx.xx" }},
    "Min, Max": {{ "Treatment A": "xxx, xxx", "Treatment B": "xxx, xxx", "Total": "xxx, xxx" }}
  }}
}}

Example JSON for multiple general characteristics query (e.g., "height and weight"):
{{
  "Height (cm)": {{
    "n": {{ "Treatment A": "xx", "Treatment B": "xx", "Total": "xx" }},
    "Mean": {{ "Treatment A": "xxx.xx", "Treatment B": "xxx.xx", "Total": "xxx.xx" }}
  }},
  "Weight (kg)": {{
    "n": {{ "Treatment A": "xx", "Treatment B": "xx", "Total": "xx" }},
    "Mean": {{ "Treatment A": "xxx.xx", "Treatment B": "xxx.xx", "Total": "xxx.xx" }}
  }}
}}

Example JSON for a specific metric query: {"Mean height for Treatment A": "xxx.xx"}

Query: {query_str}
"""
    custom_qa_template = PromptTemplate(custom_qa_template_str)

    query_engine = index.as_query_engine(
        llm=Settings.llm,
        response_synthesizer=CompactAndRefine(text_qa_template=custom_qa_template),
        similarity_top_k=2
    )
    print("QueryEngine created.")

    queries = [
        "Give me the height table", # Query that should return multiple characteristics

    ]

    print("\n--- Running Queries with RAG ---")
    for i, query_text in enumerate(queries):
        print(f"--- Query {i+1}: {query_text} ---")

        try:
            with timeout_context(600):
                response = query_engine.query(query_text)
                llm_response_text = response.response

            llm_response_text = llm_response_text.strip()
            if llm_response_text.startswith("```json") and llm_response_text.endswith("```"):
                llm_response_text = llm_response_text[len("```json"):-len("```")].strip()

            try:
                parsed_json = json.loads(llm_response_text)

                if isinstance(parsed_json, dict) and len(parsed_json) > 0:
                    processed_at_least_one_item = False
                    for characteristic_name, characteristic_data in parsed_json.items():
                        if isinstance(characteristic_data, dict) and \
                           any(isinstance(v, dict) for v in characteristic_data.values()):
                            try:
                                df = pd.DataFrame(characteristic_data).T
                                df.index.name = "Treatment / Statistic"
                                df.columns.name = characteristic_name

                                print(f"\n--- {characteristic_name} Table ---")
                                print(tabulate(df, headers='keys', tablefmt='pipe', showindex=True))
                                print(f"-----------------------------------")
                                processed_at_least_one_item = True
                            except Exception as df_error:
                                print(f"Could not format '{characteristic_name}' as a table due to error: {df_error}")
                        elif isinstance(characteristic_data, (str, int, float, bool, type(None))):
                            print(f"\n--- Specific Metric or Simple JSON ---")
                            print(json.dumps(parsed_json, indent=2))
                            print(f"-----------------------------------")
                            processed_at_least_one_item = True
                            break
                        else:
                            print(f"\n--- Unrecognized but valid JSON structure for '{characteristic_name}' ---")
                            print(json.dumps({characteristic_name: characteristic_data}, indent=2))
                            print(f"-----------------------------------")
                            processed_at_least_one_item = True

                    if not processed_at_least_one_item:
                         print("LLM returned valid JSON but no recognized table or specific metric format.")
                         print(json.dumps(parsed_json, indent=2))
                else:
                    print("LLM returned an empty or unexpected JSON format.")
                    print(llm_response_text)

            except json.JSONDecodeError as e:
                print(f"Could not parse as JSON. Raw response:\n{llm_response_text}\nError: {e}")

        except TimeoutError as te:
            print(f"Query timed out: {te}")
        except Exception as e:
            print(f"Error during query {i+1}: {e}")
        print("------------------------------------------\n")