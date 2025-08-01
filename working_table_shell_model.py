import os
import pandas as pd

# --- Docling Imports ---
try:
    from docling.document_converter import DocumentConverter
except ImportError as e:
    print(f"Error importing Docling components: {e}")
    print("Please install Docling: pip install 'docling[all]'")
    exit()

# --- LlamaIndex and LLM Imports (with PromptTemplate) ---
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def parse_clinical_data(table_df: pd.DataFrame):
    """
    This robust parsing function remains the same.
    """
    # 1. Sanitize column names
    sanitized_columns = []
    for i, col in enumerate(table_df.columns):
        col_str = str(col).strip()
        if not col_str:
            sanitized_columns.append(f"unnamed_col_{i}")
        else:
            sanitized_columns.append(col_str)
    table_df.columns = sanitized_columns

    # Use sanitized names for access
    header_col = sanitized_columns[0]
    char_col = sanitized_columns[1]
    val_cols = sanitized_columns[2:]

    records = table_df.to_dict('records')
    extracted_tables = {}
    current_table_name = None
    current_table_rows = []
    final_column_names = []

    for row in records:
        section_title = str(row[header_col]).strip()

        if section_title:
            if current_table_name and current_table_rows:
                df = pd.DataFrame(current_table_rows, columns=final_column_names)
                extracted_tables[current_table_name] = df

            current_table_name = section_title
            current_table_rows = []
            final_column_names = [str(row[char_col]).strip() or "Characteristic"] + [str(row[c]) for c in val_cols]

        elif current_table_name:
            characteristic = str(row[char_col]).strip()
            if characteristic:
                data_values = [characteristic] + [str(row[c]) for c in val_cols]
                current_table_rows.append(data_values)

    if current_table_name and current_table_rows:
        df = pd.DataFrame(current_table_rows, columns=final_column_names)
        extracted_tables[current_table_name] = df

    return extracted_tables


if __name__ == "__main__":
    file_path = "/Users/Sravya/Desktop/AI_model_table_shells/mark - can you convert this into a markdown.csv"
    PERSIST_DIR = "./rag_storage"

    # --- STEPS 1 & 2 remain the same ---
    print("🔬 Parsing document to extract structured tables...")
    try:
        doc_converter = DocumentConverter()
        conv_result = doc_converter.convert(file_path)
        doc = conv_result.document
        main_table_df = doc.tables[0].export_to_dataframe()
        parsed_tables = parse_clinical_data(main_table_df)
        print(f"✅ Parser finished. Found {len(parsed_tables)} sub-tables.")
    except Exception as e:
        print(f"❌ Critical error during parsing: {e}")
        exit()

    print("\n⚙️ Configuring RAG settings (LLM and Embedding Model)...")
    Settings.llm = Ollama(model="mistral", request_timeout=1000.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    
    # --- STEP 3 remains the same ---
    index = None
    if not os.path.exists(PERSIST_DIR):
        print(f"\n⚡ Index not found. Creating new index in '{PERSIST_DIR}'...")
        documents = []
        for table_name, df in parsed_tables.items():
            table_markdown = df.to_markdown(index=False)
            doc = Document(
                text=f"This document contains the table shell for '{table_name}'.\n\n{table_markdown}",
                metadata={"table_name": table_name}
            )
            documents.append(doc)
        
        print(f"📄 Converted {len(documents)} tables into Documents.")
        print("🧠 Building vector index...")
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("💾 Index created and saved.")
    else:
        print(f"\n📂 Found existing index in '{PERSIST_DIR}'. Loading...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("✅ Index loaded.")

    # --- STEP 4: Create Query Engine with the FINAL PROMPT TEMPLATE ---
    if index:
        # ⭐ --- START OF KEY CHANGE --- ⭐
        
        # This new prompt instructs the LLM to return the full table shell.
        qa_prompt_template_str = (
            "You are a table extraction assistant.\n"
            "The user will ask for a specific table. Your task is to find the corresponding table in the context and return **only the full, raw markdown content of that table shell.**\n"
            "Do not add any introductory text like 'Here is the table...'. Do not add any summary or explanation after the table. Your entire response should be just the markdown table.\n"
            "---------------------\n"
            "CONTEXT INFORMATION:\n{context_str}\n"
            "---------------------\n"
            "USER QUESTION: {query_str}\n"
            "ASSISTANT'S RESPONSE (full markdown table only):\n"
        )
        
        qa_template = PromptTemplate(qa_prompt_template_str)

        # Create the query engine with our custom prompt
        query_engine = index.as_query_engine(
            text_qa_template=qa_template
        )
        
        # ⭐ --- END OF KEY CHANGE --- ⭐
        
        # Queries are now more direct requests for the tables themselves.
        rag_queries = [
            "Show me the table shell for Ethnicity [for US studies only]",
            "Give me the Height (cm) table",
            "Return the table for Weight (kg)",
            "Provide the table shell for Body Mass Index (kg/m2)",
            "Show me the table shell for Body Surface Area (m2)",
            "Give me the Stratification Factor 1 (EDC) table",
            "Return the table for Stratification Factor 1 (IXRS)",
            "Provide the table shell for ECOG Performance Status",
            "Provide the table shell for 12-Lead ECG",
            "Show me the table shell for Baseline/Biomarker Subgroups",
            "Give me the Smoking Status table",
            "Return the table for Bone marrow/aspirate blast count at baseline)",
            "Provide the table shell for Region/Country of Enrollment",
            "Show me the table shell for Time from Initial Histologic Diagnosis to Randomization (days)",
            "Give me the Histology table",
            "Return the table for Tumor Stage at Initial Diagnosis",
            "Provide the table shell for Tumor Stage at Study Entry",
            "Show me the table shell for TNM Stage at Study Entry (T)",
            "Give me the TNM Stage at Study Entry (N) table",
            "Return the table for TNM Stage at Study Entry (M)",
            "Provide the table shell for Histologic Grade",
            "Provide the table shell for History of Brain Metastasis",
            "Show me the table shell for History of Other Metastasis",
            "Give me the Location of Metastasis [in a descending # order] table",
            "Show me the table shell for Any Prior Systemic Cancer Therapy",
            "Give me the Lines of Prior Systemic Cancer Therapy for Relapsed or Metastatic Disease table",
            "Return the table for Number of Regimens of Prior Systemic Cancer Therapy",
            "Provide the table shell for Number of Prior Cancer Chemotherapy",
            "Show me the table shell for Intent of Prior Cancer Chemotherapy Regimens",
            "Give me the History of Best Response to [Prior or Most Recent] Therapy table",
            "Return the table for Reason for Discontinuation of Prior Therapy",
            "Provide the table shell for Prior Radiation Therapy",
            "Provide the table shell for Type of Prior Radiation Therapy",
            "Show me the table shell for Prior Cancer Surgery",
            "Give me the Prior Cancer Biopsy table",
            "Return the table for Subjects with any prior anti-cancer therapy n (%)",
            "Provide the table shell for System Organ Class"
        ]

        print("\n--- Running RAG Queries to Retrieve Full Table Shells ---")
        for q in rag_queries:
            print(f"\n▶️ Querying: '{q}'")
            response = query_engine.query(q)
            print(f"🤖 AI Response:\n{response}")
            print("-" * 70)