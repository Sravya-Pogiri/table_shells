import streamlit as st
import pandas as pd
import io
import os

# --- Step 1: Import functions from your provided script ---
# This assumes 'working_table_shell_model.py' is in the same directory.
from working_table_shell_model import parse_clinical_data
from docling.document_converter import DocumentConverter
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Helper function to parse markdown table back to DataFrame ---
def markdown_to_dataframe(md_string: str) -> pd.DataFrame:
    """Converts a markdown table string into a pandas DataFrame."""
    try:
        # The markdown from the model is clean; we can use StringIO to read it
        # The `sep` and `skipinitialspace` are key to parsing the markdown format correctly
        data = io.StringIO(md_string)
        df = pd.read_csv(data, sep='|', skipinitialspace=True)
        # Drop the header separator line (e.g., |:---|:---|)
        df = df.drop(index=0)
        # Drop the first and last columns which are empty due to the leading/trailing '|'
        df = df.iloc[:, 1:-1]
        # Clean up column and row values
        df.columns = [col.strip() for col in df.columns]
        df = df.apply(lambda x: x.str.strip())
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Failed to parse markdown table into an editable format: {e}")
        return pd.DataFrame()


# --- Step 2: Load and cache the RAG model and query engine from your script ---
@st.cache_resource
def load_rag_query_engine():
    """
    This function encapsulates the setup logic from your 'working_table_shell_model.py'
    and caches the result so it only runs once per session.
    """
    file_path = "mark - can you convert this into a markdown.csv"
    persist_dir = "./rag_storage"

    # --- Logic from your script to parse the source data ---
    st.info("Initializing model... Parsing document to extract structured tables.")
    doc_converter = DocumentConverter()
    conv_result = doc_converter.convert(file_path)
    main_table_df = conv_result.document.tables[0].export_to_dataframe()
    parsed_tables = parse_clinical_data(main_table_df)
    st.success(f"Parser finished. Found {len(parsed_tables)} sub-tables to load into the model.")

    # --- Logic from your script to configure and build/load the index ---
    Settings.llm = Ollama(model="mistral", request_timeout=1000.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    if not os.path.exists(persist_dir):
        st.info("Building new vector index for the model...")
        documents = [
            Document(
                text=f"This document contains the table shell for '{name}'.\n\n{df.to_markdown(index=False)}",
                metadata={"table_name": name}
            ) for name, df in parsed_tables.items()
        ]
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
        st.info("Index created and saved.")
    else:
        st.info("Loading existing model index from storage.")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

    # --- Logic from your script to create the query engine ---
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
    query_engine = index.as_query_engine(text_qa_template=qa_template)
    
    # Return both the engine and the list of available tables
    available_tables = list(parsed_tables.keys())
    st.success("Model ready.")
    return query_engine, available_tables

# --- Step 3: Build the Streamlit UI ---

st.set_page_config(layout="wide")
st.title("Interactive Clinical Table Shell Generator")
st.markdown("This application uses your custom RAG model to retrieve and display table shells.")

# Load the model and get the list of categories (table names)
query_engine, categories = load_rag_query_engine()

# Initialize session state to hold data
if 'generated_tables' not in st.session_state:
    st.session_state.generated_tables = {}

st.sidebar.header("Instructions")
st.sidebar.info(
    "1. **Select Categories:** Choose the tables you want to generate from the list. This list is sourced directly from your model.\n"
    "2. **Generate Tables:** Click the 'Generate Table Shells' button.\n"
    "3. **Edit Tables:** Modify the generated tables in the main view.\n"
    "4. **Download:** Click the 'Download as CSV' button to save."
)

selected_categories = st.multiselect(
    "Select Table Shell Categories from Your Model:",
    options=categories
)

if st.button("Generate Table Shells"):
    with st.spinner("Querying your model to generate tables..."):
        # We don't want to lose edits, so we only query for new tables
        current_tables = st.session_state.generated_tables
        for category in selected_categories:
            if category not in current_tables:
                # Use the query engine to get the markdown table
                query = f"Show me the table shell for {category}"
                response = query_engine.query(query)
                
                # Convert the markdown response to a DataFrame
                df = markdown_to_dataframe(str(response))
                if not df.empty:
                    current_tables[category] = df
        
        # Clean out tables that are no longer selected
        final_tables = {cat: df for cat, df in current_tables.items() if cat in selected_categories}
        st.session_state.generated_tables = final_tables


if st.session_state.generated_tables:
    st.header("Editable Table Shells")
    st.info("These tables have been generated by your model and are now editable.")

    edited_tables = {}
    for category, df in st.session_state.generated_tables.items():
        st.subheader(category)
        edited_df = st.data_editor(df, key=f"editor_{category}", num_rows="dynamic")
        edited_tables[category] = edited_df

    # Update the session state with the latest edits
    st.session_state.generated_tables = edited_tables

    # --- Download Logic ---
    if st.session_state.generated_tables:
        output = io.StringIO()
        # Add a header to indicate the file's origin
        output.write('"Generated from Custom Table Shell Model"\n\n')

        for category, df in st.session_state.generated_tables.items():
            # Write the title of the table
            output.write(f'"{category}"\n')
            # Write the dataframe to the string buffer
            df.to_csv(output, index=False)
            # Add a newline for spacing between tables
            output.write("\n")

        csv_data = output.getvalue().encode('utf-8')

        st.download_button(
            label="Download Edited Tables as CSV",
            data=csv_data,
            file_name="custom_table_shells.csv",
            mime="text/csv",
        )