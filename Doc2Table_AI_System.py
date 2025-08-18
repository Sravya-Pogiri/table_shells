# SAPEmbeds_web.py

# --- Core Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import ollama
import PyPDF2
import re
import io
import os
import datetime
import tempfile

st.set_page_config(layout="wide")

# --- App 1 (Variable Extractor) Imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- App 2 (Table Shell Generator) Imports ---
# These helper modules must be in the same directory
# Create placeholder functions if the actual files are not available
try:
    from working_table_shell_model import parse_clinical_data
    from docling.document_converter import DocumentConverter
except ImportError:
    st.error("Warning: 'working_table_shell_model.py' or 'docling' not found. Table Shell Generator functionality will be limited.")
    # Define dummy functions to allow the app to run without crashing
    def parse_clinical_data(df): return {"Error": pd.DataFrame([{"Message": "Parser not found"}])}
    class DocumentConverter:
        def convert(self, file_path):
            class MockConvResult:
                class MockDocument:
                    tables = [pd.DataFrame([{"Message": "Docling not found"}])]
                document = MockDocument()
            return MockConvResult()

# LlamaIndex components
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Global Configurations ---
# st.set_page_config(layout="wide")


# ##############################################################################
# --- APP 1: VARIABLE EXTRACTOR (Classes and Functions) ---
# ##############################################################################

class SAPEmbedsWeb:
    """
    Manages the RAG pipeline for extracting variables from a PDF.
    Uses TF-IDF for retrieval and Ollama for generation.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.chunks = []
        self.vectors = None
        self.llm_model = ollama # Use the imported ollama client

    def process_pdf(self, file_path: str, chunk_length: int = 400, chunk_overlap: int = 80):
        """Extracts text from a PDF and splits it into overlapping chunks."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text
            
            chunks = []
            start = 0
            while start < len(full_text):
                end = min(start + chunk_length, len(full_text))
                chunks.append(full_text[start:end])
                start += chunk_length - chunk_overlap
            
            self.chunks = chunks
            if chunks:
                self.vectors = self.vectorizer.fit_transform(chunks)
            return chunks
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return []
    
    def retrieve(self, query, top=1):
        """Retrieves the most relevant text chunk using TF-IDF."""
        if not self.chunks or self.vectors is None:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        top_indices = np.argsort(similarities)[-top:][::-1]
        return [self.chunks[i] for i in top_indices]
    
    def analyzeLLM(self, query):
        """Sends the retrieved chunk and query to the LLM for analysis."""
        best_chunks = self.retrieve(query, top=1)
        if not best_chunks:
            return ["Could not find a relevant chunk in the document for your query."]
            
        results = []
        for chunk in best_chunks:
            prompt = f"""Answer the question using ONLY the given text chunk. Your primary goal is to extract and list the specific variables or items requested. Do not use outside information.
            
            Question: {query}
            
            Text Chunk:
            ---
            {chunk}
            ---
            
            Answer (list the variables):"""
            
            try:
                response = self.llm_model.chat(model='llama3.2', messages=[{
                    'role': 'user',
                    'content': prompt
                }])
                clean_response = self.clean_response(response['message']['content'])
                results.append(clean_response)
            except Exception as e:
                results.append(f"Error communicating with LLM: {str(e)}")
        
        return results
    
    def clean_response(self, response_text):
        """Cleans the LLM response to extract a list of variables."""
        lines = response_text.split('\n')
        variables = []
        
        for line in lines:
            line = line.strip()
            # Match numbered lists, bullet points (* or -), or lines that seem like items
            if re.match(r'^\d+\.\s*', line) or re.match(r'^[\*\-]\s*', line):
                clean_var = re.sub(r'^\d+\.\s*|^[\*\-]\s*', '', line).strip()
                if clean_var and len(clean_var) > 1:
                    variables.append(clean_var)
        
        # If no list was found, return the raw text as a single-item list
        return variables if variables else [response_text]

def run_variable_extractor_app():
    """Renders the UI for the Variable Extractor app."""
    st.header("1. Doc2Table AI: Variable Extractor")
    st.markdown("Upload a document, then ask questions to find and extract key variables. Use the sidebar to review and reuse past questions.")

    uploaded_file = st.file_uploader(
        "Upload a PDF Document (or use the default SAP.pdf)",
        type='pdf',
        key='var_extractor_uploader'
    )
    
    # Initialize model and process a default or uploaded file
    if 'sap_model' not in st.session_state:
        st.session_state.sap_model = SAPEmbedsWeb()

    if uploaded_file:
        if st.session_state.get('last_uploaded_file') != uploaded_file.name:
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                st.session_state.chunks = st.session_state.sap_model.process_pdf(tmp_path)
                os.unlink(tmp_path) # Clean up temp file
            st.success(f"Loaded {len(st.session_state.chunks)} chunks from '{uploaded_file.name}'")
            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.chat_history = [] # Clear history for new file
    elif 'chunks' not in st.session_state:
        with st.spinner("Loading default document: SAP.pdf..."):
            st.session_state.chunks = st.session_state.sap_model.process_pdf("./SAP.pdf")
        st.success(f"Loaded {len(st.session_state.chunks)} chunks from SAP.pdf")
        st.session_state.last_uploaded_file = 'SAP.pdf'
    
    if 'chunks' in st.session_state and st.session_state.chunks:
        st.subheader("❓ Ask a Question")
    
        st.markdown("**Sample questions:**")
        sample_questions = [
            "What are the variables in the demographics section?",
            "What are the prior cancer therapies mentioned?",
            "What are the medical history variables?",
            "What are the stratification factors?"
        ]
    
        cols = st.columns(len(sample_questions))
        for i, q in enumerate(sample_questions):
            with cols[i]:
                if st.button(q, key=f"sample_{hash(q)}"):
                    st.session_state.current_question = q
        
        question = st.text_input(
            "Enter your question:", 
            value=st.session_state.get('current_question', ''),
            placeholder="Type your question here...",
            key="question_input"
        )
        
        if st.button("Ask Question", type="primary"):
            if question:
                with st.spinner("Analyzing document..."):
                    results = st.session_state.sap_model.analyzeLLM(question)
                st.session_state.chat_history.insert(0, (question, results)) # Add to top
                if 'current_question' in st.session_state:
                    del st.session_state.current_question
                st.rerun()
            else:
                st.warning("Please enter a question.")
        
        if st.session_state.chat_history:
            st.subheader("Latest Results")
            latest_q, latest_a = st.session_state.chat_history[0]
            
            with st.container(border=True):
                st.write(f"**Your Question:** {latest_q}")
                st.divider()
                st.write("**Extracted Variables:**")
                for result_group in latest_a:
                    if isinstance(result_group, list) and result_group:
                        for var in result_group:
                            st.markdown(f"- {var}")
                    else:
                        st.write(result_group)


# ##############################################################################
# --- APP 2: TABLE SHELL GENERATOR (Classes and Functions) ---
# ##############################################################################

def markdown_to_dataframe(md_string: str) -> pd.DataFrame:
    """Converts a markdown table string into a pandas DataFrame."""
    try:
        # Clean the string to remove code block fences
        md_string = re.sub(r'```markdown\n|```', '', md_string).strip()
        data = io.StringIO(md_string)
        df = pd.read_csv(data, sep='|', skipinitialspace=True)
        # Drop the separator line (--- | --- | ---)
        df = df.drop(0).reset_index(drop=True)
        # Drop empty columns that result from trailing '|'
        df = df.dropna(axis=1, how='all')
        # Clean headers and cell content
        df.columns = [col.strip() for col in df.columns]
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        return df.reset_index(drop=True)
    except Exception:
        st.warning("Could not parse markdown perfectly, showing raw response.")
        return pd.DataFrame({'Raw Response': [md_string]})

@st.cache_resource
def load_rag_query_engine():
    """
    Loads the LlamaIndex RAG query engine, building a vector index if needed.
    """
    file_path = "mark - can you convert this into a markdown.csv"
    persist_dir = "./rag_storage"

    st.info("Initializing Table Shell Model...")
    try:
        doc_converter = DocumentConverter()
        conv_result = doc_converter.convert(file_path)
        # Assuming the first table is the main one to parse
        main_table_df = conv_result.document.tables[0].export_to_dataframe()
        parsed_tables = parse_clinical_data(main_table_df)
        st.success(f"Parser finished. Found {len(parsed_tables)} sub-tables.")
    except Exception as e:
        st.error(f"Failed to parse source file for Table Generator: {e}")
        return None, []

    Settings.llm = LlamaIndexOllama(model="mistral", request_timeout=1000.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    if not os.path.exists(persist_dir):
        st.info("Building new vector index for tables...")
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
        st.info("Loading existing table index from storage.")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

    qa_prompt_template_str = (
        "You are a table extraction assistant.\n"
        "Your task is to find the table corresponding to the user's request in the context and return **only the full, raw markdown content of that table shell.**\n"
        "Do not add any introductory text. Your entire response must be ONLY the markdown table, including the header.\n"
        "CONTEXT: {context_str}\n"
        "QUESTION: {query_str}\n"
        "ASSISTANT'S RESPONSE (full markdown table only):\n"
    )
    qa_template = PromptTemplate(qa_prompt_template_str)
    query_engine = index.as_query_engine(text_qa_template=qa_template)
    
    st.success("Table Shell Model is ready.")
    return query_engine, list(parsed_tables.keys())

def run_table_shell_app():
    st.header("2. Interactive Clinical Table Shell Generator")
    st.markdown("Select pre-defined table shells from the model, create your own from scratch, edit everything, and download the result.")

    query_engine, categories = load_rag_query_engine()
    if query_engine is None:
        st.error("Table Shell Generator could not be loaded. Please check file dependencies.")
        return

    if 'generated_tables' not in st.session_state:
        st.session_state.generated_tables = {}
    
    custom_keys = [key for key in st.session_state.generated_tables.keys() if key not in categories]
    all_available_categories = categories + custom_keys

    st.subheader("Select or Add Tables")
    selected_categories = st.multiselect(
        "Select table shells from the model:",
        options=all_available_categories,
        default=list(st.session_state.generated_tables.keys()),
        key="table_shell_multiselect"
    )
    
    if st.button("Generate/Update Selected Tables", type="primary"):
        with st.spinner("Querying model to generate tables..."):
            # Add newly selected tables
            for category in selected_categories:
                if category not in st.session_state.generated_tables and category in categories:
                    query = f"Show me the table shell for {category}"
                    response = query_engine.query(query)
                    df = markdown_to_dataframe(str(response))
                    if not df.empty:
                        st.session_state.generated_tables[category] = df
            
            # Remove deselected tables
            current_keys = list(st.session_state.generated_tables.keys())
            for key in current_keys:
                if key not in selected_categories:
                    del st.session_state.generated_tables[key]
        st.rerun()

    st.markdown("---")
    
    with st.expander("Add a New Custom Table"):
        col1, col2 = st.columns([2, 1])
        with col1:
            new_category_name = st.text_input("Enter name for new table category:", key="new_category_name")
        with col2:
            st.write("") 
            st.write("") 
            if st.button("➕ Add Custom Table"):
                name = new_category_name.strip()
                if not name:
                    st.warning("Please enter a name for the new table.")
                elif name in st.session_state.generated_tables or name in categories:
                    st.error(f"A table named '{name}' already exists.")
                else:
                    placeholder_df = pd.DataFrame({
                        'Variable': ['Placeholder Variable 1', 'Placeholder Variable 2'],
                        'N = [X]': ['...', '...'],
                        'Treatment Group A (%)': ['...', '...'],
                        'Treatment Group B (%)': ['...', '...'],
                    })
                    st.session_state.generated_tables[name] = placeholder_df
                    # Add to the multiselect options if not already there
                    if 'table_shell_multiselect' not in st.session_state:
                        st.session_state.table_shell_multiselect = []
                    st.session_state.table_shell_multiselect.append(name)
                    st.success(f"Added new editable table: '{name}'. You can now edit it below.")
                    st.rerun()

    st.markdown("---")

    if st.session_state.generated_tables:
        st.subheader("Edit and Download Tables")
        st.info("All generated and custom tables are listed below. You can edit headers and data directly. Press 'Apply Header Changes' after modifying column names.")

        edited_tables = {}
        for category, df in st.session_state.generated_tables.items():
            with st.container(border=True):
                st.subheader(f"Table: {category}")
                
                # --- HEADER EDITING LOGIC WITH APPLY BUTTON ---
                st.markdown("###### Edit Headers")
                current_headers = list(df.columns)
                header_cols = st.columns(len(current_headers))
                
                new_headers = []
                for i, header in enumerate(current_headers):
                    with header_cols[i]:
                        new_name = st.text_input(
                            label=f"Header {i+1}",
                            value=header,
                            key=f"header_input_{category}_{i}",
                            label_visibility="collapsed"
                        )
                        new_headers.append(new_name)
                
                # Only show apply button if headers have actually changed
                if new_headers != current_headers:
                    if st.button(f"Apply Header Changes for '{category}'", key=f"apply_headers_{category}"):
                        df_copy = df.copy()
                        df_copy.columns = new_headers
                        st.session_state.generated_tables[category] = df_copy
                        st.rerun()
                # --- End of Header Editing Logic ---
                
                st.markdown("###### Edit Data")
                edited_df = st.data_editor(
                    df, 
                    key=f"editor_{category}", 
                    num_rows="dynamic", 
                    use_container_width=True,
                    height= (len(df) + 1) * 35 + 3 # Dynamically adjust height
                )
                edited_tables[category] = edited_df
        
        # This is the crucial step: update the state with the edited data
        # This will run on every interaction, preserving the state of the data_editor
        st.session_state.generated_tables = edited_tables

        # --- DOWNLOAD BUTTON ---
        output = io.StringIO()
        output.write(f'"Generated from Custom Table Shell Model on {datetime.datetime.now().strftime("%Y-%m-%d")}"\n\n')
        for category, df_to_save in st.session_state.generated_tables.items():
            output.write(f'"{category}"\n')
            df_to_save.to_csv(output, index=False)
            output.write("\n")
        csv_data = output.getvalue().encode('utf-8')

        st.download_button(
            label="⬇️ Download All Edited Tables as CSV",
            data=csv_data,
            file_name="custom_table_shells.csv",
            mime="text/csv",
            type="primary"
        )


# ##############################################################################
# --- MAIN STREAMLIT APPLICATION ---
# ##############################################################################

def main():
    """Main function to run the single-page Streamlit app."""
    st.title("Dual RAG AI Systems: Document Extractor & Table Generator")
    st.markdown("---")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Controls & History")
        st.info(
            "This panel contains the conversation history for the **Variable Extractor** "
            "and provides instructions for the **Table Shell Generator**."
        )

        st.subheader("🗣️ Conversation History")
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()

        if not st.session_state.chat_history:
            st.write("No questions asked yet.")
        else:
            st.write("Click a past question to reuse it:")
            for i, (q, a) in enumerate(st.session_state.chat_history):
                if st.button(q, key=f"history_q_{i}"):
                    st.session_state.current_question = q
                    st.rerun()
        
        st.markdown("---")
        st.subheader("📋 Table Generator Guide")
        st.markdown(
            """
            1.  **Select Tables:** Choose shells from the model in the main view.
            2.  **Generate:** Fetch the selected tables.
            3.  **Add Custom:** Create a new, blank table shell.
            4.  **Edit:** Modify any table's headers or data. **Remember to click 'Apply Header Changes'** after renaming columns.
            5.  **Download:** Save your work as a single CSV file.
            """
        )

    # --- MAIN CONTENT AREA ---
    run_variable_extractor_app()

    st.divider()

    run_table_shell_app()

if __name__ == "__main__":
    main()