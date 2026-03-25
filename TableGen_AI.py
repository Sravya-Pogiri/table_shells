import streamlit as st
import pandas as pd
import numpy as np
import ollama
import openai
import PyPDF2
import re
import io
import os
import datetime
import tempfile
import time

st.set_page_config(layout="wide")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


try:
    from docling.document_converter import DocumentConverter
except ImportError:
    st.error("Warning: 'docling' is not installed. Table Shell Generator functionality will be limited. Please run: pip install 'docling[all]'")
    class DocumentConverter:
        def convert(self, file_path):
            class MockConvResult:
                class MockDocument:
                    tables = [pd.DataFrame([{"Message": "Docling not found. Please install it."}])]
                document = MockDocument()
            return MockConvResult()


from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def init_metrics_state():
    if "rag_metrics" not in st.session_state:
        st.session_state.rag_metrics = []
    if "rag_metrics_seq" not in st.session_state:
        st.session_state.rag_metrics_seq = 0

def record_rag_metric(model_name: str, retrieval_hit: bool, latency_ms: float, accuracy: float | None = None) -> int:
    init_metrics_state()
    st.session_state.rag_metrics_seq += 1
    entry = {
        "id": st.session_state.rag_metrics_seq,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model": model_name,
        "retrieval_hit": int(bool(retrieval_hit)),
        "latency_ms": float(latency_ms),
    }
    if accuracy is not None:
        entry["accuracy"] = float(accuracy)
    st.session_state.rag_metrics.append(entry)
    return entry["id"]

def set_metric_accuracy(metric_id: int, accuracy: float):
    init_metrics_state()
    for entry in st.session_state.rag_metrics:
        if entry.get("id") == metric_id:
            entry["accuracy"] = float(accuracy)
            break

def render_metrics_dashboard():
    init_metrics_state()
    st.header("📊 RAG Metrics Dashboard")
    if not st.session_state.rag_metrics:
        st.info("No metrics recorded yet. Use the app to generate results.")
        return

    df = pd.DataFrame(st.session_state.rag_metrics)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if df.empty:
        st.info("Metrics are empty after parsing timestamps.")
        return

    models = sorted(df["model"].unique())
    selected_models = st.multiselect("Filter by Model", options=models, default=models)
    if not selected_models:
        return

    df = df[df["model"].isin(selected_models)].sort_values("timestamp")

    st.subheader("Retrieval Hit Rate")
    hit_df = df.pivot_table(index="timestamp", columns="model", values="retrieval_hit", aggfunc="mean")
    st.line_chart(hit_df, height=220)

    if "accuracy" in df.columns:
        st.subheader("Accuracy (User Rating)")
        acc_df = df.pivot_table(index="timestamp", columns="model", values="accuracy", aggfunc="mean")
        if not acc_df.empty:
            st.line_chart(acc_df, height=220)

    st.subheader("Latency (ms)")
    latency_df = df.pivot_table(index="timestamp", columns="model", values="latency_ms", aggfunc="mean")
    st.line_chart(latency_df, height=220)


class SAPEmbedsWeb:
    def __init__(self):

        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1, 2))
        self.chunks = []
        self.vectors = None
        self.llm_model = ollama

    def process_pdf(self, file_path: str, chunk_length: int = 1000, chunk_overlap: int = 200):
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

    def retrieve(self, query, top=3):
        if not self.chunks or self.vectors is None:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        top_indices = np.argsort(similarities)[-top:][::-1]
        return [self.chunks[i] for i in top_indices]

    def analyzeLLM(self, query):
        best_chunks = self.retrieve(query, top=3)
        retrieval_hit = bool(best_chunks)

        if not best_chunks:
            return ["Could not find a relevant chunk in the document for your query."], retrieval_hit


        combined_context = "\n\n---\n\n".join(best_chunks)

        prompt = f"""You are extracting variables from a clinical trial protocol. Extract ALL relevant variables mentioned in the context below.

Question: {query}

Context from document:
---
{combined_context}
---

Instructions:
1. List each variable on a new line
2. Include the variable name exactly as it appears
3. Format as a numbered or bulleted list
4. Only include variables that directly answer the question

Answer (list the variables):"""

        try:
            client = openai.Client(
                base_url="https://api.groq.com/openai/v1",
                api_key=st.secrets["GROQ_API_KEY"]
            )
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1
            )
            clean_response = self.clean_response(response.choices[0].message.content)
            return [clean_response], retrieval_hit
        except Exception as e:
            return [f"Error communicating with LLM: {str(e)}"], retrieval_hit

    def clean_response(self, response_text):
        lines = response_text.split('\n')
        variables = []

        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.\s*', line) or re.match(r'^[\*\-]\s*', line):
                clean_var = re.sub(r'^\d+\.\s*|^[\*\-]\s*', '', line).strip()
                if clean_var and len(clean_var) > 1:
                    variables.append(clean_var)

        return variables if variables else [response_text]


def run_variable_extractor_app():
    st.header("1. TableGen AI: Variable Extractor")
    st.markdown("Upload a document, then ask questions to find and extract key variables.")

    uploaded_file = st.file_uploader(
        "Upload a PDF Document",
        type='pdf',
        key='var_extractor_uploader'
    )

    if 'sap_model' not in st.session_state:
        st.session_state.sap_model = SAPEmbedsWeb()

    if uploaded_file:
        if st.session_state.get('last_uploaded_file_name') != uploaded_file.name:
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                st.session_state.chunks = st.session_state.sap_model.process_pdf(tmp_path)
                os.unlink(tmp_path)
            st.success(f"Loaded {len(st.session_state.chunks)} chunks from '{uploaded_file.name}'")
            st.session_state.last_uploaded_file_name = uploaded_file.name
            st.session_state.chat_history = []
            st.rerun()

    if 'chunks' in st.session_state and st.session_state.chunks:
        st.subheader("❓ Ask a Question")
        sample_questions = [
            "What are the variables in the demographics section?",
            "What are the prior cancer therapies mentioned?",
            "What are the medical history variables?",
            "What are the stratification factors?",
        ]
        cols = st.columns(len(sample_questions))
        for i, q in enumerate(sample_questions):
            if cols[i].button(q, key=f"sample_{i}"):
                st.session_state.current_question = q
                st.session_state.query_submitted = True

        question = st.text_input(
            "Enter your question:",
            value=st.session_state.get('current_question', ''),
            key="question_input"
        )

        if st.button("Ask Question", type="primary") or st.session_state.get('query_submitted'):
            if question:
                st.session_state.query_submitted = False
                start_time = time.perf_counter()
                with st.spinner("Analyzing document..."):
                    results, retrieval_hit = st.session_state.sap_model.analyzeLLM(question)
                latency_ms = (time.perf_counter() - start_time) * 1000


                metric_id = record_rag_metric(
                    model_name="VariableExtractor",
                    retrieval_hit=retrieval_hit,
                    latency_ms=latency_ms
                )
                st.session_state.last_metric_id_var = metric_id
                st.session_state.chat_history.insert(0, (question, results))


                if 'variables_for_table' not in st.session_state:
                    st.session_state.variables_for_table = []
                category_key = question.lower().replace('what are the ', '').replace('?', '').strip()
                if 'extracted_variables' not in st.session_state:
                    st.session_state.extracted_variables = {}
                st.session_state.extracted_variables[category_key] = []
                for result_group in results:
                    if isinstance(result_group, list):
                        st.session_state.extracted_variables[category_key].extend(result_group)

                st.session_state.current_question = ""

        if st.session_state.get('chat_history'):
            st.subheader("Latest Results")
            latest_q, latest_a = st.session_state.chat_history[0]
            with st.container(border=True):
                st.write(f"**Your Question:** {latest_q}")
                st.divider()
                st.write("**Extracted Variables:**")
                extracted_vars_list = []
                for result_group in latest_a:
                    if isinstance(result_group, list):
                        for var in result_group:
                            st.markdown(f"- {var}")
                            extracted_vars_list.append(var)
                    else:
                        st.write(result_group)


                if extracted_vars_list and st.button("📊 Use These Variables in Table Generator", key="use_vars_btn"):
                    if 'variables_for_table' not in st.session_state:
                        st.session_state.variables_for_table = []
                    st.session_state.variables_for_table.extend(extracted_vars_list)
                    st.success(f"✓ Added {len(extracted_vars_list)} variables to table generator!")
                    st.info("Scroll down to the Table Generator section to create tables with these variables.")


                if st.session_state.get("last_metric_id_var"):
                    st.divider()
                    st.caption("Rate accuracy for the latest answer")
                    accuracy_value = st.slider(
                        "Accuracy (0-100)",
                        min_value=0, max_value=100, value=80,
                        key="accuracy_slider_variable"
                    )
                    if st.button("Submit Accuracy Rating", key="submit_accuracy_variable"):
                        set_metric_accuracy(st.session_state.last_metric_id_var, accuracy_value)
                        st.success("Accuracy rating saved.")


def parse_clinical_data_hierarchical(table_df: pd.DataFrame):
    sanitized_columns = [f"col_{i}" for i in range(len(table_df.columns))]
    table_df.columns = sanitized_columns

    records = table_df.to_dict('records')

    structured_data = {}
    current_main_table = None
    current_sub_table_name = None
    current_sub_table_rows = []
    main_table_headers = []

    def save_previous_sub_table():
        if current_main_table and current_sub_table_name and current_sub_table_rows and main_table_headers:
            df = pd.DataFrame(current_sub_table_rows, columns=main_table_headers)
            structured_data.setdefault(current_main_table, {})[current_sub_table_name] = df

    for row in records:
        cols = [str(val) if pd.notna(val) else "" for val in row.values()]
        first_col, second_col = cols[0].strip(), cols[1].strip()
        other_cols_str = ' '.join(cols[2:])

        if re.match(r'^Table [\d\.]+:.*', first_col, re.IGNORECASE):
            save_previous_sub_table()
            current_main_table = first_col
            current_sub_table_name, current_sub_table_rows, main_table_headers = None, [], []
            continue
        if not current_main_table:
            continue

        if 'Treatment A' in other_cols_str or 'Total (N=' in other_cols_str:
            main_table_headers = [second_col or 'Characteristic'] + [c.strip() for c in cols[2:]]
            if first_col:
                save_previous_sub_table()
                current_sub_table_name = first_col
                current_sub_table_rows = []
            continue

        if first_col and first_col.lower() not in ["parameter", "characteristics"]:
            save_previous_sub_table()
            current_sub_table_name = first_col
            current_sub_table_rows = []
            if any(c.strip() for c in cols[2:]):
                current_sub_table_rows.append([first_col] + cols[2:])
        elif second_col and current_sub_table_name:
            current_sub_table_rows.append([second_col] + cols[2:])

    save_previous_sub_table()
    return {k: v for k, v in structured_data.items() if v}


def markdown_to_dataframe(md_string: str) -> pd.DataFrame:
    try:
        lines = md_string.strip().split('\n')
        table_lines = [line for line in lines if line.strip().startswith('|')]
        cleaned_md = '\n'.join(table_lines)

        if not cleaned_md:
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(cleaned_md), sep='|', skipinitialspace=True).dropna(axis=1, how='all').iloc[1:]
        df.columns = [col.strip() for col in df.columns]
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        for col in df.columns:
            df[col] = df[col].str.strip()

        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def text_to_dataframe(text_input: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(text_input))
    except Exception as e:
        st.error(f"Failed to parse input text as a table: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_rag_query_engine():
    if 'user_table_file' in st.session_state and st.session_state.user_table_file is not None:
        file_content = st.session_state.user_table_file.getvalue()
        file_io = io.BytesIO(file_content)
        st.info("Using user-uploaded file to build table index...")
        main_table_df = pd.read_csv(file_io, header=None)
    else:

        possible_paths = [
            "mark - can you convert this into a markdown.csv",
            "./mark - can you convert this into a markdown.csv",
            os.path.join(os.getcwd(), "mark - can you convert this into a markdown.csv")
        ]
        file_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if not file_path:
            st.warning("Default table template file not found. Please upload a table CSV file above.")
            return None, {}
        st.info("Using default pre-existing file to build table index...")
        main_table_df = pd.read_csv(file_path, header=None)

    try:
        hierarchical_tables = parse_clinical_data_hierarchical(main_table_df)
        if not hierarchical_tables:
            raise ValueError("Parser did not find any valid tables.")
        st.success(f"Parser finished. Found {len(hierarchical_tables)} main categories.")
        parsed_tables = {
            sub_name: df
            for main_cat in hierarchical_tables.values()
            for sub_name, df in main_cat.items()
        }
    except Exception as e:
        st.error(f"Failed to parse source file for Table Generator: {e}")
        return None, {}

    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    st.info("Building new vector index for tables...")
    documents = [
        Document(text=f"Table: {name}\n{df.to_markdown(index=False)}", metadata={"table_name": name})
        for name, df in parsed_tables.items()
    ]
    index = VectorStoreIndex.from_documents(documents)
    st.session_state.index_ready = True


    qa_template = PromptTemplate(
        """You are a clinical trial table shell expert. Based on the context, find and return the most relevant table.

        Context: {context_str}

        Question: {query_str}

        Return the complete table in markdown format with all columns and rows. Include table headers and all data rows."""
    )
    query_engine = index.as_query_engine(
        text_qa_template=qa_template,
        similarity_top_k=3,
        response_mode="compact"
    )
    st.success("Table Shell Model is ready.")
    return query_engine, hierarchical_tables


def run_table_shell_app():
    st.header("2. Interactive Clinical Table Shell Generator")
    st.markdown("Select table shells, create custom ones, edit everything, and download.")

    uploaded_table_file = st.file_uploader(
        "**Upload a Table Shell CSV**",
        type='csv',
        key='table_uploader',
        help="Upload a new CSV file to replace the default table shells."
    )
    if uploaded_table_file and uploaded_table_file != st.session_state.get('last_uploaded_table_file'):
        st.session_state.user_table_file = uploaded_table_file
        load_rag_query_engine.clear()
        st.session_state.last_uploaded_table_file = uploaded_table_file
        st.rerun()

    query_engine, hierarchical_categories = load_rag_query_engine()
    if not query_engine:
        return

    if 'generated_tables' not in st.session_state:
        st.session_state.generated_tables = {}
    if 'table_order' not in st.session_state:
        st.session_state.table_order = []

    st.subheader("Select or Add Tables")
    main_category_options = list(hierarchical_categories.keys())

    if 'selected_main_category' not in st.session_state or st.session_state.selected_main_category not in main_category_options:
        st.session_state.selected_main_category = main_category_options[0] if main_category_options else None

    selected_main = st.selectbox(
        "**Step 1: Select a Main Category**",
        options=main_category_options,
        index=main_category_options.index(st.session_state.selected_main_category) if st.session_state.selected_main_category else 0
    )
    st.session_state.selected_main_category = selected_main

    if selected_main:
        sub_category_options = list(hierarchical_categories.get(selected_main, {}).keys())
        default_selections = [sub for sub in sub_category_options if sub in st.session_state.generated_tables]
        selected_sub_categories = st.multiselect(
            f"**Step 2: Select Sub-Tables from '{selected_main}'**",
            options=sub_category_options, default=default_selections
        )

        if st.button("Generate/Update Selected Tables", type="primary"):
            with st.spinner("Querying model and updating tables..."):
                for category in selected_sub_categories:
                    if category not in st.session_state.generated_tables:
                        start_time = time.perf_counter()
                        response = query_engine.query(f"Show me the table shell for {category}")
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        df = markdown_to_dataframe(str(response))


                        metric_id = record_rag_metric(
                            model_name="TableGenerator",
                            retrieval_hit=not df.empty,
                            latency_ms=latency_ms
                        )
                        st.session_state.last_metric_id_table = metric_id

                        if not df.empty:
                            st.session_state.generated_tables[category] = df
                            if category not in st.session_state.table_order:
                                st.session_state.table_order.append(category)

                for category in list(st.session_state.generated_tables.keys()):
                    if category in sub_category_options and category not in selected_sub_categories:
                        del st.session_state.generated_tables[category]
                        if category in st.session_state.table_order:
                            st.session_state.table_order.remove(category)
            st.rerun()


    if st.session_state.get('variables_for_table'):
        with st.expander("🔗 Variables from Document Analysis", expanded=True):
            st.success(f"**{len(st.session_state.variables_for_table)} variables** extracted from your document:")
            for i, var in enumerate(st.session_state.variables_for_table, 1):
                st.write(f"{i}. {var}")

            if st.button("📋 Create Table from These Variables"):
                table_name = f"Extracted Variables - {datetime.datetime.now().strftime('%H:%M:%S')}"
                new_df = pd.DataFrame({
                    'Variable': st.session_state.variables_for_table,
                    'Group A (N=XX)': [''] * len(st.session_state.variables_for_table),
                    'Group B (N=XX)': [''] * len(st.session_state.variables_for_table),
                    'Total (N=XX)': [''] * len(st.session_state.variables_for_table)
                })
                st.session_state.generated_tables[table_name] = new_df
                st.session_state.table_order.append(table_name)
                st.success(f"✓ Created table '{table_name}' with {len(st.session_state.variables_for_table)} variables!")
                st.session_state.variables_for_table = []
                st.rerun()

    with st.expander("Add a New Custom Table"):
        new_category_name = st.text_input("Enter a name for the new table:", key="new_table_name")

        if st.button("➕ Add Custom Table"):
            name = new_category_name.strip()
            if not name:
                st.warning("Please enter a name for the new table.")
            elif name in st.session_state.generated_tables:
                st.error(f"A table with the name '{name}' already exists. Please choose a different name.")
            else:
                st.session_state.generated_tables[name] = pd.DataFrame({
                    'Variable': ['Placeholder Row 1'],
                    'Group A (N=XX)': ['...'],
                    'Group B (N=XX)': ['...'],
                    'Group C (N=XX)': ['...']
                })
                st.session_state.table_order.append(name)
                st.success(f"Added new placeholder table: '{name}'. You can now edit it below.")
                st.rerun()

    if st.session_state.generated_tables:
        st.subheader("Edit and Download Tables")
        st.info("Use the buttons to reorder, and edit headers/data. Changes are saved instantly.")

        for i, category in enumerate(st.session_state.table_order[:]):
            df = st.session_state.generated_tables[category]
            with st.container(border=True):
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.subheader(f"Table: {category}")
                with col2:
                    reorder_cols = st.columns(2)
                    if reorder_cols[0].button("⬆️", key=f"up_{category}", help="Move table up", use_container_width=True):
                        if i > 0:
                            st.session_state.table_order.insert(i - 1, st.session_state.table_order.pop(i))
                            st.rerun()
                    if reorder_cols[1].button("⬇️", key=f"down_{category}", help="Move table down", use_container_width=True):
                        if i < len(st.session_state.table_order) - 1:
                            st.session_state.table_order.insert(i + 1, st.session_state.table_order.pop(i))
                            st.rerun()

                new_headers = [
                    st.text_input(f"Header for '{col}'", value=col, key=f"h_{category}_{j}")
                    for j, col in enumerate(df.columns)
                ]
                if st.button(f"Apply Header Changes for '{category}'", key=f"apply_{category}"):
                    df_copy = df.copy()
                    df_copy.columns = new_headers
                    st.session_state.generated_tables[category] = df_copy
                    st.rerun()

                edited_df = st.data_editor(df, key=f"editor_{category}", num_rows="dynamic", use_container_width=True)

                if not df.equals(edited_df):
                    st.session_state.generated_tables[category] = edited_df
                    st.rerun()

        output = io.StringIO()
        for category in st.session_state.table_order:
            df_to_save = st.session_state.generated_tables[category]
            output.write(f'"{category}"\n')
            df_to_save.to_csv(output, index=False)
            output.write("\n")

        st.download_button(
            "⬇️ Download All Tables as CSV",
            output.getvalue().encode('utf-8'),
            "custom_table_shells.csv",
            "text/csv"
        )


    if st.session_state.get("last_metric_id_table"):
        st.divider()
        st.caption("Rate accuracy for the latest table generation")
        accuracy_value = st.slider(
            "Accuracy (0-100)",
            min_value=0, max_value=100, value=80,
            key="accuracy_slider_table"
        )
        if st.button("Submit Accuracy Rating", key="submit_accuracy_table"):
            set_metric_accuracy(st.session_state.last_metric_id_table, accuracy_value)
            st.success("Accuracy rating saved.")


def main():
    st.title("TableGen AI: Document Extractor & Table Generator")


    if 'extracted_variables' not in st.session_state:
        st.session_state.extracted_variables = {}
    if 'variables_for_table' not in st.session_state:
        st.session_state.variables_for_table = []

    with st.sidebar:
        st.header("Controls & History")


        if st.session_state.get('variables_for_table'):
            st.success(f"✅ {len(st.session_state.variables_for_table)} variables ready for table generation")

        st.info("This panel contains the conversation history for the **Variable Extractor**.")
        st.subheader("🗣️ Conversation History")
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()

        if st.session_state.get('chat_history'):
            for i, (q, a) in enumerate(st.session_state.chat_history):
                if st.button(q, key=f"history_{i}"):
                    st.session_state.current_question = q
                    st.session_state.query_submitted = True
                    st.rerun()

    run_variable_extractor_app()
    st.divider()
    run_table_shell_app()
    st.divider()

    render_metrics_dashboard()


if __name__ == "__main__":
    main()
