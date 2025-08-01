import os
import pandas as pd
import json

# --- Docling Imports ---
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.pipeline_options import TableFormerMode
except ImportError as e:
    print(f"Error importing Docling components: {e}")
    print("Please install Docling using one of these methods:")
    print("1. Basic installation: pip install docling")
    print("2. With OCR support: pip install 'docling[ocr]'")
    print("3. Full installation: pip install 'docling[all]'")
    exit()

# --- LlamaIndex and LLM Imports ---
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def is_header_row(first_df_in_group, next_df):
    """
    Heuristic to check if the first row of next_df looks like a repeating header.
    Returns True if it's likely a new table header, False otherwise.
    """
    if first_df_in_group is None or next_df.empty:
        return False

    # Clean the column names from the first table in the current group
    header_cols = [str(c).strip().lower() for c in first_df_in_group.columns]
    
    # Clean the values from the first row of the next table being considered
    next_row_values = [str(v).strip().lower() for v in next_df.iloc[0]]

    # If the first row is identical to the header, it's a clear header
    if header_cols == next_row_values:
        return True
        
    # More advanced check: count how many row cells match header columns
    matches = 0
    # Use a high threshold to avoid false positives
    similarity_threshold = 0.7 
    
    for h_col, n_val in zip(header_cols, next_row_values):
        if h_col == n_val or (n_val and n_val in h_col):
            matches += 1
    
    # If a high percentage of cells in the row match the headers, treat it as a new table
    if len(header_cols) > 0 and (matches / len(header_cols)) >= similarity_threshold:
        print(f"INFO: Detected a new header row, starting a new table.")
        return True
        
    return False

def combine_tables(tables):
    """
    Combine sequential tables split across pages.
    Stops combining if a new table appears to start with a repeating header.
    """
    if not tables:
        return []

    table_dfs = []
    for i, table in enumerate(tables):
        try:
            df = table.export_to_dataframe()
            if not df.empty:
                table_dfs.append({'index': i, 'dataframe': df})
        except Exception as e:
            print(f"Warning: Could not convert table {i} to DataFrame: {e}")

    if not table_dfs:
        return []

    final_combined_tables = []
    current_group = []

    for table_info in table_dfs:
        current_df = table_info['dataframe']
        first_df_in_group = current_group[0]['dataframe'] if current_group else None
        
        # A new table starts if:
        # 1. It's the first table we've seen.
        # 2. The column count is different from the current group.
        # 3. The first row looks like a new header.
        start_new_table = (
            not current_group or
            len(first_df_in_group.columns) != len(current_df.columns) or
            is_header_row(first_df_in_group, current_df)
        )

        if start_new_table:
            if current_group:
                # Finalize the previous group and add it to our list
                combined_df = pd.concat([item['dataframe'] for item in current_group], ignore_index=True)
                combined_df.columns = current_group[0]['dataframe'].columns
                indices = [item['index'] for item in current_group]
                is_combined = len(indices) > 1
                table_name = f"Combined_Table_{'-'.join(map(str, indices))}" if is_combined else f"Table_{indices[0]}"
                
                final_combined_tables.append({
                    'dataframe': combined_df.drop_duplicates().reset_index(drop=True),
                    'original_indices': indices, 'combined': is_combined, 'table_name': table_name
                })
                if is_combined:
                    print(f"✅ Combined segments {indices} into one table.")
            
            # Start a new group with the current table
            current_group = [table_info]
        else:
            # This is a continuation, add it to the current group
            current_group.append(table_info)

    # Process the very last group after the loop finishes
    if current_group:
        combined_df = pd.concat([item['dataframe'] for item in current_group], ignore_index=True)
        combined_df.columns = current_group[0]['dataframe'].columns
        indices = [item['index'] for item in current_group]
        is_combined = len(indices) > 1
        table_name = f"Combined_Table_{'-'.join(map(str, indices))}" if is_combined else f"Table_{indices[0]}"

        final_combined_tables.append({
            'dataframe': combined_df.drop_duplicates().reset_index(drop=True),
            'original_indices': indices, 'combined': is_combined, 'table_name': table_name
        })
        if is_combined:
            print(f"✅ Combined segments {indices} into one table.")

    return final_combined_tables


def export_tables_to_files(extracted_tables, output_dir="./extracted_tables"):
    """
    Export extracted tables to various file formats for visualization.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Exporting tables to '{output_dir}' for visualization...")
    
    for table_name, df in extracted_tables.items():
        # Clean table name for filename
        safe_name = "".join(c for c in table_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        try:
            # Export to Excel
            excel_path = os.path.join(output_dir, f"{safe_name}.xlsx")
            df.to_excel(excel_path, index=False)
            
            # Export to CSV
            csv_path = os.path.join(output_dir, f"{safe_name}.csv")
            df.to_csv(csv_path, index=False)
            
            # Export to Markdown
            md_path = os.path.join(output_dir, f"{safe_name}.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {table_name}\n\n")
                f.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n\n")
                f.write(df.to_markdown(index=False))
            
            # Export to HTML for easy viewing
            html_path = os.path.join(output_dir, f"{safe_name}.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>{table_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .info {{ background-color: #e7f3ff; padding: 10px; margin-bottom: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="info">
        <h1>{table_name}</h1>
        <p><strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
    </div>
    {df.to_html(table_id="main-table", classes="table", index=False)}
</body>
</html>
                """)
            
            print(f"✅ Exported '{table_name}' to multiple formats")
            
        except Exception as e:
            print(f"❌ Error exporting table '{table_name}': {e}")
    
    print(f"📊 Export complete! Check '{output_dir}' folder for visualization files.")


def parse_docling_document(file_path, combine_tables_flag=True, export_tables=True):
    """
    Parse document (PDF or DOCX) using Docling and extract tables with optional combination.
    Returns a dictionary of table_name -> DataFrame pairs.
    """
    try:
        # Determine file format from extension
        file_ext = os.path.splitext(file_path)[1].lower()
        doc_converter = None

        if file_ext == '.pdf':
            print("📄 Detected PDF file. Configuring PDF pipeline...")
            # PDF needs special options for layout analysis
            from docling.document_converter import PdfFormatOption
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

        elif file_ext == '.docx':
            print("📄 Detected DOCX file. Using default converter.")
            # For DOCX, the default DocumentConverter is sufficient.
            doc_converter = DocumentConverter()
        
        else:
            print(f"❌ Unsupported file type: '{file_ext}'. Please provide a PDF or DOCX file.")
            return {}

        print(f"Processing document with Docling: {file_path}")
        conv_result = doc_converter.convert(file_path)
        doc = conv_result.document

        print(f"Found {len(doc.tables)} initial table segments in the document")

        if not doc.tables:
            print("No tables found in the document.")
            return {}

        # Process tables with optional combination
        if combine_tables_flag:
            print("Intelligently combining table segments...")
            processed_tables = combine_tables(doc.tables)
            print(f"After combination: {len(processed_tables)} final table(s)")
        else:
            # Fallback for no combination
            processed_tables = []
            for i, table in enumerate(doc.tables):
                try:
                    df = table.export_to_dataframe()
                    processed_tables.append({
                        'dataframe': df, 'original_indices': [i],
                        'combined': False, 'table_name': f"Table_{i}"
                    })
                except Exception as e:
                    print(f"Error processing table {i}: {e}")

        # Convert to the expected format (table_name -> DataFrame)
        extracted_tables = {}
        for table_info in processed_tables:
            table_name = table_info['table_name']
            df = table_info['dataframe']
            
            if not df.empty:
                if table_info['combined']:
                    table_name += f" (Combined from segments: {table_info['original_indices']})"
                extracted_tables[table_name] = df
                print(f"Added table '{table_name}' with shape: {df.shape}")

        if export_tables and extracted_tables:
            export_tables_to_files(extracted_tables)

        return extracted_tables

    except Exception as e:
        print(f"❌ Critical error during Docling parsing: {e}")
        return {}


if __name__ == "__main__":
    # --- CHOOSE YOUR INPUT FILE ---
    # You can now point this to a .pdf OR a .docx file.
    file_path = "/Users/Sravya/Desktop/AI_model_table_shells/Table shell standard.docx" # <-- CHANGE THIS PATH
    PERSIST_DIR = "./rag_storage_docling"

    # --- STEP 1: Parse document and extract tables ---
    if not os.path.exists(file_path):
        print(f"❌ File not found at path: {file_path}")
        exit()
        
    print("🔬 Parsing document to extract structured tables with Docling...")
    try:
        parsed_tables = parse_docling_document(
            file_path, 
            combine_tables_flag=True, 
            export_tables=True
        )
        print(f"✅ Parser finished. Found {len(parsed_tables)} final table(s).")
        
        if not parsed_tables:
            print("❌ No tables were extracted. Exiting.")
            exit()
            
    except Exception as e:
        print(f"❌ Critical error during parsing: {e}")
        exit()

    # --- STEP 2: Configure RAG settings ---
    print("\n⚙️ Configuring RAG settings (LLM and Embedding Model)...")
    Settings.llm = Ollama(model="mistral", request_timeout=1000.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    
    # --- STEP 3: Create or load vector index ---
    index = None
    if not os.path.exists(PERSIST_DIR):
        print(f"\n⚡ Index not found. Creating new index in '{PERSIST_DIR}'...")
        documents = []
        
        for table_name, df in parsed_tables.items():
            # Create multiple representations of each table
            table_markdown = df.to_markdown(index=False)
            
            # Main document with markdown format
            doc = Document(
                text=f"This document contains the table shell for '{table_name}'.\n\n{table_markdown}",
                metadata={
                    "table_name": table_name,
                    "format": "markdown",
                    "shape": f"{df.shape[0]}x{df.shape[1]}"
                }
            )
            documents.append(doc)
        
        print(f"📄 Converted {len(parsed_tables)} tables into {len(documents)} Documents for indexing.")
        print("🧠 Building vector index...")
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("💾 Index created and saved.")
    else:
        print(f"\n📂 Found existing index in '{PERSIST_DIR}'. Loading...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("✅ Index loaded.")

    # --- STEP 4: Create Query Engine with Custom Prompt ---
    if index:
        # Custom prompt template for table extraction
        qa_prompt_template_str = (
            "You are a table extraction assistant specialized in clinical and demographic data.\n"
            "The user will ask for a specific table or table shell. Your task is to find the corresponding table in the context and return **only the full, raw markdown content of that table.**\n"
            "Do not add any introductory text like 'Here is the table...'. Do not add any summary or explanation after the table. Your entire response should be just the markdown table.\n"
            "If you cannot find the exact table requested, return the closest matching table available.\n"
            "---------------------\n"
            "CONTEXT INFORMATION:\n{context_str}\n"
            "---------------------\n"
            "USER QUESTION: {query_str}\n"
            "ASSISTANT'S RESPONSE (full markdown table only):\n"
        )
        
        qa_template = PromptTemplate(qa_prompt_template_str)

        # Create the query engine with custom prompt
        query_engine = index.as_query_engine(
            text_qa_template=qa_template,
            similarity_top_k=3
        )
        
        # Sample queries based on common clinical table requests
        rag_queries = [
            "Show me the demographic characteristics table",
            "Give me the baseline characteristics table",
            "Provide the table shell for age and gender",
            "Return the adverse events table shell"
        ]

        print("\n--- Running RAG Queries to Retrieve Table Shells ---")
        print("Available tables:")
        for i, table_name in enumerate(parsed_tables.keys(), 1):
            print(f"  {i}. {table_name}")
        
        print(f"\n💡 Note: Extracted tables have been saved to './extracted_tables/'")
        
        print("\n" + "="*80)
        
        for q in rag_queries:
            print(f"\n▶️ Querying: '{q}'")
            try:
                response = query_engine.query(q)
                print(f"🤖 AI Response:\n{response}")
            except Exception as e:
                print(f"❌ Error processing query: {e}")
            print("-" * 70)
            
        # Interactive query mode
        print("\n" + "="*80)
        print("🎯 Interactive Mode - Enter your own queries (type 'quit' to exit)")
        print("="*80)
        
        while True:
            user_query = input("\n🔍 Enter your query: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            if user_query:
                try:
                    response = query_engine.query(user_query)
                    print(f"\n🤖 AI Response:\n{response}")
                except Exception as e:
                    print(f"❌ Error processing query: {e}")
        
        print("\n👋 Goodbye!")
    else:
        print("❌ Failed to create or load index.")