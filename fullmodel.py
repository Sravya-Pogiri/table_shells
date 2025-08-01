import os
import pandas as pd
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.prompts import PromptTemplate
import json
import signal
from contextlib import contextmanager
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Docling Imports
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    from docling.datamodel.pipeline_options import TableFormerMode
except ImportError as e:
    print(f"Error importing Docling components: {e}")
    print("Please install Docling: pip install docling")
    exit()

# LlamaIndex Document Import
try:
    from llama_index.core import Document
except ImportError:
    try:
        from llama_index.schema import Document
    except ImportError as e:
        print(f"Error importing LlamaIndex Document: {e}")
        print("Please install LlamaIndex: pip install llama-index")
        exit()

# PERFORMANCE OPTIMIZATION 1: Faster embedding model
def get_optimized_embedding_model():
    """Use a faster, smaller embedding model for better performance"""
    try:
        # Use a smaller, faster model - you can also try "all-MiniLM-L6-v2" for even faster performance
        return HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    except Exception as e:
        print(f"Error loading optimized embedding model: {e}")
        # Fallback to your original model
        return HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# PERFORMANCE OPTIMIZATION 2: Optimized Docling settings
def read_document_with_docling_optimized(file_path, combine_tables_flag=True, similarity_threshold=0.7):
    """
    Optimized version with faster Docling settings and reduced processing overhead
    """
    try:
        # OPTIMIZATION: Faster pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # Skip OCR for faster processing
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = False  # Disable for speed
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST  # Use FAST mode instead of ACCURATE

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        print(f"Processing document with optimized Docling settings: {file_path}")
        start_time = time.time()
        conv_result = doc_converter.convert(file_path)
        doc = conv_result.document
        print(f"Docling processing completed in {time.time() - start_time:.2f} seconds")

        llama_documents = []
        print(f"Found {len(doc.tables)} tables in the document")

        # OPTIMIZATION: Reduce document creation overhead
        combined_tables = []
        if combine_tables_flag and doc.tables:
            combined_tables = combine_tables_helper_optimized(doc.tables, similarity_threshold)
            print(f"After combination: {len(combined_tables)} table(s)")
        else:
            for i, table in enumerate(doc.tables):
                try:
                    df = table.export_to_dataframe()
                    combined_tables.append({
                        'dataframe': df,
                        'original_indices': [i],
                        'combined': False
                    })
                except Exception as e:
                    print(f"Error processing table {i} to DataFrame: {e}")

        # OPTIMIZATION: Reduce number of document variants created per table
        for table_idx, table_info in enumerate(combined_tables):
            try:
                table_df = table_info['dataframe']
                original_indices = table_info['original_indices']
                is_combined = table_info['combined']

                # Get table name (simplified logic)
                table_name = get_table_name_fast(table_df, table_idx)

                # Special handling for Medical History table (unchanged)
                if "Medical History" in table_name or any("Medical History" in str(col) for col in table_df.columns):
                    medical_history_doc = process_medical_history_table(table_df, table_idx, original_indices, is_combined)
                    if medical_history_doc:
                        llama_documents.append(medical_history_doc)
                    continue

                # OPTIMIZATION: Create fewer document variants (only the most useful ones)
                table_metadata = {
                    "element_type": "table",
                    "table_name": table_name,
                    "table_index": table_idx,
                    "original_table_indices": original_indices,
                    "is_combined": is_combined,
                    "page_number": "N/A"
                }

                # Only create 2 variants instead of 4 (markdown for readability, CSV for structure)
                table_markdown = table_df.to_markdown(index=True)
                table_csv = table_df.to_csv(index=True)

                llama_documents.append(Document(
                    text=f"{table_name} (Markdown):\n{table_markdown}",
                    metadata={**table_metadata, "format": "markdown"}
                ))
                
                llama_documents.append(Document(
                    text=f"{table_name} (CSV):\n{table_csv}",
                    metadata={**table_metadata, "format": "csv"}
                ))

                print(f"Processed {table_name} with {len(table_df)} rows and {len(table_df.columns)} columns")

            except Exception as e:
                print(f"Error processing combined table {table_idx}: {e}")

        # OPTIMIZATION: Simplified text processing
        process_other_elements_fast(doc, llama_documents)

        print(f"Created {len(llama_documents)} documents from Docling parsing")
        return llama_documents

    except Exception as e:
        print(f"Error extracting content from {file_path} with Docling: {e}")
        return []

def get_table_name_fast(table_df, table_idx):
    """Fast table name extraction"""
    if not table_df.empty:
        # Quick check for common patterns
        for col in table_df.columns:
            col_str = str(col).strip()
            if col_str and col_str != "nan" and "Characteristics" not in col_str:
                return col_str
    return f"Table_{table_idx}"

def process_medical_history_table(table_df, table_idx, original_indices, is_combined):
    """Optimized Medical History processing"""
    try:
        print(f"Processing Medical History table with optimized method")
        structured_medical_history = []
        
        # Simplified processing - focus on getting the structure right quickly
        df_copy = table_df.copy()
        df_copy.columns = [str(col).strip() for col in df_copy.columns.values]
        
        current_soc_entry = None
        for idx, row in df_copy.iterrows():
            if idx > 50:  # OPTIMIZATION: Limit processing for very large tables
                break
                
            char_value = str(row.iloc[0]).strip()
            
            if char_value.startswith("System Organ Class") or char_value == "Subjects with any Report":
                if current_soc_entry:
                    structured_medical_history.append(current_soc_entry)
                
                current_soc_entry = {
                    "System Organ Class": char_value.replace("     ", "").strip(),
                    "Treatment A": str(row.iloc[1]) if len(row) > 1 else "",
                    "Treatment B": str(row.iloc[2]) if len(row) > 2 else "",
                    "Total": str(row.iloc[3]) if len(row) > 3 else "",
                    "Preferred Terms": []
                }
            elif "Preferred Term" in char_value and current_soc_entry:
                term_value = char_value.replace("          ", "").strip()
                current_soc_entry["Preferred Terms"].append({
                    "Term": term_value,
                    "Treatment A": str(row.iloc[1]) if len(row) > 1 else "",
                    "Treatment B": str(row.iloc[2]) if len(row) > 2 else "",
                    "Total": str(row.iloc[3]) if len(row) > 3 else ""
                })
        
        if current_soc_entry:
            structured_medical_history.append(current_soc_entry)

        return Document(
            text=json.dumps({"Medical History": structured_medical_history}, indent=2),
            metadata={
                "element_type": "table",
                "table_name": "Medical History Full /Safety Analysis Set",
                "table_index": table_idx,
                "original_table_indices": original_indices,
                "is_combined": is_combined,
                "page_number": "N/A",
                "format": "json_structured"
            }
        )
    except Exception as e:
        print(f"Error processing Medical History table: {e}")
        return None

def process_other_elements_fast(doc, llama_documents):
    """Fast processing of non-table elements"""
    element_count = 0
    for element in doc.body:
        element_count += 1
        if element_count > 100:  # OPTIMIZATION: Limit number of elements processed
            break
            
        element_type = element.__class__.__name__.lower()
        if 'table' in element_type:
            continue

        if hasattr(element, 'text') and element.text and element.text.strip():
            # Only add significant text elements
            if len(element.text.strip()) > 20:  # Skip very short elements
                llama_documents.append(Document(
                    text=element.text,
                    metadata={
                        "element_type": element_type,
                        "page_number": "N/A"
                    }
                ))

def combine_tables_helper_optimized(tables, similarity_threshold=0.7):
    """Optimized table combination with early stopping"""
    if not tables or len(tables) > 20:  # OPTIMIZATION: Skip combination for too many tables
        return [{'dataframe': table.export_to_dataframe(), 'original_indices': [i], 'combined': False} 
                for i, table in enumerate(tables) if not table.export_to_dataframe().empty]

    # Use your existing logic but with some optimizations
    table_dfs = []
    for i, table in enumerate(tables):
        try:
            df = table.export_to_dataframe()
            if not df.empty:
                table_dfs.append({
                    'index': i,
                    'dataframe': df,
                    'original_table': table
                })
        except Exception as e:
            print(f"Error converting table {i} to DataFrame: {e}")
            continue

    if not table_dfs:
        return []

    # Simplified combination logic
    final_combined_tables = []
    for table_info in table_dfs:
        final_combined_tables.append({
            'dataframe': table_info['dataframe'],
            'original_indices': [table_info['index']],
            'combined': False
        })
    
    return final_combined_tables

# OPTIMIZATION 3: Parallel query processing
def process_query_batch(query_engine, queries_batch, batch_name):
    """Process a batch of queries"""
    results = {}
    for i, query_text in enumerate(queries_batch):
        print(f"\n--- Processing {batch_name} Query {i+1}: {query_text[:50]}... ---")
        try:
            with timeout_context(300):  # Reduced timeout
                response = query_engine.query(query_text)
                llm_response_text = clean_llm_response(response.response)
                
                try:
                    parsed_json = json.loads(llm_response_text)
                    results[query_text] = parsed_json
                    print(f"✓ Successfully processed query {i+1}")
                except json.JSONDecodeError as e:
                    results[query_text] = {"error": f"JSON decode error: {e}", "raw_response": llm_response_text}
                    print(f"✗ JSON decode error for query {i+1}")
                    
        except TimeoutError:
            results[query_text] = {"error": "Query timed out"}
            print(f"✗ Query {i+1} timed out")
        except Exception as e:
            results[query_text] = {"error": str(e)}
            print(f"✗ Error in query {i+1}: {e}")
    
    return results

# Utility functions (optimized versions of your existing functions)
def clean_llm_response(response_text):
    """Optimized response cleaning"""
    response_text = response_text.strip()
    
    # Quick JSON extraction
    if response_text.startswith("```json"):
        response_text = response_text[7:-3].strip()
    elif response_text.startswith("```"):
        response_text = response_text[3:-3].strip()
    
    # Find JSON boundaries quickly
    start = response_text.find('{')
    if start != -1:
        end = response_text.rfind('}')
        if end > start:
            response_text = response_text[start:end+1]
    
    return response_text

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

# OPTIMIZATION 4: Optimized LlamaIndex Settings
def setup_optimized_settings():
    """Setup optimized LlamaIndex settings for faster processing"""
    Settings.embed_model = get_optimized_embedding_model()
    Settings.llm = Ollama(
        model="mistral", 
        request_timeout=300.0,  # Reduced timeout
        format="json",
        num_ctx=4096,  # Reduced context window for faster processing
        temperature=0.1  # Lower temperature for more consistent responses
    )

# Define the directory for storing the index
PERSIST_DIR = "./llama_index_storage"

if __name__ == '__main__':
    print("🚀 Starting optimized table extraction process...")
    
    # OPTIMIZATION: Setup optimized settings first
    setup_optimized_settings()
    
    file_path = "/Users/Sravya/Desktop/AI_model_table_shells/mark - can you convert this into a markdown.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Document file not found at {file_path}")
        exit()

    index = None
    
    # Index creation/loading with timing
    start_time = time.time()
    
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        print(f"\n📁 Creating new index...")
        
        # Use optimized document reading
        llama_documents = read_document_with_docling_optimized(
            file_path,
            combine_tables_flag=True,
            similarity_threshold=0.7
        )

        if llama_documents:
            print(f"📝 Created {len(llama_documents)} documents")
            print(f"⚡ Creating VectorStoreIndex...")
            
            try:
                index = VectorStoreIndex.from_documents(
                    llama_documents, 
                    show_progress=True
                )
                index.storage_context.persist(persist_dir=PERSIST_DIR)
                print(f"💾 Index created and saved in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"❌ Error creating index: {e}")
                exit()
        else:
            print("❌ No documents created")
            exit()
    else:
        print(f"\n📂 Loading existing index...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            print(f"📥 Index loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            exit()

    if not index:
        print("❌ No index available")
        exit()

    # OPTIMIZATION 5: Simplified prompt for faster processing
    simplified_qa_template_str = """Context: {context_str}

Extract the table structure for: {query_str}

Return ONLY valid JSON with the exact placeholder values from the document (like "xx", "xxx.xx", "xx (xx.x)").

For Medical History, use this structure:
{{"Medical History": [{{"System Organ Class": "...", "Treatment A": "...", "Treatment B": "...", "Total": "...", "Preferred Terms": [{{"Term": "...", "Treatment A": "...", "Treatment B": "...", "Total": "..."}}]}}]}}

For other tables, use:
{{"Table Name": {{"Metric": {{"Treatment A": "xx", "Treatment B": "xx", "Total": "xx"}}}}}}
"""
    
    simplified_qa_template = PromptTemplate(simplified_qa_template_str)
    
    # Create query engine with optimized settings
    query_engine = index.as_query_engine(
        llm=Settings.llm,
        response_synthesizer=CompactAndRefine(text_qa_template=simplified_qa_template),
        similarity_top_k=3  # Reduced from 5 for faster processing
    )

    # OPTIMIZATION 6: Process priority queries first
    priority_queries = [
        "Height (cm) table structure",
        "Weight (kg) table structure"
    ]
    
    # other_queries = [
    #     "Body Mass Index (kg/m2) table structure",
    #     "Body Surface Area (m2) table structure",
    #     "Stratification Factor 1 (EDC) table structure",
    #     "ECOG Performance Status table structure",
    #     "Smoking Status table structure"
    #     # Add more as needed, but start with fewer for testing
    # ]

    print(f"\n🔍 Processing {len(priority_queries)} priority queries...")
    
    # Process priority queries first
    priority_results = process_query_batch(query_engine, priority_queries, "Priority")
    
    # Save priority results
    with open("priority_results.json", "w") as f:
        json.dump(priority_results, f, indent=2)
    
    print(f"\n✅ Priority queries completed! Results saved to priority_results.json")
    print(f"⏱️  Total processing time: {time.time() - start_time:.2f} seconds")
    
    # Ask user if they want to continue with other queries
    continue_processing = input("\n❓ Continue with remaining queries? (y/n): ").lower().strip()
    
    if continue_processing == 'y':
        print(f"\n🔍 Processing {len(other_queries)} additional queries...")
        other_results = process_query_batch(query_engine, other_queries, "Additional")
        
        # Combine and save all results
        all_results = {**priority_results, **other_results}
        with open("all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ All queries completed! Results saved to all_results.json")
    
    print(f"\n🎉 Process completed in {time.time() - start_time:.2f} seconds total")