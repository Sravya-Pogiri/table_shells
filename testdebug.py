import os
import json
from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Correct import for loading index
from llama_index.core import load_index_from_storage # <--- ADD THIS LINE

# Setup settings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="mistral", request_timeout=1000.0)

def debug_export_folder(export_folder="./table_exports"):
    """Debug what's in the export folder"""
    print("🔍 DEBUGGING EXPORT FOLDER")
    print("=" * 50)
    
    if not os.path.exists(export_folder):
        print(f"❌ Export folder does not exist: {export_folder}")
        return
    
    print(f"📁 Export folder: {export_folder}")
    
    # List all files
    files = list(Path(export_folder).glob("*"))
    print(f"📊 Total files: {len(files)}")
    
    # Group by extension
    by_extension = {}
    for file in files:
        ext = file.suffix.lower()
        if ext not in by_extension:
            by_extension[ext] = []
        by_extension[ext].append(file.name)
    
    for ext, filenames in by_extension.items():
        print(f"  {ext}: {len(filenames)} files")
        for filename in filenames[:3]:  # Show first 3
            print(f"    - {filename}")
        if len(filenames) > 3:
            print(f"    ... and {len(filenames) - 3} more")
    
    # Try to read a sample CSV
    csv_files = [f for f in files if f.suffix.lower() == '.csv']
    if csv_files:
        print(f"\n📋 Sample CSV content from: {csv_files[0].name}")
        try:
            import pandas as pd
            df = pd.read_csv(csv_files[0])
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Sample data:\n{df.head(2).to_string()}")
        except Exception as e:
            print(f"  ❌ Error reading CSV: {e}")

def debug_vector_store(persist_dir="./shellstorage"):
    """Debug what's in the vector store"""
    print("\n🔍 DEBUGGING VECTOR STORE")
    print("=" * 50)
    
    if not os.path.exists(persist_dir):
        print(f"❌ Vector store does not exist: {persist_dir}")
        return None
    
    print(f"📁 Vector store folder: {persist_dir}")
    
    # List files in vector store
    vs_files = list(Path(persist_dir).glob("*"))
    print(f"📊 Vector store files: {len(vs_files)}")
    for file in vs_files:
        print(f"  - {file.name}")
    
    # Check metadata
    metadata_file = os.path.join(persist_dir, "index_metadata.json")
    if os.path.exists(metadata_file):
        print(f"\n📋 Metadata:")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            for key, value in metadata.items():
                print(f"  {key}: {value}")
    
    # Try to load the index
    try:
        print(f"\n🔄 Loading vector store...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        # index = VectorStoreIndex.from_storage_context(storage_context) # <--- REMOVE THIS LINE
        index = load_index_from_storage(storage_context) # <--- USE THIS LINE INSTEAD
        print(f"✅ Vector store loaded successfully")
        
        # Get some basic info
        docstore = index.docstore
        print(f"📚 Documents in store: {len(docstore.docs)}")
        
        # Show sample document metadata
        if docstore.docs:
            sample_doc_id = list(docstore.docs.keys())[0]
            sample_doc = docstore.docs[sample_doc_id]
            print(f"\n📄 Sample document metadata:")
            for key, value in sample_doc.metadata.items():
                print(f"  {key}: {value}")
            
            print(f"\n📝 Sample document text (first 200 chars):")
            print(f"  {sample_doc.text[:200]}...")
        
        return index
        
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None

def test_simple_queries(index):
    """Test with very simple queries"""
    if not index:
        return
    
    print("\n🧪 TESTING SIMPLE QUERIES")
    print("=" * 50)
    
    query_engine = index.as_query_engine()
    
    simple_queries = [
        "What documents are available?",
        "List all tables",
        "What data do you have?",
        "Show me any table information"
    ]
    
    for query in simple_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 30)
        try:
            response = query_engine.query(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

def inspect_document_content(index):
    """Inspect what's actually in the documents"""
    if not index:
        return
        
    print("\n🔍 INSPECTING DOCUMENT CONTENT")
    print("=" * 50)
    
    docstore = index.docstore
    
    # Group documents by type
    doc_types = {}
    table_names = set() # Use 'logical_title' from your Docling integration
    
    for doc_id, doc in docstore.docs.items():
        # Changed to use metadata keys from your Docling integration
        element_type = doc.metadata.get('element_type', 'unknown') 
        logical_title = doc.metadata.get('logical_title', 'N/A_No_Title') 
        
        if element_type not in doc_types:
            doc_types[element_type] = 0
        doc_types[element_type] += 1
        
        if element_type == 'table' and logical_title != 'N/A_No_Title': # Only add if it's a table with a title
            table_names.add(logical_title)
    
    print(f"📊 Document types:")
    for doc_type, count in doc_types.items():
        print(f"  {doc_type}: {count} documents")
    
    print(f"\n📋 Table logical titles found:") # Changed from 'table names' to 'logical titles'
    if not table_names:
        print("  (No logical table titles found. Ensure tables are correctly parsed with titles in LlamaIndex documents.)")
    for table_name in sorted(list(table_names)):
        print(f"  - {table_name}")
    
    # Show a few sample documents
    print(f"\n📄 Sample document contents:")
    sample_docs_to_show = 3
    # Try to get a mix of table and text docs if available
    sample_doc_ids = []
    table_doc_ids = [doc_id for doc_id, doc in docstore.docs.items() if doc.metadata.get('element_type') == 'table']
    text_doc_ids = [doc_id for doc_id, doc in docstore.docs.items() if doc.metadata.get('element_type') == 'text']

    if table_doc_ids:
        sample_doc_ids.append(table_doc_ids[0]) # First table doc
    if len(table_doc_ids) > 1:
        sample_doc_ids.append(table_doc_ids[1]) # Second table doc if exists
    if text_doc_ids:
        sample_doc_ids.append(text_doc_ids[0]) # First text doc

    # If we still don't have enough, just take from the beginning
    if len(sample_doc_ids) < sample_docs_to_show:
        for doc_id in list(docstore.docs.keys()):
            if doc_id not in sample_doc_ids:
                sample_doc_ids.append(doc_id)
            if len(sample_doc_ids) >= sample_docs_to_show:
                break


    for i, doc_id in enumerate(sample_doc_ids):
        if doc_id not in docstore.docs: # Safety check
            continue
        doc = docstore.docs[doc_id]
        print(f"\n--- Document {i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Text (first 300 chars): {doc.text[:300]}...")

def main():
    """Run all debugging functions"""
    print("🚀 VECTOR STORE DEBUGGING TOOL")
    print("=" * 60)
    
    # Step 1: Check export folder
    debug_export_folder()
    
    # Step 2: Check vector store
    # Ensure that your vector store has been *created* by your main script (which calls read_document_with_docling_structured_tables and then builds an index from it)
    # If you haven't created it yet, this debugger will report that it doesn't exist.
    index = debug_vector_store()
    
    # Step 3: Test queries
    test_simple_queries(index)
    
    # Step 4: Inspect content
    inspect_document_content(index)
    
    print("\n" + "=" * 60)
    print("🏁 Debugging complete!")

if __name__ == "__main__":
    main()