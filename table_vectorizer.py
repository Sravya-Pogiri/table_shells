import os
import pandas as pd
import json
from pathlib import Path
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
import pickle
from typing import List, Dict, Any


def setup_llama_settings():
    """Configure LlamaIndex settings"""
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = Ollama(model="mistral", request_timeout=1000.0)
    print("✓ LlamaIndex settings configured")


def load_table_from_files(table_files: Dict[str, str], table_info: Dict[str, Any]) -> List[Document]:
    """
    Load a table from its exported files and create multiple Document objects
    with different representations
    """
    documents = []
    table_name = table_info.get('description', 'Unknown Table')
    
    try:
        # Load CSV data
        if table_files.get('csv') and os.path.exists(table_files['csv']):
            df = pd.read_csv(table_files['csv'], index_col=0)
            
            # Create base metadata
            base_metadata = {
                "table_name": table_name,
                "table_index": table_info.get('table_index', 0),
                "original_indices": table_info.get('original_indices', []),
                "is_combined": table_info.get('is_combined', False),
                "shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
                "columns": list(df.columns),
                "source_file": table_files.get('csv', ''),
                "data_type": "table"
            }
            
            # 1. Create a structured overview document
            overview_text = f"""
Table: {table_name}
Shape: {df.shape[0]} rows × {df.shape[1]} columns
Columns: {', '.join(df.columns)}
Combined Table: {'Yes' if table_info.get('is_combined', False) else 'No'}
{f"Original Tables: {table_info.get('original_indices', [])}" if table_info.get('is_combined', False) else ""}

Summary Statistics:
{df.describe(include='all').to_string() if not df.empty else "No data available"}
            """
            
            documents.append(Document(
                text=overview_text,
                metadata={**base_metadata, "format": "overview", "content_type": "summary"}
            ))
            
            # 2. Create full table as CSV format document
            csv_text = f"Table: {table_name}\n\n{df.to_csv()}"
            documents.append(Document(
                text=csv_text,
                metadata={**base_metadata, "format": "csv", "content_type": "full_data"}
            ))
            
            # 3. Create table as markdown format document
            if not df.empty:
                markdown_text = f"# {table_name}\n\n{df.to_markdown()}"
                documents.append(Document(
                    text=markdown_text,
                    metadata={**base_metadata, "format": "markdown", "content_type": "full_data"}
                ))
            
            # 4. Create column-wise documents for detailed search
            for col in df.columns:
                if not df[col].dropna().empty:
                    col_data = df[col].dropna()
                    col_text = f"""
Column: {col} from {table_name}
Data Type: {col_data.dtype}
Non-null Values: {len(col_data)}
Unique Values: {col_data.nunique()}

Sample Values:
{col_data.head(10).to_string()}

{f"Value Counts (top 10):" if col_data.dtype == 'object' else "Statistics:"}
{col_data.value_counts().head(10).to_string() if col_data.dtype == 'object' else col_data.describe().to_string()}
                    """
                    
                    documents.append(Document(
                        text=col_text,
                        metadata={
                            **base_metadata, 
                            "format": "column_analysis",
                            "content_type": "column_data",
                            "column_name": col,
                            "column_dtype": str(col_data.dtype)
                        }
                    ))
            
            # 5. If table has reasonable size, create row-wise chunks
            if len(df) > 0 and len(df) <= 1000:  # Only for reasonably sized tables
                chunk_size = min(50, max(5, len(df) // 10))  # Adaptive chunk size
                for i in range(0, len(df), chunk_size):
                    chunk_df = df.iloc[i:i+chunk_size]
                    chunk_text = f"""
Rows {i+1}-{min(i+chunk_size, len(df))} from {table_name}:

{chunk_df.to_string()}
                    """
                    
                    documents.append(Document(
                        text=chunk_text,
                        metadata={
                            **base_metadata,
                            "format": "row_chunk",
                            "content_type": "data_chunk",
                            "chunk_start": i,
                            "chunk_end": min(i+chunk_size, len(df)),
                            "chunk_size": len(chunk_df)
                        }
                    ))
        
        # Load JSON metadata if available
        if table_files.get('json') and os.path.exists(table_files['json']):
            with open(table_files['json'], 'r') as f:
                json_data = json.load(f)
                json_text = f"JSON representation of {table_name}:\n\n{json.dumps(json_data, indent=2)}"
                documents.append(Document(
                    text=json_text,
                    metadata={**base_metadata, "format": "json", "content_type": "structured_data"}
                ))
        
        print(f"  ✓ Created {len(documents)} documents for {table_name}")
        return documents
        
    except Exception as e:
        print(f"  ❌ Error loading table {table_name}: {e}")
        return []


def scan_table_export_folder(export_folder: str) -> List[Dict[str, Any]]:
    """
    Scan the table export folder and identify all table files
    """
    export_path = Path(export_folder)
    if not export_path.exists():
        print(f"❌ Export folder not found: {export_folder}")
        return []
    
    print(f"📁 Scanning export folder: {export_folder}")
    
    # Group files by table
    table_groups = {}
    
    for file_path in export_path.glob("*"):
        if file_path.is_file():
            filename = file_path.stem
            extension = file_path.suffix
            
            # Skip summary files
            if "summary" in filename.lower():
                continue
            
            # Extract table identifier from filename
            # Expected format: {doc_name}_{table_identifier}.{extension}
            parts = filename.split('_')
            if len(parts) >= 2:
                # The table identifier should be the last parts
                if 'table' in filename or 'combined' in filename:
                    table_key = '_'.join(parts[:-1]) if len(parts) > 2 else filename
                    
                    if table_key not in table_groups:
                        table_groups[table_key] = {'files': {}, 'base_name': table_key}
                    
                    if extension.lower() == '.csv':
                        table_groups[table_key]['files']['csv'] = str(file_path)
                    elif extension.lower() == '.json':
                        table_groups[table_key]['files']['json'] = str(file_path)
                    elif extension.lower() == '.md':
                        table_groups[table_key]['files']['markdown'] = str(file_path)
                    elif extension.lower() == '.html':
                        table_groups[table_key]['files']['html'] = str(file_path)
                    elif extension.lower() == '.xlsx':
                        table_groups[table_key]['files']['excel'] = str(file_path)
    
    # Convert to list and add metadata
    table_list = []
    for idx, (table_key, table_info) in enumerate(table_groups.items()):
        table_data = {
            'table_index': idx,
            'description': table_key.replace('_', ' ').title(),
            'original_indices': [idx],  # We don't have original info, so use current
            'is_combined': 'combined' in table_key.lower(),
            'files': table_info['files']
        }
        table_list.append(table_data)
    
    print(f"  ✓ Found {len(table_list)} table groups")
    for table in table_list:
        print(f"    - {table['description']}: {list(table['files'].keys())}")
    
    return table_list


def create_vector_store(documents: List[Document], persist_dir: str = "./shellstorage") -> VectorStoreIndex:
    """
    Create and persist a vector store from documents
    """
    print(f"🔧 Creating vector store in: {persist_dir}")
    
    # Create the persist directory
    os.makedirs(persist_dir, exist_ok=True)
    
    try:
        # Create storage context with persistence
        vector_store = SimpleVectorStore()
        doc_store = SimpleDocumentStore()
        index_store = SimpleIndexStore()
        
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=doc_store,
            index_store=index_store
        )
        
        # Create the index
        print(f"  📊 Processing {len(documents)} documents...")
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            show_progress=True
        )
        
        # Persist to disk
        print(f"  💾 Persisting to disk...")
        index.storage_context.persist(persist_dir=persist_dir)
        
        # Save additional metadata
        metadata_file = os.path.join(persist_dir, "index_metadata.json")
        metadata = {
            "total_documents": len(documents),
            "creation_date": pd.Timestamp.now().isoformat(),
            "document_types": list(set([doc.metadata.get("content_type", "unknown") for doc in documents])),
            "table_names": list(set([doc.metadata.get("table_name", "unknown") for doc in documents])),
            "formats": list(set([doc.metadata.get("format", "unknown") for doc in documents]))
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Vector store created successfully!")
        print(f"  📈 Total documents indexed: {len(documents)}")
        print(f"  📋 Metadata saved to: {metadata_file}")
        
        return index
        
    except Exception as e:
        print(f"  ❌ Error creating vector store: {e}")
        raise


def load_existing_vector_store(persist_dir: str = "./shellstorage") -> VectorStoreIndex:
    """
    Load an existing vector store from disk
    """
    try:
        if not os.path.exists(persist_dir):
            print(f"❌ Vector store not found at: {persist_dir}")
            return None
        
        print(f"📂 Loading existing vector store from: {persist_dir}")
        
        # Load storage context
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        
        # Load the index
        index = VectorStoreIndex.from_storage_context(storage_context)
        
        # Load metadata if available
        metadata_file = os.path.join(persist_dir, "index_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"  ✅ Loaded vector store with {metadata.get('total_documents', 'unknown')} documents")
            print(f"  📅 Created: {metadata.get('creation_date', 'unknown')}")
        else:
            print(f"  ✅ Vector store loaded successfully")
        
        return index
        
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None


def vectorize_table_exports(export_folder: str = "./table_exports", 
                          persist_dir: str = "./shellstorage", 
                          rebuild: bool = False) -> VectorStoreIndex:
    """
    Main function to vectorize all table exports into a persistent vector store
    
    Args:
        export_folder: Path to the folder containing exported table files
        persist_dir: Path where the vector store should be persisted
        rebuild: If True, rebuild the vector store even if it exists
    
    Returns:
        VectorStoreIndex: The created or loaded vector store index
    """
    print("🚀 Starting table vectorization process...")
    
    # Setup LlamaIndex
    setup_llama_settings()
    
    # Check if vector store already exists
    if not rebuild and os.path.exists(persist_dir):
        print(f"📂 Vector store already exists at: {persist_dir}")
        response = input("Do you want to rebuild it? (y/N): ").lower()
        if response != 'y':
            return load_existing_vector_store(persist_dir)
    
    # Scan for table files
    table_list = scan_table_export_folder(export_folder)
    
    if not table_list:
        print("❌ No tables found in export folder")
        return None
    
    # Load all tables and create documents
    all_documents = []
    print(f"\n📚 Loading {len(table_list)} tables...")
    
    for table_info in table_list:
        print(f"  📊 Processing: {table_info['description']}")
        documents = load_table_from_files(table_info['files'], table_info)
        all_documents.extend(documents)
    
    if not all_documents:
        print("❌ No documents were created from the tables")
        return None
    
    print(f"\n📈 Total documents created: {len(all_documents)}")
    
    # Create vector store
    print(f"\n🔧 Creating vector store...")
    index = create_vector_store(all_documents, persist_dir)
    
    print(f"\n🎉 Vectorization complete!")
    print(f"   📁 Vector store location: {persist_dir}")
    print(f"   📊 Total documents: {len(all_documents)}")
    print(f"   🔍 Ready for querying!")
    
    return index


def test_vector_store(index: VectorStoreIndex, test_queries: List[str] = None):
    """
    Test the vector store with some sample queries
    """
    if not index:
        print("❌ No index provided for testing")
        return
    
    if not test_queries:
        test_queries = [
            "Show the System Organ Class Preferred Term table"
        ]
    
    print(f"\n🧪 Testing vector store with {len(test_queries)} queries...")
    
    query_engine = index.as_query_engine()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: {query} ---")
        try:
            response = query_engine.query(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"❌ Error with query: {e}")


if __name__ == "__main__":
    # Configuration
    EXPORT_FOLDER = "./table_exports"  # Folder containing exported table files
    PERSIST_DIR = "./shellstorage"     # Where to store the vector database
    REBUILD = False                    # Set to True to force rebuild
    
    # Run vectorization
    index = vectorize_table_exports(
        export_folder=EXPORT_FOLDER,
        persist_dir=PERSIST_DIR,
        rebuild=REBUILD
    )
    
    # Test the vector store
    if index:
        test_vector_store(index)
        
        print(f"\n💡 Usage Example:")
        print(f"```python")
        print(f"from llama_index.core import StorageContext")
        print(f"from llama_index.core import VectorStoreIndex")
        print(f"")
        print(f"# Load the vector store")
        print(f"storage_context = StorageContext.from_defaults(persist_dir='{PERSIST_DIR}')")
        print(f"index = VectorStoreIndex.from_storage_context(storage_context)")
        print(f"")
        print(f"# Create query engine")
        print(f"query_engine = index.as_query_engine()")
        print(f"")
        print(f"# Query the tables")
        print(f"response = query_engine.query('Your question about the tables')")
        print(f"print(response)")
        print(f"```")