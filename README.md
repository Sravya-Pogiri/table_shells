# IDSWG Table Shells - Dual RAG System for Clinical Document Analysis

An **Advanced RAG (Retrieval-Augmented Generation)** system with dual architecture for analyzing Statistical Analysis Plan (SAP) documents and generating clinical trial table shells. The system leverages both TF-IDF and neural embeddings with specialized LLMs for variable extraction and hierarchical table generation.

## Overview

This project implements a **dual RAG architecture** with two specialized subsystems designed for clinical trial document analysis:

### RAG System 1: Variable Extractor (TF-IDF-Based)
- **Purpose**: Extract clinical variables from SAP documents
- **Embeddings**: TF-IDF via scikit-learn (sparse vectors, 1000 max features)
- **Retrieval**: Cosine similarity search
- **LLM**: Llama 3.2 (via Ollama)
- **Parser**: PyPDF2 for PDF text extraction
- **Chunking**: Sliding window (400 chars, 80 char overlap)

### RAG System 2: Table Generator (Neural Embeddings)
- **Purpose**: Generate table shells from clinical protocols
- **Embeddings**: BAAI/bge-base-en-v1.5 (768-dimensional dense vectors)
- **Framework**: LlamaIndex with VectorStoreIndex
- **LLM**: Mistral (via Ollama)
- **Parser**: Docling for hierarchical table extraction
- **Features**: Preserves nested table structures, markdown conversion

## System Classification

This is an **Advanced RAG** system with modular characteristics:
- **Multiple embedding strategies**: TF-IDF (sparse) and neural (dense)
- **Specialized retrieval**: Cosine similarity with domain-specific chunking
- **Dual LLM approach**: Llama 3.2 for variables, Mistral for tables
- **Cross-system integration**: Variable extraction feeds table generation
- **Persistent indexing**: Cached vector stores for efficient reuse

## Features

- **Dual RAG Architecture**: TF-IDF + neural embeddings for complementary strengths
- **Variable Extraction Mode**: Automatic identification and classification of clinical variables
- **Table Shell Generation**: Hierarchical table parsing from clinical documents
- **Interactive Web Interface**: Streamlit dashboard with file upload capability
- **Cross-System Workflow**: Extract variables → Generate table shells
- **Type Classification**: Automatic categorization as Numerical/Categorical
- **Session State Management**: Preserves variables across extraction and table generation
- **Multiple Model Support**: Llama 3.2, Mistral via Ollama

## Project Structure

```
IDSWG_TableShells/
├── Doc2Table_AI_System.py      # Main dual RAG application (Variable Extractor + Table Generator)
├── table_shell_generator.py    # LlamaIndex-based table parsing with Mistral
├── SAPEmbeds.py                # Legacy RAG model (SentenceTransformers-based)
├── SAPEmbeds_simple.py         # Simplified TF-IDF RAG implementation
├── SAP_RAG.py                  # Alternative RAG implementation
├── SAP_Web.py                  # Web interface for SAP analysis
├── enhanced_streamlit_app.py   # Enhanced web UI with file upload
├── simple_app.py               # Basic Streamlit interface
├── working_streamlit_app.py    # Working Streamlit dashboard
├── EnhancedRAG.py              # Enhanced RAG with improved retrieval
├── WorkingRAG.py               # Stable RAG implementation
├── shelltable.py               # Table generation utilities
├── test_cleanup.py             # Cleanup utilities
├── docling_extractor.py        # Docling integration for hierarchical parsing
├── AI.ipynb                    # Jupyter notebook for AI experiments
├── RAGModel.ipynb              # RAG model development notebook
├── shelltable.ipynb            # Table shell notebook experiments
├── requirements.txt            # Python dependencies
├── CRFAI/
│   └── CRFAI.py                # CRF-specific analysis model
├── table_shells/
│   └── shelltable.py           # Table shell utilities
└── rag_storage/
    ├── default__vector_store.json  # Cached vector embeddings
    ├── docstore.json               # Document store
    ├── graph_store.json            # Graph relationships
    ├── image__vector_store.json    # Image embeddings
    └── index_store.json            # Index metadata
```

### Core Files

**Doc2Table_AI_System.py** - Main application with dual RAG architecture:
- `SAPEmbedsWeb` class: TF-IDF-based variable extraction
- `run_variable_extractor_app()`: UI for document analysis with extraction mode
- `extract_variables_for_table()`: Parses LLM responses into structured variables
- `determine_variable_type()`: Classifies variables as Numerical/Categorical
- `run_table_shell_app()`: Table generation interface
- `main()`: Application entry point with session state management

**table_shell_generator.py** - LlamaIndex-based table parser:
- `parse_clinical_data_hierarchical()`: Preserves nested table structures
- `markdown_to_dataframe()`: Converts LLM markdown to DataFrames
- `load_rag_query_engine()`: Cached query engine with index persistence
- Uses Mistral LLM and BAAI/bge-base-en-v1.5 embeddings

## Installation

### Prerequisites

- Python 3.8+ (tested with Python 3.12)
- Ollama installed and running locally
- Llama 3.2 and Mistral models available in Ollama
- Git

### Ollama Setup

1. Install Ollama from https://ollama.ai

2. Pull required models:
```bash
# For variable extraction
ollama pull llama3.2

# For table generation
ollama pull mistral
```

3. Verify models are available:
```bash
ollama list
```

### Python Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd IDSWG_TableShells
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install additional system dependencies (macOS):
```bash
# For PDF processing
brew install poppler
```

### Core Dependencies

**Dual RAG System:**
- `streamlit >= 1.28.0` - Web interface framework
- `ollama` - Local LLM inference
- `llama-index >= 0.10.0` - RAG framework for table generation
- `llama-index-embeddings-huggingface >= 0.2.0` - HuggingFace embeddings integration
- `docling` - Hierarchical document parsing

**Variable Extraction (RAG System 1):**
- `PyPDF2 >= 3.0.0` - PDF text extraction
- `scikit-learn >= 1.3.0` - TF-IDF vectorization
- `pandas >= 1.5.0` - Data manipulation
- `numpy >= 1.21.0` - Numerical operations

**Table Generation (RAG System 2):**
- `transformers >= 4.30.0` - HuggingFace model support
- `torch >= 2.0.0` - PyTorch backend

**Optional (Legacy Models):**
- `sentence-transformers >= 2.2.0` - Neural embeddings (SAPEmbeds.py)
- `faiss-cpu >= 1.7.0` - Vector similarity search
- `unstructured >= 0.10.0` - Document processing

## Usage

### Dual RAG Application (Recommended)

Launch the main application with both variable extraction and table generation:

```bash
streamlit run Doc2Table_AI_System.py
```

This provides:
- **Variable Extraction Mode**: Toggle to extract clinical variables from SAP documents
- **File Upload**: Upload PDF documents for analysis
- **Quick Questions**: Predefined queries for common SAP sections (Demographics, Medical History, etc.)
- **Free-form Chat**: Ask any question about the document
- **Variable Classification**: Automatic categorization as Numerical/Categorical
- **Table Generation**: Generate table shells using extracted variables
- **Cross-System Integration**: Variables from extraction feed directly into table generator

#### Workflow Example

1. **Upload SAP Document**: Use the file uploader in the sidebar
2. **Extract Variables**: 
   - Enable "Variable Extraction Mode"
   - Ask: "What are the demographic variables in this SAP?"
   - System extracts and classifies variables
3. **Generate Table Shell**:
   - Navigate to "Table Shell Generator" tab
   - Use extracted variables as input
   - System generates structured table with Mistral LLM

### Variable Extraction Only

For focused variable extraction without table generation:

```bash
python -c "from Doc2Table_AI_System import SAPEmbedsWeb; rag = SAPEmbedsWeb(); rag.build('path/to/sap.pdf'); results = rag.analyzeLLM_variables('Extract all demographic variables', 'Demographics')"
```

### Table Generation Only

For standalone table shell generation:

```bash
python table_shell_generator.py
```

### Alternative RAG Models

Run the legacy SentenceTransformers-based model:
```bash
python SAPEmbeds.py
```

Run the alternative SAP RAG model:
```bash
python SAP_RAG.py
```

Run the CRF analysis model:
```bash
cd CRFAI
python CRFAI.py
```

## Architecture Deep Dive

### RAG System 1: Variable Extractor

**Embedding Strategy:**
- **Type**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Implementation**: scikit-learn's TfidfVectorizer
- **Dimensionality**: Sparse vectors with max 1000 features
- **Advantages**: Fast, interpretable, works well for keyword-based queries

**Retrieval Process:**
1. Document chunked into 400-character segments with 80-character overlap
2. TF-IDF matrix computed for all chunks
3. Query vectorized using same TF-IDF model
4. Cosine similarity computed between query and chunk vectors
5. Top-k most similar chunks retrieved (k=3 for variable extraction)

**LLM Integration:**
- **Model**: Llama 3.2 (via Ollama)
- **Context**: Retrieved chunks + system prompt
- **Output**: Structured JSON with variable name, type, description
- **Post-processing**: `extract_variables_for_table()` parses LLM response into structured format

**Variable Classification:**
- `determine_variable_type()` classifies as Numerical or Categorical
- Heuristics: age, weight, height → Numerical; sex, race, ethnicity → Categorical

### RAG System 2: Table Generator

**Embedding Strategy:**
- **Type**: Dense neural embeddings
- **Model**: BAAI/bge-base-en-v1.5 (768-dimensional)
- **Framework**: LlamaIndex with HuggingFaceEmbedding wrapper
- **Advantages**: Semantic understanding, captures context, handles synonyms

**Retrieval Process:**
1. Documents parsed with Docling (preserves hierarchical table structures)
2. Neural embeddings computed for all document sections
3. VectorStoreIndex created and cached in `rag_storage/`
4. Query engine retrieves semantically similar sections
5. Retrieved context passed to Mistral LLM

**LLM Integration:**
- **Model**: Mistral (via Ollama)
- **Framework**: LlamaIndex QueryEngine
- **Context**: Retrieved document sections + query
- **Output**: Markdown-formatted table shells
- **Post-processing**: `markdown_to_dataframe()` converts to pandas DataFrames

**Hierarchical Parsing:**
- `parse_clinical_data_hierarchical()` preserves nested table structures
- Handles multi-level headers and grouped variables
- Maintains parent-child relationships in table columns

### Cross-System Integration

**Session State Management:**
```python
st.session_state['extracted_variables']  # From RAG System 1
st.session_state['generated_table']       # To RAG System 2
```

**Workflow:**
1. User uploads PDF → RAG System 1 extracts variables
2. Variables stored in session state
3. User navigates to Table Generator tab
4. RAG System 2 uses extracted variables as context
5. Mistral LLM generates table shell with proper structure

## Core Components

### Doc2Table_AI_System.py - Dual RAG Application

**Key Classes:**

`SAPEmbedsWeb` - TF-IDF RAG for variable extraction:
- `__init__()`: Initializes TF-IDF vectorizer and Ollama client
- `build(pdf_path)`: Processes PDF, creates chunks, computes TF-IDF matrix
- `retrieve(query, k=3)`: Returns top-k most similar chunks
- `analyzeLLM(query)`: Basic Q&A with document
- `analyzeLLM_variables(query, section_name)`: Extracts structured variables
- `process_pdf_content(text)`: Sliding window chunking (400 chars, 80 overlap)

**Key Functions:**

`extract_variables_for_table(llm_response)`:
- Parses LLM JSON response into structured variable list
- Handles multiple JSON formats and edge cases
- Returns list of dicts with name, type, description

`determine_variable_type(var_name, description)`:
- Classifies variables as Numerical or Categorical
- Uses keyword matching (age, weight → Numerical; sex, race → Categorical)
- Fallback: description length heuristics

`run_variable_extractor_app()`:
- Streamlit UI for variable extraction
- File upload, extraction mode toggle
- Quick questions sidebar for common SAP sections
- Session state management for extracted variables

`run_table_shell_app()`:
- Streamlit UI for table generation
- Uses LlamaIndex query engine with Mistral
- Integrates extracted variables from session state
- Interactive table editor with pandas DataFrames

`main()`:
- Application entry point
- Sets page config (must be first Streamlit command)
- Tab navigation between variable extractor and table generator
- Session state initialization

### table_shell_generator.py - LlamaIndex Table Parser

**Key Functions:**

`load_rag_query_engine(file_path, model_name="mistral")`:
- Creates LlamaIndex VectorStoreIndex from PDF
- Uses BAAI/bge-base-en-v1.5 embeddings
- Caches index in `rag_storage/` directory
- Returns QueryEngine for semantic search
- Decorated with `@st.cache_resource` for persistence

`parse_clinical_data_hierarchical(file_path)`:
- Uses Docling for hierarchical table extraction
- Preserves nested structures and parent-child relationships
- Returns structured representation of tables

`markdown_to_dataframe(markdown_text)`:
- Converts LLM markdown output to pandas DataFrame
- Handles various markdown table formats
- Fallback parsing for malformed tables

### Configuration

**Model Settings:**

**RAG System 1 (Variable Extractor):**
- **LLM**: llama3.2 (via Ollama)
- **Embeddings**: TF-IDF (scikit-learn, max 1000 features)
- **Chunk Size**: 400 characters
- **Chunk Overlap**: 80 characters
- **Retrieval**: Top-3 chunks (cosine similarity)

**RAG System 2 (Table Generator):**
- **LLM**: mistral (via Ollama)
- **Embeddings**: BAAI/bge-base-en-v1.5 (768-dim)
- **Framework**: LlamaIndex with VectorStoreIndex
- **Caching**: Persistent vector store in `rag_storage/`

**Predefined Sections:**

The variable extractor includes optimized queries for:
- **Demographics**: age, sex, race, ethnicity, height, weight
- **Prior Cancer Therapies**: treatment history, medications, radiation
- **Medical History**: prior conditions, surgeries, hospitalizations
- **Stratification Factors**: randomization variables, study arms
- **Extent of Cancer**: staging, tumor size, metastases, biomarkers

## API Reference

### Variable Extraction API

```python
from Doc2Table_AI_System import SAPEmbedsWeb

# Initialize RAG system
rag = SAPEmbedsWeb()

# Build index from PDF
rag.build('path/to/sap.pdf')

# Extract variables
results = rag.analyzeLLM_variables(
    query='Extract all demographic variables',
    section_name='Demographics'
)

# Results format:
# {
#     'variables': [
#         {'name': 'Age', 'type': 'Numerical', 'description': '...'},
#         {'name': 'Sex', 'type': 'Categorical', 'description': '...'}
#     ]
# }
```

### Table Generation API

```python
from table_shell_generator import load_rag_query_engine

# Load query engine
engine = load_rag_query_engine('path/to/protocol.pdf', model_name='mistral')

# Generate table shell
query = "Create a table shell for baseline demographics"
response = engine.query(query)

# Convert to DataFrame
from table_shell_generator import markdown_to_dataframe
df = markdown_to_dataframe(response.response)
```

### Variable Structure

Extracted variables follow this format:
```python
{
    'name': 'Variable Name',
    'type': 'Numerical' or 'Categorical',
    'description': 'Detailed description from SAP'
}
```

## Requirements

### Python Dependencies

**Core (Dual RAG):**
- streamlit >= 1.28.0
- ollama
- llama-index >= 0.10.0
- llama-index-embeddings-huggingface >= 0.2.0
- docling
- PyPDF2 >= 3.0.0
- scikit-learn >= 1.3.0
- pandas >= 1.5.0
- numpy >= 1.21.0
- transformers >= 4.30.0
- torch >= 2.0.0

**Optional (Legacy Models):**
- sentence-transformers >= 2.2.0
- faiss-cpu >= 1.7.0
- langchain-ollama >= 0.1.0
- unstructured >= 0.10.0
- python-docx >= 0.8.11
- tabulate >= 0.9.0

### System Dependencies

- **Poppler**: For PDF processing (install via `brew install poppler` on macOS)
- **Ollama**: For LLM inference (llama3.2 and mistral models required)

## Performance Considerations

**Variable Extractor (TF-IDF):**
- **Memory Usage**: ~500KB per 1000 chunks (sparse matrices)
- **Processing Speed**: 1-2 seconds per query
- **Accuracy**: Excellent for keyword-based queries, good for domain-specific terms
- **Scalability**: Handles large documents efficiently due to sparse representation

**Table Generator (Neural Embeddings):**
- **Memory Usage**: ~3MB per 1000 chunks (dense vectors, 768-dim)
- **Processing Speed**: 3-5 seconds per query (includes embedding + LLM)
- **Accuracy**: Superior semantic understanding, handles paraphrasing
- **Scalability**: Requires more memory but provides better semantic matching
- **Caching**: Vector stores persisted in `rag_storage/` for instant reuse

**Optimization Tips:**
- Use TF-IDF for keyword-heavy queries (variable names, specific terms)
- Use neural embeddings for semantic queries (concepts, descriptions)
- Cache query engines with `@st.cache_resource` in Streamlit
- Adjust chunk size/overlap based on document structure

## Troubleshooting

### Common Issues

**1. Ollama Connection Errors**
```bash
# Verify Ollama is running
ollama list

# Restart Ollama if needed
ollama serve
```

**2. Model Not Found**
```bash
# Pull required models
ollama pull llama3.2
ollama pull mistral
```

**3. PDF Processing Errors**
- Ensure Poppler is installed: `brew install poppler` (macOS)
- Check PDF is not encrypted or password-protected
- Verify PDF has extractable text (not scanned images)

**4. Memory Issues with Large Documents**
- Reduce chunk size in `process_pdf_content()` (default: 400 chars)
- Increase chunk overlap for better context (default: 80 chars)
- Clear cached vector stores in `rag_storage/` if stale

**5. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify Python version
python --version  # Should be 3.8+
```

**6. Streamlit ScriptRunContext Error**
- Ensure `st.set_page_config()` is first command in `main()`
- Avoid calling Streamlit functions at module level
- Use `st.rerun()` instead of deprecated `st.experimental_rerun()`

### Error Handling

The system includes robust error handling for:
- **Malformed PDFs**: Graceful fallback to text extraction
- **LLM Timeouts**: Retry logic with exponential backoff
- **Invalid JSON Responses**: Fallback parsing with regex
- **Missing Vector Stores**: Automatic rebuild from source documents
- **Session State Issues**: Automatic reinitialization

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with proper documentation
4. Add tests if applicable
5. Ensure code follows PEP 8 style guidelines
6. Submit a pull request with detailed description

## License

This project is part of academic research and is intended for educational and research purposes.

## Acknowledgments

**RAG System 1 (Variable Extractor):**
- scikit-learn for TF-IDF implementation
- PyPDF2 for PDF parsing
- Ollama for local Llama 3.2 inference

**RAG System 2 (Table Generator):**
- LlamaIndex for RAG framework
- HuggingFace for BAAI/bge-base-en-v1.5 embeddings
- Docling for hierarchical document parsing
- Ollama for local Mistral inference

**Web Framework:**
- Streamlit for interactive UI
- pandas for data manipulation