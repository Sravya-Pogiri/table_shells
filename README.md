# IDSWG Table Shells - RAG-Based Document Analysis

A comprehensive Retrieval-Augmented Generation (RAG) system for analyzing Statistical Analysis Plan (SAP) documents and extracting variables for table shell creation.

## Overview

This project implements advanced RAG models to parse and analyze clinical trial documents, specifically focusing on extracting demographic variables, medical history, prior cancer therapies, stratification factors, and extent of cancer information from SAP documents. The system is designed for creating structured table shells used in clinical research.

## Features

- **Document Processing**: PDF parsing with intelligent text extraction and chunking
- **Semantic Search**: Vector embeddings using SentenceTransformers for accurate content retrieval
- **Multiple RAG Models**: Specialized models for different document types (SAP, CRF)
- **Variable Extraction**: Automated identification and classification of numerical and categorical variables
- **Web Interface**: Streamlit-based dashboard for interactive document analysis
- **Chat Interface**: Real-time Q&A with documents
- **Section Analysis**: Predefined queries for specific document sections

## Project Structure

```
IDSWG_TableShells/
├── SAPEmbeds.py           # Main RAG model for SAP document analysis
├── streamlit_app.py       # Web interface for document interaction
├── SAP_RAG.py            # Alternative RAG implementation
├── CRFAI/
│   ├── CRFAI.py          # CRF-specific analysis model
│   └── CRFAI.pdf         # Sample CRF document
├── table_shells/
│   └── shelltable.py     # Table generation utilities
├── SAP.pdf               # Sample SAP document
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sravya-Pogiri/table_shells.git
cd IDSWG_TableShells
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install additional system dependencies:
```bash
# For PDF processing (macOS)
brew install poppler

# For enhanced PDF parsing
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

## Usage

### Command Line Interface

Run the basic RAG model:
```bash
python SAPEmbeds.py
```

This will start an interactive session where you can ask questions about the SAP document.

### Web Interface

Launch the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

The web interface provides:
- **Section Analysis**: Predefined queries for Demographics, Prior Cancer Therapies, Medical History, Stratification Factors, and Extent of Cancer
- **Chat Interface**: Free-form questions about the document
- **Variable Classification**: Automatic categorization of variables as numerical or categorical

### Alternative RAG Models

Run the alternative SAP RAG model:
```bash
python SAP_RAG.py
```

Run the CRF analysis model:
```bash
cd CRFAI
python CRFAI.py
```

## Core Components

### SAPEmbeds.py - Main RAG Engine

Key functions:
- `embed_chunks()`: Converts text chunks to vector embeddings
- `build()`: Creates FAISS index for similarity search
- `retrieve()`: Finds most relevant chunks for queries
- `analyzeLLM()`: Generates answers using LLM with retrieved context
- `analyzeLLM_variables()`: Specialized variable extraction
- `build_cosine()`: Cosine similarity-based indexing
- `process_pdf()`: PDF parsing and chunking

### Vector Embeddings

The system uses SentenceTransformers (`all-MiniLM-L6-v2`) to create 384-dimensional embeddings that capture semantic meaning of text chunks. FAISS provides efficient similarity search with both L2 distance and cosine similarity options.

### Document Processing

Documents are processed using the `unstructured` library with PDF partitioning capabilities. Text is chunked with configurable overlap to maintain context while enabling efficient retrieval.

## Configuration

### Model Settings

- **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **LLM**: `llama3.2` (via Ollama)
- **Chunk Size**: 300 characters (configurable)
- **Chunk Overlap**: 80 characters (configurable)
- **Retrieval**: Top-k chunks (default k=1 for basic queries, k=3 for variable extraction)

### Predefined Sections

The system includes optimized queries for:
- Demographics (age, sex, race, ethnicity, height, weight)
- Prior Cancer Therapies (treatment history, medications)
- Medical History (prior conditions, surgeries)
- Stratification Factors (randomization variables)
- Extent of Cancer (staging, tumor characteristics)

## API Reference

### Core Functions

```python
# Basic document analysis
results = analyzeLLM(chunks, query)

# Variable extraction with structured output
variables = analyzeLLM_variables(chunks, query, section_name)

# Process PDF into chunks
chunks = process_pdf(file_path, chunk_length=300, chunk_overlap=80)

# Build cosine similarity index
index = build_cosine(embeddings)
```

### Variable Structure

Extracted variables follow this format:
```python
{
    'name': 'Variable Name',
    'type': 'Numerical' or 'Categorical',
    'description': 'Detailed description'
}
```

## Requirements

### Python Dependencies

- streamlit >= 1.28.0
- sentence-transformers >= 2.2.0
- faiss-cpu >= 1.7.0
- langchain-ollama >= 0.1.0
- unstructured >= 0.10.0
- pandas >= 1.5.0
- numpy >= 1.21.0
- python-docx >= 0.8.11
- tabulate >= 0.9.0

### System Dependencies

- Poppler (for PDF processing)
- Ollama (for LLM inference)
- Optional: detectron2 (for enhanced PDF parsing)

## Performance Considerations

- **Memory Usage**: Embeddings require ~1.5MB per 1000 chunks
- **Processing Speed**: ~2-5 seconds per query depending on document size
- **Accuracy**: Cosine similarity generally provides better semantic matching than L2 distance
- **Scalability**: FAISS enables efficient search across large document collections

## Troubleshooting

### Common Issues

1. **PDF Processing Errors**: Ensure Poppler is installed and accessible
2. **Memory Issues**: Reduce chunk size or implement batch processing
3. **Slow Performance**: Consider using GPU acceleration for embeddings
4. **Import Errors**: Verify all dependencies are installed in the virtual environment

### Error Handling

The system includes robust error handling for:
- Malformed PDF documents
- Network timeouts during LLM inference
- Invalid JSON responses
- Missing document sections

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of academic research and is intended for educational and research purposes.

## Contact

For questions or issues, please contact the development team or open an issue on the repository.

## Acknowledgments

- Built with SentenceTransformers for semantic embeddings
- FAISS for efficient similarity search
- Streamlit for web interface
- Ollama for local LLM inference
- Unstructured for document processing