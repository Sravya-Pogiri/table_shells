# Create SAPEmbeds_web.py - same functionality, web compatible
import streamlit as st
import ollama
from langchain_ollama.llms import OllamaLLM
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sentence_transformers import SentenceTransformer

import unstructured.partition
from unstructured.partition.pdf import partition_pdf

chunking_model = SentenceTransformer("all-MiniLM-L6-v2")
llm_model = OllamaLLM(model = "llama3.2")


class SAPEmbedsWeb:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.chunks = []
        self.vectors = None

    def smart_chunking(self, text, chunk_size = 400, overlap = 80):
        lines = text.split('\n')
        chunks = []
        current_chunk = ""

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if len(current_chunk + '\n' + line) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                if overlap > 0:
                    overlap_lines = current_chunk.split('\n')[-2:]
                    current_chunk = '\n'.join(overlap_lines) + '\n' + line
                else:
                    current_chunk = line
            else:
                current_chunk += '\n' + line if current_chunk else line

        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def og_process_pdf(self, file_path: str, chunk_length: int = 300, chunk_overlap: int = 80):
        elements = partition_pdf(filename = file_path)

        ele_dict = []
        for element in elements:
            ele_dict.append({
                "type": getattr(element, "category", ""),
                "text": getattr(element, "text", "")
            })
        element_type = set()
        for element in ele_dict:
            element_type.add(element["type"])

        full = "\n\n".join([element["text"] for element in ele_dict if "text" in element])

        chunks = []
        start = 0
        while start < len(full):
            end = min(start + chunk_length, len(full))
            chunks.append(full[start:end])
            start += chunk_length - chunk_overlap

        return chunks
        
    def process_pdf(self, file_path: str, chunk_length: int = 300, chunk_overlap: int = 80):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text()
        
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
    
    def retrieve(self, query, top=1):
        if not self.chunks or self.vectors is None:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        top_indices = np.argsort(similarities)[-top:][::-1]
        return [self.chunks[i] for i in top_indices]
    
    def analyzeLLM(self, query):
        best_chunks = self.retrieve(query, top=1)
        results = []
        
        for chunk in best_chunks:
            prompt = f"""Answer the question using the given chunk (no outside information) Remember the goal of this is to generate empty table shells and I need to use the RAG model to extract the variables that will be used for the table generation
            Question: {query}
            Text: {chunk}
            Answer:"""
            
            try:
                response = ollama.chat(model='llama3.2', messages=[{
                    'role': 'user',
                    'content': prompt
                }])
                # Clean the response to extract just the content
                clean_response = self.clean_response(response['message']['content'])
                results.append(clean_response)
            except Exception as e:
                results.append(f"Error: {str(e)}")
        
        return results
    
    def clean_response(self, response_text):
        """Clean up the response to extract just variable names"""
        lines = response_text.split('\n')
        variables = []
        
        for line in lines:
            line = line.strip()
            # Match numbered lists or bullet points
            if re.match(r'^\d+\.\s*', line) or line.startswith('- '):
                # Extract the variable name after the number/bullet
                clean_var = re.sub(r'^\d+\.\s*|-\s*', '', line)
                if clean_var and len(clean_var) > 1:
                    variables.append(clean_var)
        
        return variables if variables else [response_text]

# Streamlit App
def main():
    st.title("Doc2Table AI System")
    st.markdown("Fully functional RAG AI system that will generate table shells for medical purposes")

    uploaded_file = st.file_uploader(
        "Upload Your Documents (PDF)",
        type = 'pdf'
    )
    
    # Initialize the model
    if 'sap_model' not in st.session_state:
        st.session_state.sap_model = SAPEmbedsWeb()
        # Load the PDF
        with st.spinner("Loading SAP.pdf..."):
            st.session_state.chunks = st.session_state.sap_model.process_pdf("./SAP.pdf")
        st.success(f"Loaded {len(st.session_state.chunks)} chunks from SAP.pdf")

    if uploaded_file:
        file_type = uploaded_file.type
        with st.spinner("Processing uploaded document..."):
            if file_type == "application/pdf":
                import tempfile
                with tempfile.NamedTemporaryFile(delete = False, suffix = '.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                st.session_state.chunks = st.session_state.sap_model.process_pdf(tmp_path)
            else:
                st.error("Unsupported file type.")
                return
        st.success(f"Loaded {len(st.session_state.chunks)} chunks obtained from document") 
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if st.session_state.chunks:
        # Chat interface
        st.subheader("Ask a Question")
    
        # Sample questions
        st.markdown("**Sample questions:**")
        sample_questions = [
            "What are the variables in the demographics section?",
            "What are the prior cancer therapies mentioned?",
            "What are the medical history variables?",
            "What are the stratification factors?"
        ]
    
        # Create columns for sample questions
        cols = st.columns(2)
        for i, q in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(q, key=f"sample_{hash(q)}"):
                    st.session_state.current_question = q
        
        # Question input
        question = st.text_input(
            "Enter your question:", 
            value=st.session_state.get('current_question', ''),
            placeholder="Type your question here..."
        )
        
        # Create columns for buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            ask_button = st.button("Ask Question", type="primary")
        
        with col2:
            clear_history = st.button("Clear History", type="secondary")
        
        # Handle clear history
        if clear_history:
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
        
        # Handle question asking
        if ask_button and question:
            with st.spinner("Analyzing document..."):
                results = st.session_state.sap_model.analyzeLLM(question)
            
            # Store in chat history
            st.session_state.chat_history.append((question, results))
            
            # Clear the current question from session state
            if 'current_question' in st.session_state:
                del st.session_state.current_question
            
            st.success("Question processed! Check results below.")
            st.rerun()
        
        # Display current results (most recent)
        if st.session_state.chat_history:
            st.subheader("Latest Results:")
            latest_q, latest_a = st.session_state.chat_history[-1]
            
            st.write(f"**Question:** {latest_q}")
            
            for i, result in enumerate(latest_a):
                if isinstance(result, list) and result:
                    st.write(f"**Variables found:**")
                    for j, var in enumerate(result, 1):
                        st.write(f"{j}. {var}")
                else:
                    st.write(result)
        
        # Chat history section
        if st.session_state.chat_history:
            st.subheader(f"Conversation History ({len(st.session_state.chat_history)} conversations)")
            
            # Option to show/hide history
            show_full_history = st.checkbox("Show full conversation history", value=False)
            
            if show_full_history:
                # Display all conversations in reverse order (newest first)
                for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
                    conversation_num = len(st.session_state.chat_history) - i
                    
                    with st.expander(f"Conversation {conversation_num}: {q[:60]}{'...' if len(q) > 60 else ''}"):
                        st.write(f"**Question:** {q}")
                        st.write("**Answer:**")
                        
                        for result in a:
                            if isinstance(result, list) and result:
                                st.write("Variables found:")
                                for j, var in enumerate(result, 1):
                                    st.write(f"  {j}. {var}")
                            else:
                                st.write(result)
                        
                        # Add timestamp if you want
                        import datetime
                        st.caption(f"Asked at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                # Show just the last 3 conversations
                recent_conversations = st.session_state.chat_history[-3:]
                st.write("Showing last 3 conversations (check box above to see all)")
                
                for i, (q, a) in enumerate(reversed(recent_conversations)):
                    conversation_num = len(st.session_state.chat_history) - i
                    
                    with st.expander(f"Recent {len(recent_conversations) - i}: {q[:50]}{'...' if len(q) > 50 else ''}"):
                        st.write(f"**Q:** {q}")
                        for result in a:
                            if isinstance(result, list) and result:
                                for j, var in enumerate(result, 1):
                                    st.write(f"{j}. {var}")
                            else:
                                st.write(result)
        
        # Download conversation history
        if st.session_state.chat_history:
            st.subheader("Export Options")
            
            if st.button("Download Conversation History"):
                # Create downloadable content
                history_text = "# Conversation History\n\n"
                
                for i, (q, a) in enumerate(st.session_state.chat_history, 1):
                    history_text += f"## Conversation {i}\n"
                    history_text += f"**Question:** {q}\n\n"
                    history_text += f"**Answer:**\n"
                    
                    for result in a:
                        if isinstance(result, list) and result:
                            for j, var in enumerate(result, 1):
                                history_text += f"{j}. {var}\n"
                        else:
                            history_text += f"{result}\n"
                    
                    history_text += "\n---\n\n"
                
                st.download_button(
                    label="📄 Download as Text File",
                    data=history_text,
                    file_name=f"conversation_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    else:
        st.info("Please upload the proper documents to start.")

if __name__ == "__main__":
    main()