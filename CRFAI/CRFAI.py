from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file: str) -> str:
    reader = PdfReader(pdf_file)
    text = "" 
    for page in reader.pages:  # This line was incorrectly indented
        if page.extract_text():
            text += page.extract_text()
    return text  # This line was incorrectly indented

pdf_path = r"C:\Users\hydin\OneDrive\Desktop\CRF\CRFAI.pdf"     #Change to actual paths
extracted_text = extract_text_from_pdf(pdf_path) #before you sleep set a alarm to remind yourself 
#to get rid of ALL of your thoughts fro the REST of your life and put phone in the desk drawers 
# and ONLY use insta on phone at the gym no where else even at church 

def create_chunks(text: str, chunk_size: int = 100, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += step
    return chunks

pdf_path = r"C:\Users\hydin\OneDrive\Desktop\CRF\CRFAI.pdf"  # replace with your actual path
text = extract_text_from_pdf(pdf_path)
chunks = create_chunks(text)

for i, chunk in enumerate(chunks[:3]):  # just show first 3 chunks
    print(f"--- Chunk {i+1} ---\n{chunk}\n")

# Note: You have this function defined twice, I'm keeping just one instance
def find_best_chunk(chunks: list[str], question: str) -> str:
    best_score = 0
    best_chunk = ""
    question_words = question.lower().split()
    for chunk in chunks:
        chunk_text = chunk.lower()
        score = sum(word in chunk_text for word in question_words)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk

def answer_question_from_pdf(pdf_path: str, question: str) -> str:
    text = extract_text_from_pdf(pdf_path)
    
    # Check if text extraction was successful
    if not text:
        return "Failed to extract text from the PDF."
        
    chunks = create_chunks(text)
    best_chunk = find_best_chunk(chunks, question)
    
    # Check if best_chunk is a default message
    if best_chunk == "No relevant information found." or not best_chunk:
        return best_chunk
    
    # Simple keyword-based answering approach
    question_lower = question.lower()
    chunk_lower = best_chunk.lower()
    
    import re  # This import should be at the top of the file, but I'm keeping it here as in your original code
    sentences = re.split(r'[.!?]', best_chunk)
    relevant_sentences = []
    
    # Look for sentences that contain keywords from the question
    keywords = [word for word in question_lower.split() if len(word) > 3]
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check if the sentence contains any keywords from the question
        if any(keyword in sentence.lower() for keyword in keywords):
            relevant_sentences.append(sentence)
    
    if relevant_sentences:
        answer = " ".join(relevant_sentences)
        return f"Based on the document, I found this information: {answer}"
    else:
        # If no specific sentences found, return the best chunk
        return f"I couldn't find a specific answer, but here's the most relevant section from the document:\n\n{best_chunk}"

# 5. Run the QA system
if __name__ == "__main__":
    pdf_path = r"C:\Users\hydin\OneDrive\Desktop\CRF\CRFAI.pdf"
    question = "What is the date of birth format?"

    # Call the function and capture the returned answer
    answer = answer_question_from_pdf(pdf_path, question)
    print("Answer:", answer) #change file path? 