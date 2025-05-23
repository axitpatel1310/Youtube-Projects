import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

embedder = SentenceTransformer('all-MiniLM-L6-v2') 
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad') 
index = faiss.IndexFlatL2(384)

chunks = []
embeddings = None

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        if not text:
            raise ValueError("No text could be extracted from the PDF.")
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def chunk_text(text, chunk_size=100):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def store_in_vector_store(chunks_param):
    global embeddings, chunks
    chunks = chunks_param 
    embeddings = embedder.encode(chunks)
    index.reset()  
    index.add(np.array(embeddings, dtype='float32'))
    return chunks

def query_pdf(question):
    if not chunks:
        return "No PDF processed yet."
    question_embedding = embedder.encode([question])[0]
    distances, indices = index.search(np.array([question_embedding], dtype='float32'), k=1)
    top_chunk = chunks[indices[0][0]]
    result = qa_pipeline(question=question, context=top_chunk)
    return result['answer']

def main():
    global chunks
    
    file_path = input("Enter the path to your PDF file: ").strip()
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return
    
    print("Extracting text from PDF...")
    try:
        pdf_text = extract_text_from_pdf(file_path)
    except Exception as e:
        print(e)
        return
    
    print("Chunking text...")
    chunks_local = chunk_text(pdf_text)
    if not chunks_local:
        print("Error: No text chunks created.")
        return
    
    print("Storing in vector store...")
    store_in_vector_store(chunks_local)
    print("PDF processed successfully!")
    
    print("\nAsk questions about your PDF (type 'exit' to quit):")
    while True:
        question = input("Question: ").strip()
        if question.lower() == 'exit':
            print("Exiting...")
            break
        if not question:
            print("Please enter a question.")
            continue
        answer = query_pdf(question)
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()
