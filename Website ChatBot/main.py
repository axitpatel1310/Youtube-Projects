import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re

def scrape_website(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator=' ')
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except requests.RequestException as e:
        return f"Error scraping website: {str(e)}"

def answer_question(context, question):
    try:
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        
        result = qa_pipeline(question=question, context=context)
        
        return result['answer']
    except Exception as e:
        return f"Error answering question: {str(e)}"

def chatbot(url, question):
    context = scrape_website(url)
    
    if "Error" in context:
        return context
    
    answer = answer_question(context, question)
    
    return answer

if __name__ == "__main__":
    example_url = "https://en.wikipedia.org/wiki/ChatGPT"
    while True:
        example_question = input("Enter A Question: ")
        if example_question.lower() == "exit":
            break
        response = chatbot(example_url, example_question)
        print(f"Question: {example_question}")
        print(f"Answer: {response}")
