import asyncio
import os
from config import load_configuration
from embedding import CustomEmbeddings
from scraping import scrape_website
from retrieval import query_retrieval
from utils import pretty_print
from langchain_chroma import Chroma

async def main():
    """Main function to orchestrate the scraping, processing, and querying."""
    groq_api_key, api_url, headers = load_configuration()
    page_urls = ["sigsystems.com/gpu-rental"]
    persist_dir = 'docs/chroma'
    
    # Check if the persist directory exists and is not empty
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("Documents already exist. Skipping scraping.")
        # Load the existing database
        embeddings = CustomEmbeddings(api_url, headers)
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        if page_urls:
            docs = await scrape_website(page_urls)
            if docs:
                print(f"Fetched {len(docs)} documents")
                embeddings = CustomEmbeddings(api_url, headers)
                db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
                print(f"Number of documents in the collection: {db._collection.count()}")
            else:
                print("No documents fetched.")
                return
        else:
            print("No URLs found.")
            return

    query = "Give short summary about this page"
    answers = query_retrieval(db, query, groq_api_key)
    pretty_print(answers)
    

    query = "make it consice"
    answers = query_retrieval(db, query, groq_api_key)
    pretty_print(answers)

    query = "list products"
    answers = query_retrieval(db, query, groq_api_key)
    pretty_print(answers)
    """for x in range(len(db.get()["ids"])):
    # print(db.get()["metadatas"][x])
        doc = db.get()["metadatas"][x]
        if doc["source"]=="https://github.com/langchain-ai/langchain/discussions/14153":
            li=db.get()["documents"][x]
            print(li)"""
    

if __name__ == "__main__":
    asyncio.run(main())
