from ollama import Client
import json
import chromadb
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

client = chromadb.PersistentClient()
remote_client = Client(host=f"http://172.16.8.170:11434")
collection = client.get_or_create_collection(name="articles_demo")
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20, separators=["."])
with open("counter.txt","r") as f:
    counter = int(f.read().strip())

    print("Reading articles.jsonl and generating embeddings...")
    with open("articles.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i < counter:  
                print("Skipping already processing articles:",i)
                continue 
            counter += 1
            article = json.loads(line)
            content = article["content"]
            sentences=text_splitter.split_text(content)
            for each_Sentence in sentences:

                response = remote_client.embed(model="nomic-embed-text", input=f"search_document: {each_Sentence}")
                embedding = response["embeddings"][0]

                collection.add(
                    ids=[f"article_{i}"],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{"title": article["title"]}],
                )
        

print("Database built successfully!")
with open("counter.txt", "w") as f:
    f.write(str(counter))

while True:
    print("--------------------------------")
    query = input("🤖 🧠 : how may i help you ? \n")
    if query == "break":
        break
    # query = "what are different problems provinces of nepal are facing?"
    #query = "are there any predicted hindrance for upcoming election ?"
    query_embed = remote_client.embed(model="nomic-embed-text", input=f"query: {query}")["embeddings"][0]
    results = collection.query(query_embeddings=[query_embed], n_results=1)
    #print(f"\nQuestion: {query}")
    #print(f'\n Title : {results["metadatas"][0][0]["title"]} \n {results["documents"][0][0]} ')

    context='\n'.join(results["documents"][0])
    prompt = f"""You are a helpful assistant. Answer the question based on the context provided.
    Use the information in the context to form your answer.
    Answer strictly using the context.
    If the context indicates something is unlikely or not happening, clearly state that.
    Only say "I don't know" if the topic is completely unrelated.
    
    Context: {context}
    Question: {query}
    Answer:"""
    response = remote_client.generate(
            model="qwen3:4b-q4_K_M",
            prompt=prompt,
            options={
                "temperature": 0.1
            }
        )

    answer = response['response']

    print(answer)
