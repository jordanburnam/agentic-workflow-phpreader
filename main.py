import os
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from getpass import getpass



#1. Set up Google API key
api_key = getpass("Enter your Google AI API Key:") 
os.environ["GOOGLE_API_KEY"] = api_key # This is used by the LLM and the embeddings 

#2. Initialize LLM
llm = GoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0)

#3. Load PHP files
print("Loading PHP Files ....")
php_directory = "../macrovo/app-legacy/server/src"
loader = DirectoryLoader(php_directory, glob="**/*.php", show_progress=True)
documents = loader.load()
print(f"Loaded {len(documents)} php files.")

#4. Split PHP files into chunks
print("Splitting php files into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Split php files into {len(chunks)} chunks.")

#5. Store in ChromaDB with Google Generative AI embeddings
print("Storing chunks in ChromaDB...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print("ChromaDB setup complete!")

#6. Create Retrieval Chain (New API)
retriever = vector_store.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


print("Initializing agent...")
agent = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory, # Adding conversational memory to help with context when it is answering queries
)
print("Agent is ready!")

while True:
    query = input("\nEnter your question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    response =  agent.invoke({"question": query})
    print("\nAnswer:", response["answer"])
