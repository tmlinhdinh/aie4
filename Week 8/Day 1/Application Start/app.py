### Import Section ###
import uuid
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_qdrant import QdrantVectorStore

import chainlit as cl
from chainlit.types import AskFileResponse


### Global Section ###
set_llm_cache(InMemoryCache())

rag_system_prompt_template = """\
You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existance of context.
"""

rag_message_list = [
    {"role" : "system", "content" : rag_system_prompt_template},
]

rag_user_prompt_template = """\
Question:
{question}
Context:
{context}
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt_template),
    ("human", rag_user_prompt_template)
])


class VectorDatabase:
    def __init__(self, embeddings: OpenAIEmbeddings()) -> None:
        self.embeddings = embeddings
        
    async def build_retriever(self, docs) -> None:
        collection_name = f"pdf_to_parse_{uuid.uuid4()}"
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        # Adding cache!
        store = LocalFileStore("./cache/")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.embeddings, store, namespace=self.embeddings.model
        )

        # Typical QDrant Vector Store Set-up
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=cached_embedder)
        vectorstore.add_documents(docs)
        
        return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    
    
class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.retriever = vector_db_retriever
        
    async def arun_pipeline(self, user_query: str):
        retrieval_augmented_qa_chain = (
                {"context": itemgetter("question") | self.retriever, "question": itemgetter("question")}
                | chat_prompt | self.llm | StrOutputParser()
            )
        
        async def generate_response():
            async for chunk in retrieval_augmented_qa_chain.astream({"question": user_query}):
                yield chunk

        return {"response": generate_response()}
    

def process_pdf_file(file: AskFileResponse):
    import tempfile
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf") as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(file.content)
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    Loader = PyMuPDFLoader
    loader = Loader(temp_file_path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    
    return docs
    

### On Chat Start (Session Start) Section ###
@cl.on_chat_start
async def on_chat_start():
    """ SESSION SPECIFIC CODE HERE """
    files = None
    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["pdf"],
            max_size_mb=2,
            timeout=180,
        ).send()
    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()
    docs = process_pdf_file(file)
    print(f"Processing {len(docs)} text chunks")

    # Create a dict vector store
    vector_db = VectorDatabase(embeddings=OpenAIEmbeddings(model="text-embedding-3-small"))
    vector_db = await vector_db.build_retriever(docs)

    # Create a chain
    retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        vector_db_retriever=vector_db
    )
    
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", retrieval_augmented_qa_pipeline)


### Rename Chains ###
@cl.author_rename
def rename(orig_author: str):
    """ RENAME CODE HERE """
    rename_dict = {"LLMMathChain": "Albert Einstein", "Chatbot": "Assistant"}
    return rename_dict.get(orig_author, orig_author)


### On Message Section ###
@cl.on_message
async def main(message: cl.Message):
    """
    MESSAGE CODE HERE
    """
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")
    result = await chain.arun_pipeline(message.content)

    async for stream_resp in result["response"]:
        await msg.stream_token(stream_resp)

    await msg.send()
    