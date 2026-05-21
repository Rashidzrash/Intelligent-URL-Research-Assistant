from uuid import uuid4

from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from prompt import PROMPT, EXAMPLE_PROMPT
from langchain_classic.chains.qa_with_sources.loading import load_qa_with_sources_chain




load_dotenv()
chunk_size = 1000
VECTOR_STORE_DIRECTORY=Path(__file__).parent / 'resourses/vectorestore'
collection_name='real_state'
llm=None
vector_store=None

load_dotenv()
def initialize_components():
    global llm
    global vector_store
    if llm is None:
        llm=ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=0.9,
        max_tokens=500
        )
    if vector_store is None:
        ef=HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'trust_remote_code': True,
                      }
        )
        vector_store=Chroma(
        collection_name=collection_name,
        persist_directory=str(VECTOR_STORE_DIRECTORY),
        embedding_function=ef
        )


def processed(urls):
    yield 'intializing components'
    initialize_components()
    vector_store.reset_collection()
    yield 'loading data'
    # loder = UnstructuredURLLoader(urls=urls)
    loader = WebBaseLoader(
        urls,
        header_template={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/120.0 Safari/537.36"
            )
        }
    )
    data=loader.load()
    yield 'splitting data'
    text_spliter=RecursiveCharacterTextSplitter(
                  chunk_size=chunk_size,
                  separators=['\n\n','\n','.',' '],
    )
    doc=text_spliter.split_documents(data)
    uuids=[str(uuid4()) for _ in range(len(doc))]
    yield 'saving data'
    vector_store.add_documents(doc,ids=uuids)

def generate_answers(query):

    if vector_store is None:
        raise RuntimeError("Vector database is not initialized")

    # Create custom QA chain
    qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff",
                                          prompt=PROMPT,
                                          document_prompt=EXAMPLE_PROMPT)
    chain = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, retriever=vector_store.as_retriever(),
                                        reduce_k_below_max_tokens=True, max_tokens_limit=8000,
                                        return_source_documents=True)
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources_docs = [doc.metadata['source'] for doc in result['source_documents']]

    return result['answer'], sources_docs
if __name__ == '__main__':
    urls = [
        "https://www.cnbc.com/2026/05/06/mortgage-rates-hit-the-highest-level-in-a-month-causing-lower-income-homebuyers-to-drop-out.html",
        "https://www.cnbc.com/2026/04/29/fed-interest-rate-decision-april-2026.html"
    ]
    processed(urls)
    answer,source=generate_answers('Tell me the mortage rate')
    print(answer)
    print(source)

