from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import ApifyDatasetLoader

from dotenv import load_dotenv

load_dotenv()


def apify_crawler_and_save(url: str, persist_directory: str, chunk_size: int = 512, chunk_overlap: int = 20,
                           dataset_id: str = '') -> Chroma:
    """
    Crawls the website with the given URL using Apify and creates embeddings for every text chunk in the crawled content.

    Args:
        url (str): The URL of the website to crawl.
        persist_directory (str): The directory in which to save the embeddings as a Chroma database.
        chunk_size (int, optional): The maximum number of characters to include in each text chunk. Defaults to 512.
        chunk_overlap (int, optional): The number of overlapping characters to include between adjacent text chunks. Defaults to 20.

    Returns:
        Chroma: A Chroma object representing the embedding's database.
    """

    if dataset_id:
        # load from dataset
        loader = apify_dataset_loader(dataset_id)
    else:
        # use apify to crawl
        loader = apify_crawl_loader(url)

    # write embeddings to disk

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    embedding = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()

    return vectordb


def apify_dataset_loader(dataset_id):
    loader = ApifyDatasetLoader(
        dataset_id=dataset_id,
        dataset_mapping_function=lambda item: Document(
            page_content=item["text"] or "", metadata={"source": item["url"]}
        ),
    )
    return loader

def apify_crawl_loader(url):
    apify = ApifyWrapper()
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={"startUrls": [{"url": url}]},
        dataset_mapping_function=lambda item: Document(
            page_content=item["text"] or "", metadata={"source": item["url"]}
        ),
    )
    return loader