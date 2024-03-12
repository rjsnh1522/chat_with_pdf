#!/usr/bin/env python3
import os
import glob
from typing import List
import multiprocessing
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


class FileIngester():

    def __init__(self, streamlit=None):
        load_dotenv()
        # Load environment variables
        self.persist_directory = os.getenv('PERSIST_DIRECTORY', 'db')
        self.source_directory = os.getenv('SOURCE_DIRECTORY', 'source_documents')
        self.embeddings_model_name = os.getenv('EMBEDDINGS_MODEL_NAME', 'hkunlp/instructor-xl')
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.streamlit = streamlit
    
    def get_mappings(self):
        LOADER_MAPPING = {
            ".csv": (CSVLoader, {}),
            ".doc": (UnstructuredWordDocumentLoader, {}),
            ".docx": (UnstructuredWordDocumentLoader, {}),
            ".enex": (EverNoteLoader, {}),
            ".eml": (MyElmLoader, {}),
            ".epub": (UnstructuredEPubLoader, {}),
            ".html": (UnstructuredHTMLLoader, {}),
            ".md": (UnstructuredMarkdownLoader, {}),
            ".odt": (UnstructuredODTLoader, {}),
            ".pdf": (PyMuPDFLoader, {}),
            ".ppt": (UnstructuredPowerPointLoader, {}),
            ".pptx": (UnstructuredPowerPointLoader, {}),
            ".txt": (TextLoader, {"encoding": "utf8"}),
            # Add more mappings for other file extensions and loaders as needed
        }
        return LOADER_MAPPING
    
    def remove_files_from_source_directory(self):
        try:
            # Iterate over all files in the folder
            for filename in os.listdir(self.source_directory):
                file_path = os.path.join(self.source_directory, filename)

                # Check if the file is a regular file
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

            print(f"All files in {self.source_directory} have been deleted.")
        except Exception as e:
            print(f"Error: {e}")

    def ingest(self):
         # Create embeddings
        self.streamlit.success("Injesting the file provided")
        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)
        self.streamlit.success("****Doing embedding *****")
        if self.does_vectorstore_exist(self.persist_directory):
            self.streamlit.success("Db exists")
            # Update and store locally vectorstore
            print(f"Appending to existing vectorstore at {self.persist_directory}")
            db = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
            collection = db.get()
            texts = self.process_documents([metadata['source'] for metadata in collection['metadatas']])
            print(f"Creating embeddings. May take some minutes...")
            self.streamlit.success("adding texts to vector database")
            db.add_documents(texts)
        else:
            # Create and store locally vectorstore
            print("Creating new vectorstore")
            self.streamlit.success("Creating new vectorstore")
            texts = self.process_documents()
            print(f"Creating embeddings. May take some minutes...")
            self.streamlit.success("Creating embeddings. May take some minutes...")
            db = Chroma.from_documents(texts, embeddings, persist_directory=self.persist_directory)
        db.persist()
        db = None

        print(f"Ingestion complete! You can now run privateGPT.py to query your documents")
        self.streamlit.success("Ingestion complete! You can now run privateGPT.py to query your documents")
        self.remove_files_from_source_directory()
    

    def load_single_document(self, file_path: str) -> List[Document]:
        ext = "." + file_path.rsplit(".", 1)[-1]
        loader_mappings = self.get_mappings()
        if ext in loader_mappings:
            loader_class, loader_args = loader_mappings[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()

        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_documents(self, source_dir: str, ignored_files: List[str] = []) -> List[Document]:
        """
        Loads all documents from the source documents directory, ignoring specified files
        """
        all_files = []
        for ext in self.get_mappings():
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
            )
        filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

        
        results = []
        self.streamlit.success("I am in multi processor")
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for file_path in filtered_files:
                doc = self.load_single_document(file_path)
                self.streamlit.success(all_files)
                results.extend(doc)
                pbar.update()
        self.streamlit.success("loaded all the documents")
        return results

    def process_documents(self, ignored_files: List[str] = []) -> List[Document]:
        """
        Load documents and split in chunks
        """
        documents = self.load_documents(self.source_directory, ignored_files)
        if not documents:
            print("No new documents to load")
            exit(0)
        self.streamlit.success("Spliting texts of file to before start doing embeddings")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks of text (max. {self.chunk_size} tokens each)")
        return texts

    def does_vectorstore_exist(self, persist_directory: str) -> bool:
        """
        Checks if vectorstore exists
        """
        self.streamlit.success(f"Db exists, {persist_directory}")
        if os.path.exists(os.path.join(persist_directory, 'index')):
            if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
                list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
                list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
                # At least 3 documents are needed in a working vectorstore
                if len(list_index_files) > 3:
                    return True
        return False
