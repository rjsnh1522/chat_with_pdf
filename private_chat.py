import chromadb
import os
import argparse
import time
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from constants import CHROMA_SETTINGS



class ChatResponse:
    def __init__(self) -> None:
        load_dotenv()
        self.model = os.getenv("MODEL", "mistral")
        # For embeddings model, the example uses a sentence-transformers model
        # https://www.sbert.net/docs/pretrained_models.html 
        # "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
        self.embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
        self.persist_directory = os.getenv("PERSIST_DIRECTORY", "db")
        self.target_source_chunks = int(os.getenv('TARGET_SOURCE_CHUNKS',4))
    
    def __load_query_model(self):
        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)
        db = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": self.target_source_chunks})
         # activate/deactivate the streaming StdOut callback for LLMs
        callbacks = [StreamingStdOutCallbackHandler()]

        llm = Ollama(model=self.model, callbacks=callbacks)

        qa = RetrievalQA.from_chain_type(llm=llm, 
                                         chain_type="stuff", 
                                         retriever=retriever, 
                                         return_source_documents=False)
        return qa
    
    
    def get_answer(self, user_question):
        qa = self.__load_query_model()
        
        # Get the answer from the chain
        start = time.time()
        res = qa(user_question)
        answer = res['result']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(user_question)
        print(answer)

        return answer




# def main():
#     # Parse the command line arguments
#     args = parse_arguments()
    

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
#                                                  'using the power of LLMs.')
#     parser.add_argument("--hide-source", "-S", action='store_true',
#                         help='Use this flag to disable printing of source documents used for answers.')

#     parser.add_argument("--mute-stream", "-M",
#                         action='store_true',
#                         help='Use this flag to disable the streaming StdOut callback for LLMs.')

#     return parser.parse_args()


# if __name__ == "__main__":
#     main()