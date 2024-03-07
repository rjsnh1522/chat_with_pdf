import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import css, bot_template, user_template
from ingest import FileIngester


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    print(text)
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)

    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunk):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(
        texts=text_chunk,
        embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversaion({'question': user_question})
    st.write(response)


def save_uploadedfile(uploadedfiles):
    try:
        source_directory = os.getenv('SOURCE_DIRECTORY', 'source_documents')
        for uploadedfile in uploadedfiles:
            with open(os.path.join(source_directory, uploadedfile.name),"wb") as f:
                f.write(uploadedfile.getbuffer())
                st.success("File saved".format(uploadedfile.name))
        return True
    except Exception as e:
        return False



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with multiple PDFS :books:")


    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "hello BOT"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        docs = st.file_uploader(
            "Upload your pdfs here and click upload",
            accept_multiple_files=True,
            type=["doc", "pdf", "epub", "txt", "odt"]
        )
        if st.button("process"):
            with st.spinner("processing"):

                if docs is not None:
                    is_saved = save_uploadedfile(docs)
                    if is_saved:
                        st.success("Please wait.... we are proccessing the files")
                        file_ingester = FileIngester(streamlit=st)
                        file_ingester.ingest()


                # raw_text = get_pdf_text(pdf_docs)

                # text_chunks = get_text_chunks(raw_text)

                # vectorstore = get_vectorstore(text_chunks)

                # st.session_state.conversation = get_conversation_chain(vectorstore)




if __name__ == "__main__":
    main()