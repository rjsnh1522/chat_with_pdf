import os
import streamlit as st
from dotenv import load_dotenv
from htmlTemplate import css, bot_template, user_template
from ingest import FileIngester
from private_chat import ChatResponse


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



def get_response(user_question):
    chat = ChatResponse()
    response = chat.get_answer(user_question=user_question)
    return response


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFS :books:")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        response = get_response(user_question=user_question)
        with st.chat_message("assistant"):
            st.markdown(response)
            # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

  

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
                    else:
                        st.error("Error saving your file")

                # st.session_state.conversation = get_conversation_chain(vectorstore)




if __name__ == "__main__":
    main()