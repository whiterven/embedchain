import os
import queue
import re
import tempfile
import threading

import streamlit as st

from embedchain import App
from embedchain.config import BaseLlmConfig
from embedchain.helpers.callbacks import (StreamingStdOutCallbackHandlerYield,
                                          generate)
from cachetools import TTLCache

# Initialize cache for storing responses
response_cache = TTLCache(maxsize=100, ttl=300)

def embedchain_bot(db_path, api_key):
    return App.from_config(
        config={
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-3.5-turbo-16k",  # Use a model with a higher token limit if available
                    "temperature": 0.7,  # Increase temperature for more creative and detailed responses
                    "max_tokens": 4096,
                    "top_p": 1,
                    "stream": True,
                    "api_key": api_key,
                },
            },
            "vectordb": {
                "provider": "chroma",
                "config": {"collection_name": "chat-pdf", "dir": db_path, "allow_reset": True},
            },
            "embedder": {"provider": "openai", "config": {"api_key": api_key}},
            "chunker": {"chunk_size": 2000, "chunk_overlap": 0, "length_function": "len"},
        }
    )


def get_db_path():
    return tempfile.mkdtemp()


def get_ec_app(api_key):
    if "app" in st.session_state:
        app = st.session_state.app
    else:
        db_path = get_db_path()
        app = embedchain_bot(db_path, api_key)
        st.session_state.app = app
    return app


def add_pdf_to_knowledge_base(pdf_file, app):
    file_name = pdf_file.name
    temp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, prefix=file_name, suffix=".pdf") as f:
            f.write(pdf_file.getvalue())
            temp_file_name = f.name
        if temp_file_name:
            app.add(temp_file_name, data_type="pdf_file")
            os.remove(temp_file_name)
            return f"Added {file_name} to knowledge base!"
    except Exception as e:
        return f"Error adding {file_name} to knowledge base: {e}"

def add_website_to_knowledge_base(web_page, app):
    try:
        app.add(web_page, data_type="web_page")
        return f"Successfully saved {web_page} (DataType.WEB_PAGE)."
    except Exception as e:
        return f"Failed to save {web_page} (DataType.WEB_PAGE): {e}"




def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_user_input(prompt, app):
    # Check if the user has added any PDFs or links
    if not st.session_state.get("add_pdf_files") and not st.session_state.get("add_website_files"):
        # Engage in a more human-like conversation
        with st.chat_message("assistant"):
            st.markdown(
                """
                It seems like you haven't added any PDFs or links yet. No worries! I'm here to help you with any questions you might have. 
                Is there a particular topic or area you're interested in? Feel free to share, and I'll do my best to assist you!
                """
            )
    else:
        with st.chat_message("user"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown(prompt)

        if prompt in response_cache:
            response = response_cache[prompt]
            with st.chat_message("assistant"):
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            return

        with st.chat_message("assistant"):
            msg_placeholder = st.empty()
            msg_placeholder.markdown("Thinking...")

            full_response = ""
            q = queue.Queue()

            def app_response(result):
                llm_config = app.llm.config.as_dict()
                llm_config["callbacks"] = [StreamingStdOutCallbackHandlerYield(q=q)]
                config = BaseLlmConfig(**llm_config)
                answer, citations = app.chat(prompt, config=config, citations=True)
                result["answer"] = answer
                result["citations"] = citations

            results = {}
            thread = threading.Thread(target=app_response, args=(results,))
            thread.start()

            for answer_chunk in generate(q):
                full_response += answer_chunk
                msg_placeholder.markdown(full_response)

            thread.join()

            answer, citations = results.get("answer", ""), results.get("citations", [])
            if citations:
                full_response += "\n\n**Sources**:\n"
                sources = []
                for citation in citations:
                    source = citation[1]["url"]
                    pattern = re.compile(r"([^/]+)\.[^\.]+\.pdf$")
                    match = pattern.search(source)
                    if match:
                        source = match.group(1) + ".pdf"
                    sources.append(source)
                sources = list(set(sources))
                for source in sources:
                    full_response += f"- {source}\n"

            msg_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Cache the response
            response_cache[prompt] = full_response

            # Collect user feedback
            if st.button("Is this answer helpful?"):
                feedback = st.radio("Feedback", ["Yes", "No"])
                if feedback == "No":
                    st.text_input("What was wrong with the answer?", key="feedback_input")
                    st.button("Submit Feedback")





st.title("Chat with Raven 🤖")
styled_caption = '<p style="font-size: 8px; color: #aaa;">🚀 An <a href="https://github.com/embedchain/embedchain">Embedchain</a> app powered by OpenAI!</p>'
st.markdown(styled_caption, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """
                Hey there 👋! I'm Raven🤖., my human friends call me Seyyidi.
                I'm great at answering questions about PDF docs 📄 and links 🔗. 
                Drop your PDFs or links here and let's chat! 
                Even if you don't have a link, we can still chat about anything 😊. 
                Go ahead, ask me anything! 
            """,
        }
    ]

with st.sidebar:
    openai_access_token = st.text_input("OpenAI API Key", key="api_key", type="password")
    "WE DO NOT STORE YOUR OPENAI KEY."
    "Just paste your OpenAI API key here and we'll use it to power the chatbot. [Get your OpenAI API key](https://platform.openai.com/api-keys)"

    if st.session_state.api_key:
        app = get_ec_app(st.session_state.api_key)

    pdf_files = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type="pdf")
    add_pdf_files = st.session_state.get("add_pdf_files", [])

    for pdf_file in pdf_files:
        file_name = pdf_file.name
        if file_name in add_pdf_files:
            continue
        if not st.session_state.api_key:
            st.error("Please enter your OpenAI API Key")
            st.stop()
        message = add_pdf_to_knowledge_base(pdf_file, app)
        st.markdown(message)
        if "Error" not in message:
            add_pdf_files.append(file_name)
            st.session_state.messages.append({"role": "assistant", "content": message})

    st.session_state["add_pdf_files"] = add_pdf_files

    # New section for website links
    st.write("## Add Website Links")
    website_url = st.text_input("Enter website link", key="website_url")
    if st.button("Add Website"):
        if not st.session_state.api_key:
            st.error("Please enter your OpenAI API Key")
            st.stop()
        message = add_website_to_knowledge_base(website_url, app)
        st.markdown(message)
        if "Error" not in message:
            if "add_website_files" not in st.session_state:
                st.session_state["add_website_files"] = []
            st.session_state["add_website_files"].append(website_url)
            st.session_state.messages.append({"role": "assistant", "content": message})

display_messages()

st.write("## Raven🤖")


if prompt := st.chat_input("Ask me anything!"):
        if not st.session_state.api_key:
            st.error("Please enter your OpenAI API Key", icon="🤖")
            st.stop()

        app = get_ec_app(st.session_state.api_key)
        process_user_input(prompt, app)
    