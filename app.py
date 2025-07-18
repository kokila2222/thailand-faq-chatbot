import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

openai_api_key = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="Thailand Relocation Assistant", layout="wide")

# 🎨 CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&family=Noto+Sans:wght@400;500;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Noto Sans', sans-serif;
}

section[data-testid="stSidebar"] {
    background-color: #f8fafc;
    padding: 2rem 1.5rem;
    border-right: 1px solid #e2e8f0;
}

.sidebar-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0d141c;
    margin-bottom: 0.5rem;
}

.sidebar-item {
    font-size: 0.9rem;
    color: #0d141c;
    padding: 0.5rem 0;
    border-radius: 8px;
    transition: background 0.2s;
}

.sidebar-item:hover {
    background-color: #e7edf4;
    padding-left: 0.75rem;
    font-weight: 500;
}

.hero {
    background-image: linear-gradient(0deg, rgba(0, 0, 0, 0.4), rgba(0,0,0,0)), url("https://lh3.googleusercontent.com/aida-public/AB6AXuC9JD6-GBfr8mcNvVUHP8sqSY2yEMKg4IWdSaJhGfhlE4nzGqxm5M5aiAuVcqkAgcL-JV7KOZkCiixQzJMftVqZSUVoJYDHHbI2kd6IwfHpc-UAGWKwGt-XXSCDn9AhJXUpYDXTDMR34-l1W5NNzH7eQ_cfW-I1ZsSyoEnrx1gSSOQnCGPvtgPrvC8ti6yhIavCGv2CcoUZ39orUlAy6wwD95Te3C3D-6qTsXpj0w704M0iRqgIDzKKAF0qldHrWJFzX9fEGPk7qGs");
    background-size: cover;
    background-position: center;
    padding: 2rem;
    border-radius: 1rem;
    color: white;
    margin-top: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# 🧭 Sidebar
with st.sidebar:
    st.markdown("<div class='sidebar-header'>Thailand Relocation Assistant</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-header'>How I Built This App</div>", unsafe_allow_html=True)
    steps = [
        "🔹 Loaded .txt files using TextLoader",
        "🔹 Split them with RecursiveCharacterTextSplitter",
        "🔹 Embedded text with OpenAIEmbeddings",
        "🔹 Stored in FAISS (no server needed)",
        "🔹 Used RetrievalQA with ChatOpenAI",
        "🔹 Built with Streamlit and styled like Stitch",
        "🔹 Stored key in st.secrets"
    ]
    for step in steps:
        st.markdown(f"<div class='sidebar-item'>{step}</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-header'>Tools Used</div>", unsafe_allow_html=True)
    tools = [
        "Streamlit", "LangChain", "FAISS", "OpenAI API", "TextLoader",
        "RecursiveCharacterTextSplitter", "RetrievalQA", "Local Files", "st.secrets"
    ]
    for tool in tools:
        st.markdown(f"<div class='sidebar-item'>{tool}</div>", unsafe_allow_html=True)

# 🎉 Hero banner
st.markdown("""
<div class="hero">
    <h2 style="font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem;">
        Welcome to the Thailand Relocation Assistant!
    </h2>
    <p style="font-size: 1rem; font-weight: 400;">
        I'm here to help you navigate your move to Thailand. Ask me anything about visas, housing, healthcare, and more.
    </p>
</div>
""", unsafe_allow_html=True)

# 📂 Load documents
docs_directory = "thai_docs"

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    all_docs = []
    for filename in os.listdir(docs_directory):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(docs_directory, filename))
            all_docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(all_docs)
    return FAISS.from_documents(documents, embeddings)

def get_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# 🧠 App state
if "chain" not in st.session_state:
    st.session_state.chain = get_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 💬 Chat form
sources = []
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything about Thailand:", key="user_input_form")
    submit = st.form_submit_button("Ask")

if submit and user_input:
    with st.spinner("Thinking..."):
        result = st.session_state.chain(user_input)
        answer = result["result"]
        sources = result["source_documents"]

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", answer))
        st.session_state.chat_history.append(("Bot", "💬 Ready for your next question?"))

# 💬 Show chat
for role, message in st.session_state.chat_history:
    st.markdown(f"**{role}:** {message}", unsafe_allow_html=True)

# 📄 Show sources
if sources:
    with st.expander("📄 Sources"):
        for doc in sources:
            st.markdown(f"**{os.path.basename(doc.metadata['source'])}**")
            st.write(doc.page_content)

# 👣 Footer
st.markdown('<div style="text-align:center; color:#888; margin-top:3rem;">Made by Kokila Mallikarjuna</div>', unsafe_allow_html=True)
