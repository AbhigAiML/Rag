import streamlit as st
import requests

st.set_page_config(page_title="Advanced RAG Portal", page_icon="🎛️", layout="wide")

# Configurable endpoints
BASE_URL = "http://127.0.0.1:8000" # Change to your Render URL in production
SEARCH_URL = f"{BASE_URL}/api/v1/search"
UPLOAD_URL = f"{BASE_URL}/api/v1/upload"

# --- SIDEBAR INTERFACE CONTROLS ---
st.sidebar.title("🎛️ Pipeline Configurations")
st.sidebar.markdown("---")

# 1. Temperature Controller
st.sidebar.subheader("Hyperparameters")
user_temp = st.sidebar.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1, 
                              help="Lower values make the response deterministic; higher values make it creative.")

# 2. Dynamic System Prompt Editor
st.sidebar.subheader("System Prompt Editor")
default_prompt = (
    "You are a document-grounded assistant. Answer ONLY using information present in the provided context.\n"
    "If the answer is not explicitly stated or cannot be reasonably inferred from the context, say: "
    "The provided documents do not contain sufficient information to answer this question. Do not use external knowledge."
)
custom_prompt = st.sidebar.text_area("System Instructions", value=default_prompt, height=220)

# 3. Dynamic File Uploader Block
st.sidebar.markdown("---")
st.sidebar.subheader("Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload new document (PDF, TXT, CSV, etc.)", type=["pdf", "txt", "csv", "xlsx", "docx", "json"])

if uploaded_file is not None:
    if st.sidebar.button("🚀 Index Document"):
        with st.sidebar.spinner("Uploading and processing vectors..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                upload_res = requests.post(UPLOAD_URL, files=files, timeout=120)
                if upload_res.status_code == 200:
                    st.sidebar.success(f"✅ {uploaded_file.name} successfully indexed!")
                else:
                    st.sidebar.error(f"Failed to process file: {upload_res.text}")
            except Exception as e:
                st.sidebar.error(f"Inbound network block error: {str(e)}")

# --- MAIN CONVERSATION SCREEN ---
st.title("🤖 Enterprise RAG Chat Interface")
st.caption("Adjust generation constraints or upload context files directly using the left sidebar panel.")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask something about your data corpus..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Package payload matching AdvancedQueryPayload schema perfectly
        payload = {
            "query": user_query,
            "top_k": 3,
            "temperature": user_temp,
            "system_prompt": custom_prompt
        }
        
        try:
            with st.spinner("Executing retrieval loop..."):
                response = requests.post(SEARCH_URL, json=payload, timeout=60)
            
            if response.status_code == 200:
                api_result = response.json().get("result", "")
                response_placeholder.markdown(api_result)
                st.session_state.messages.append({"role": "assistant", "content": api_result})
            else:
                response_placeholder.error(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            response_placeholder.error(f"📡 Connection failure: {str(e)}")