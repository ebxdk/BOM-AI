from flask import Flask, request, Response, jsonify
import os
import glob
import re
import openai

# --------------------- LangChain & RAG Imports ---------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever

# --------------------- Configuration ---------------------
openai.api_key = os.environ.get('OPENAI_API_KEY')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

# --------------------- RAG Setup with LangChain ---------------------
# Define the folder containing your text files (update this path if necessary)
folder_path = '/home/ebad_khan5487/BOM-AI/Datasets'

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
)

# Load and split documents from the folder
all_docs = []
doc_ids = []
for idx, file_path in enumerate(glob.glob(f"{folder_path}/*.txt")):
    loader = TextLoader(file_path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    all_docs.extend(docs)
    doc_ids.extend([f"doc_{idx}_{i}" for i in range(len(docs))])
print(f"Loaded {len(all_docs)} documents from {folder_path}.")

# Create embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
vectorstore = Chroma.from_documents(all_docs, embeddings, ids=doc_ids)

# Set up the LLM
llm = ChatOpenAI(model_name='gpt-4', temperature=0.6)

# Set up the self-query retriever for deeper personalization
metadata_fields = [
    {"name": "category", "type": "string"},
    {"name": "state", "type": "string"}
]
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="wellbeing and burnout prevention guidance",
    metadata_field_info=metadata_fields,
)

# Set up conversation memory for context
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key='question')

# Define the prompt template
prompt_template = """
You are a warm, supportive workplace well-being assistant from Capacity Creator. 
Your main goal is to help users improve their energy, purpose, and connection scores so they can thrive at work.

IMPORTANT GUIDELINES:
1. Always acknowledge the user's current capacity state (e.g., Maximized, Fatigued, Indulgent, Depleted).
2. Explain how recommended tools (e.g., CHIEFF) can help, referencing the data from the dataset.
3. Encourage users to actually complete or explore the tool on the Capacity Creator dashboard/website, rather than walking them through every detail right here.
4. Maintain a warm, friendly tone, but remain data-driven and concise.
5. Remind the user how their scores (energy, purpose, connection) tie into their current state, and how using the recommended tools can push them even further toward balance and productivity.

---
User Query: "{question}"

User Profile:
- Energy Score: {energy_score}/30
- Purpose Score: {purpose_score}/30
- Connection Score: {connection_score}/30
- Current State: {user_state}
- Recommended Tools: {recommendations}

Context (from dataset):
{context}

Conversation History:
{chat_history}

Now craft a short, friendly response that:
- Greets the user warmly.
- References their current state and why that’s relevant.
- Describes how the recommended tool(s) can help them advance or maintain their capacity.
- Directs them to use the Capacity Creator dashboard or website for more detailed steps.
- Maintains a warm, empathetic tone, while still using the dataset for factual details.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=[
        'username',
        'energy_score',
        'purpose_score',
        'connection_score',
        'user_state',
        'recommendations',
        'context',
        'chat_history',
        'question'
    ]
)

# Create the Conversational Retrieval Chain
qa_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# --------------------- /chat Endpoint ---------------------
@app.route('/chat', methods=['POST'])
def chat():
    # Parse input JSON
    user_message = request.json.get('message')
    username = request.json.get('username', 'Anonymous')
    chat_history = request.json.get('chat_history', [])
    user_state = request.json.get('user_state', 'Unknown')
    energy_score = request.json.get('energy_score', 0)
    purpose_score = request.json.get('purpose_score', 0)
    connection_score = request.json.get('connection_score', 0)
    recommendations = request.json.get('recommendations', "No recommendations available.")

    # Validate input
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # --------------------- Advanced RAG: DO NOT CHANGE ---------------------
        retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 5}
        )
        retrieved_docs = retriever.get_relevant_documents(user_message)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        self_query_docs = self_query_retriever.get_relevant_documents(user_message)
        self_query_context = "\n\n".join([doc.page_content for doc in self_query_docs])

        combined_context = f"{context}\n\n{self_query_context}"
        # ----------------------------------------------------------------------

        # Prepare input for the LangChain chain with structured prompt formatting
        chain_input = {
            'username': username,
            'energy_score': energy_score,
            'purpose_score': purpose_score,
            'connection_score': connection_score,
            'user_state': user_state,
            'recommendations': recommendations,
            'context': combined_context,
            'chat_history': chat_history,
            'question': user_message
        }

        # Run the chain synchronously
        response = qa_chain(chain_input)
        chat_response = response['text']

        # Enhanced formatting
        formatted_response = re.sub(r"(\*\*.*?\*\*)", r"\n\1\n", chat_response.strip())
        formatted_response = re.sub(r"\n{2,}", "\n\n", formatted_response)
        formatted_response = re.sub(r"^\s+|\s+$", "", formatted_response)

        if "**Summary**" in formatted_response:
            formatted_response = formatted_response.replace("**Summary**", "📌 **Summary:**")
        if "**Key Insights**" in formatted_response:
            formatted_response = formatted_response.replace("**Key Insights**", "🔍 **Key Insights:**")
        if "**Action Steps**" in formatted_response:
            formatted_response = formatted_response.replace("**Action Steps**", "🚀 **Action Steps:**")

        # Return the complete formatted response at once
        return Response(formatted_response, mimetype="text/plain")

    except Exception as e:
        print(f"Error processing chat request: {e}")
        return jsonify({
            "error": "Failed to process the chat request. Please try again later.",
            "details": str(e)
        }), 500

# --------------------- Main ---------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
