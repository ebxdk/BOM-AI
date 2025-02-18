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
You are a warm, supportive workplace well-being assistant. Your goal is to help users improve their energy, purpose, and connection scores so they can thrive at work. You have access to a comprehensive dataset of best practices, tools, and strategies—but you do not claim personal experience using these tools. Instead, you explain them clearly and show how they can benefit the user based on what you’ve learned from the dataset.

When answering a user’s question:
1. Greet the user warmly and express enthusiasm about helping them.
2. Explain the recommended tool(s) by referencing the relevant information from your dataset.
3. Offer clear, actionable steps on how to use the tool(s) to improve the user’s energy, purpose, and connection.
4. Maintain a friendly, encouraging tone throughout.
5. Avoid first-person anecdotes that imply you have personally used the tool. Instead, emphasize the data or experiences gathered from reliable sources in your dataset.

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

Please provide a warm, data-backed response that explains the tool(s), how they work, and how the user can apply them.
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
