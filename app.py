from flask import Flask, request, jsonify
import os
from transformers import BartForConditionalGeneration, BartTokenizer
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import glob

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Load the paraphrase model
paraphrase_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
paraphrase_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Load documents for RAG
folder_path = '/home/ebad_khan5487/BOM-AI/Datasets'  # Update this path if needed
all_docs = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

for file_path in glob.glob(f"{folder_path}/*.txt"):
    loader = TextLoader(file_path)
    documents = loader.load()
    all_docs.extend(text_splitter.split_documents(documents))

# Create embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
vectorstore = Chroma.from_documents(all_docs, embeddings)

# Set up LangChain components
llm = ChatOpenAI(
    model_name='gpt-4',  # or "gpt-3.5-turbo" if GPT-4 is unavailable
    temperature=0.6,
    openai_api_key=openai_api_key
)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

prompt_template = """
You are a helpful AI assistant. Use the following user context and retrieved data to answer their question clearly and concisely.

User Information:
- Username: {username}
- Energy Score: {energy_score}
- Purpose Score: {purpose_score}
- Connection Score: {connection_score}
- User State: {user_state}

Relevant Context:
{context}

Chat History:
{chat_history}

User Question: {question}
Assistant Response:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=[
        'username', 'energy_score', 'purpose_score',
        'connection_score', 'user_state', 'context',
        'chat_history', 'question'
    ]
)

qa_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json

    # Validate input
    required_fields = [
        'message', 'username', 'energy_score',
        'purpose_score', 'connection_score',
        'user_state', 'context', 'chat_history'
    ]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

    # Extract user inputs
    user_message = data['message']
    username = data['username']
    energy_score = data['energy_score']
    purpose_score = data['purpose_score']
    connection_score = data['connection_score']
    user_state = data['user_state']
    context = data['context']  # This might be an empty string if none is provided
    chat_history = data['chat_history']  # Typically a list of messages

    # Retrieve relevant context from vectorstore
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    try:
        docs = retriever.get_relevant_documents(user_message)
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve context: {str(e)}"}), 500

    # Combine retrieved documents into a single string
    retrieved_context = "\n\n".join([doc.page_content for doc in docs])

    # Prepare the input for the QA chain
    chain_input = {
        'username': username,
        'energy_score': energy_score,
        'purpose_score': purpose_score,
        'connection_score': connection_score,
        'user_state': user_state,
        'context': retrieved_context,
        'chat_history': chat_history,
        'question': user_message
    }

    try:
        # Pass the input to the QA chain
        response = qa_chain.invoke(chain_input)  # This is correct in newer LangChain versions
        chat_response = response['text']
    except Exception as e:
        # Log and return the error if the chain fails
        return jsonify({"error": f"QA Chain Error: {str(e)}"}), 500

    # Return the chatbot's response
    return jsonify({"response": chat_response})

if __name__ == '__main__':
    # Make sure to adjust host and port if necessary.
    # Also consider setting debug=False for production.
    app.run(host='0.0.0.0', port=5001, debug=False)
