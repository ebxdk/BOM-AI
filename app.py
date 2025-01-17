from flask import Flask, request, jsonify
import os
from transformers import BartForConditionalGeneration, BartTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import glob

app = Flask(__name__)

# Environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the paraphrase model
paraphrase_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
paraphrase_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Load documents for RAG
folder_path = '/home/ebad_khan5487/BOM-AI/Datasets'  # Updated based on your initial file
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
llm = ChatOpenAI(model_name='gpt-4', temperature=0.6)
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
    input_variables=['username', 'energy_score', 'purpose_score', 'connection_score', 'user_state', 'context', 'chat_history', 'question']
)

qa_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    username = data.get('username', 'User')
    energy_score = data.get('energy_score', 0)
    purpose_score = data.get('purpose_score', 0)
    connection_score = data.get('connection_score', 0)
    user_state = data.get('user_state', 'Unknown')

    # Retrieve relevant context
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(user_message)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create input for LangChain
    chain_input = {
        'username': username,
        'energy_score': energy_score,
        'purpose_score': purpose_score,
        'connection_score': connection_score,
        'user_state': user_state,
        'context': context,
        'chat_history': memory.load_memory_variables({})['chat_history'],
        'question': user_message
    }

    # Get response from QA chain
    response = qa_chain(chain_input)
    chat_response = response['text']

    return jsonify({"response": chat_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
