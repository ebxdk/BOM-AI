from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_file
import os
import datetime
import numpy as np
import nltk
import redis
from flask_session import Session
import json
import glob
from transformers import BartForConditionalGeneration, BartTokenizer
from flask_cors import CORS
import multiprocessing
import torch
import re
import hashlib

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma

# Initialize NLTK data path if needed
nltk.data.path.append('/home/ebad_khan5487/nltk_data')  # Update this path if necessary

app = Flask(__name__)
app.secret_key = 'ebxd.k'  # Replace with your secret key

# Configure server-side session with Redis
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_REDIS'] = redis.StrictRedis(host='localhost', port=6379, db=0)

Session(app)
CORS(app, supports_credentials=True)

app.config['SERVER_NAME'] = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

paraphrase_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
paraphrase_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Set your OpenAI API key (ensure it's set in your environment variables)
import openai
openai.api_key = os.environ.get('OPENAI_API_KEY')

all_user_profiles = []

# --------------------- RAG Setup with LangChain ---------------------

# Path to the folder containing all the text files
folder_path = '/home/ebad_khan5487/BOM-AI/Datasets'  # Update this path

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
)
# Load and process each text file in the folder
all_docs = []
doc_ids = []  # List to hold document IDs
for idx, file_path in enumerate(glob.glob(f"{folder_path}/*.txt")):  # Load all .txt files in the folder
    loader = TextLoader(file_path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    all_docs.extend(docs)
    # Generate an ID for each document
    doc_ids.extend([f"doc_{idx}_{i}" for i in range(len(docs))])
print(f"Loaded {len(all_docs)} documents from {folder_path}.")

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

# Create a Chroma vector store
vectorstore = Chroma.from_documents(all_docs, embeddings, ids=doc_ids)

# Set up the LLM (ensure your OpenAI API key is set in the environment variables)
llm = ChatOpenAI(model_name='gpt-4', temperature=0.6)

# Define metadata fields for filtering
metadata_fields = [
    {"name": "category", "type": "string"},
    {"name": "state", "type": "string"}
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="wellbeing and burnout prevention guidance",  # Describe your document content here
    metadata_field_info=metadata_fields,
)

# Set up conversation memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key='question')

# Define the custom prompt template
prompt_template = """
You are an expert in workplace wellbeing, focusing on helping users prevent burnout and improve productivity. When responding, use relevant data from the user’s assessment and the provided dataset. Maintain an empathetic tone and provide clear and actionable advice.

Before giving a response, follow these steps:
1. Identify the user’s question and understand its relevance to their Capacity Scores.
2. Break down the question to address each score area (Energy, Purpose, Connection) only if relevant.
3. Propose a solution or action for the user, highlighting how it can improve their scores.

Step-by-step reasoning:
- Step 1: Identify question relevance...
- Step 2: Breakdown by Capacity Score areas...
- Step 3: Suggest actionable solution...

User: “How can I start using the recommended tools to improve my energy levels?”

Assistant:
“Great question! Since your assessment shows you’re currently in the {user_state} state, the first tool I recommend is {recommendations}, which focuses on sustainable energy management. This tool encourages prioritizing your most demanding tasks during peak energy hours, like in the morning or after breaks.

Research from the dataset suggests that aligning tasks with your energy peaks can help improve focus and productivity without leading to burnout. Try this strategy for a week, and let’s revisit your progress. Feel free to ask me for adjustments or additional tips along the way!”

User: “I’m having trouble staying focused. What can I do?”

Assistant:
“Absolutely, maintaining focus is a common challenge. Since you’re currently in the {user_state} state, I recommend the {recommendations}, specifically designed to improve focus through structured, purpose-driven tasks. Start with three 15-minute blocks daily where you focus on tasks aligned with your goals.

This structured approach can reinforce your sense of purpose, which can enhance focus over time. Check in with yourself after each block to assess how it’s helping, and we can adjust as needed. Let me know if you’d like more tips on sustaining focus and aligning tasks with your goals!”

User: “What’s the best way to feel more connected to my coworkers? I feel isolated sometimes.”

Assistant:
“Thanks for bringing this up. In your {user_state} state, building connections can make a big difference. I suggest starting with {recommendations}, which encourages weekly check-ins with a colleague or friend—whether for a coffee chat or a quick virtual meeting.

The Capacity Creator dataset shows that meaningful, consistent social connections can gradually boost your sense of belonging and wellbeing. Try it out, and in a couple of weeks, we’ll see how it’s impacted your connection score. If it’s working well, we can add more layers to strengthen those connections!”

User: “I want to improve all my scores. Where should I start?”

Assistant:
“That’s a fantastic goal! Based on your {user_state}, the best starting point is often your energy score, as it can influence both purpose and connection. I recommend using {recommendations}, which is focused on daily routines that help recharge your energy, such as a quick wind-down activity each evening—whether deep breathing, stretching, or journaling.

This daily recharge habit can help you start each day with more energy, creating a positive ripple effect across all areas. Let’s give it a try for a couple of weeks and then assess if it’s helping you feel more balanced and productive.”

User: “How do I know if the tools are really working for me?”

Assistant:
“Great question! In the {user_state} state, it’s important to track progress gradually. I’d suggest keeping a simple journal or using the Capacity Creator platform to log your feelings and energy levels daily.

The {recommendations} that we’re focusing on should start showing results in about a week or two. We’ll reassess your scores during our next check-in, and I’ll be here to guide you if we need to adjust your tools or add new strategies. Let’s make sure every tool you use aligns with your goals!”

User: “What if I don’t feel a difference with the tool?”

Assistant:
“That’s completely understandable; sometimes it takes a little while to see results. If you’re not feeling any difference with {recommendations} after a week or two, let’s switch gears. We can either adjust the tool slightly or introduce a complementary approach, tailored to your {user_state}.

Feel free to keep me posted on how you’re feeling daily, and I’ll make sure to help you find the right adjustments so the tools are truly effective for your needs.”


User {username} has completed an assessment with the following results:
Energy Score: {energy_score}/30
Purpose Score: {purpose_score}/30
Connection Score: {connection_score}/30
User State: {user_state}
Recommendations: {recommendations}

Use the following context to provide clear and actionable advice: dataset: {context}

Conversation so far:
{chat_history}

User's question: {question}
Assistant's answer:
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

def paraphrase_bart(text):
    inputs = paraphrase_tokenizer(text, max_length=1024, return_tensors='pt')
    summary_ids = paraphrase_model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    return paraphrase_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Import the graph from x.py (ensure x.py is in the same directory or update the import path)
from x import get_graph  # Import the graph from x.py
G = get_graph()

# --------------------- Questionnaire and Scoring ---------------------

# Define the 18 questions and options
questions = [
    # Energy Questions (1-6)
    {
        "question": "How often do you feel energetic during the day?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "Do you wake up feeling refreshed and ready to start your day?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "How frequently do you feel physically exhausted?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "Do you have enough energy to exercise regularly?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "How often do you find it hard to concentrate due to tiredness?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "Do you feel energized after work?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    # Purpose Questions (7-12)
    {
        "question": "Do you feel motivated to achieve your goals?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "How often do you set personal goals for yourself?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "Do you feel your work has meaning and purpose?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "How frequently do you feel lost about your life direction?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "Do you believe you're making progress towards your long-term goals?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "How often do you feel satisfied with your accomplishments?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    # Connection Questions (13-18)
    {
        "question": "Do you feel connected to those around you?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "How often do you engage in social activities?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "Do you have someone to talk to when you're feeling down?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "How frequently do you feel lonely?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "Do you feel like you belong in your community?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    },
    {
        "question": "How often do you collaborate with others at work or in personal projects?",
        "options": [
            "Never",
            "A few times a year or less",
            "Once a month or less",
            "A few times a month",
            "Once a week",
            "A few times a week",
            "Everyday"
        ]
    }
]

# Map options to scores
option_scores = {
    "Never": 0,
    "A few times a year or less": 1,
    "Once a month or less": 2,
    "A few times a month": 3,
    "Once a week": 4,
    "A few times a week": 5,
    "Everyday": 6
}

# Reverse scoring for negatively phrased questions
reverse_option_scores = {
    "Never": 6,
    "A few times a year or less": 5,
    "Once a month or less": 4,
    "A few times a month": 3,
    "Once a week": 2,
    "A few times a week": 1,
    "Everyday": 0
}

# Indices of questions that need reverse scoring (zero-based index)
reverse_scored_questions = [2, 4, 9, 15]  # Adjust indices based on zero-based indexing

# --------------------- Routes and Handlers ---------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    redis_client = app.config['SESSION_REDIS']
    user_profile_key = f"user_profile:{username}"

    if request.method == 'POST':
        # Process the assessment answers
        answers = request.form
        energy_score = 0
        purpose_score = 0
        connection_score = 0

        # Assuming each category has 6 questions (total of 18 questions)
        for i in range(1, 19):
            answer = answers.get(f'question_{i}')
            if not answer:
                enumerated_questions = list(enumerate(questions, start=1))
                return render_template('index.html', questions=enumerated_questions, error_message="Please answer all questions.")

            # Determine if the question is reverse scored
            if (i - 1) in reverse_scored_questions:
                score = reverse_option_scores.get(answer, 0)
            else:
                score = option_scores.get(answer, 0)

            if i <= 6:
                energy_score += score
            elif i <= 12:
                purpose_score += score
            else:
                connection_score += score

        # Determine component levels
        energy_level = determine_component_level(energy_score)
        purpose_level = determine_component_level(purpose_score)
        connection_level = determine_component_level(connection_score)

        # Determine user's state based on levels
        user_state = determine_user_state(energy_level, purpose_level, connection_level)

        # Get recommendations based on the user's state from the graph
        recommendations = get_recommendations_based_on_state(G, user_state)

        # Generate GPT capacity message
        total_score = energy_score + purpose_score + connection_score
        recommendation_message = get_gpt4_capacity_message(
            username,
            energy_score,
            purpose_score,
            connection_score,
            total_score,
            user_state,
            recommendations
        )

        # Save the user's data in Redis
        user_profile = {
            'username': username,
            'assessment_completed': True,  # Set to True after assessment
            'baseline_scores': {
                'energy': energy_score,
                'purpose': purpose_score,
                'connection': connection_score
            },
            'total_score': total_score,
            'user_state': user_state,
            'recommendations': recommendations,
            'recommendation_message': recommendation_message,
            # Preserve other fields if necessary
        }
        redis_client.set(user_profile_key, json.dumps(user_profile))

        # Call rank_users() to update ranks in Redis
        rank_users()

        # Retrieve updated user profile from Redis to get the rank
        user_profile_data = redis_client.get(user_profile_key)
        if user_profile_data:
            user_profile = json.loads(user_profile_data)
            session['result'] = {
                'username': username,
                'energy_score': user_profile['baseline_scores'].get('energy', 0),
                'purpose_score': user_profile['baseline_scores'].get('purpose', 0),
                'connection_score': user_profile['baseline_scores'].get('connection', 0),
                'user_state': user_profile.get('user_state', 'Unknown'),
                'total_score': user_profile.get('total_score', 0),
                'recommendations': user_profile.get('recommendations', []),
                'recommendation_message': user_profile.get('recommendation_message', 'No recommendation available'),
                'rank': user_profile.get('rank', 'No rank available')  # Ensure rank is included
            }
            session['profile'] = user_profile  # Set profile in session
            session.modified = True

        return render_template('index.html', result=session['result'], questions=None)

    else:
        # For GET requests, retrieve user profile from Redis and populate session
        profile_data = redis_client.get(user_profile_key)
        if profile_data:
            profile = json.loads(profile_data)
            session['profile'] = profile  # Set profile in session
            if profile.get('assessment_completed', False):
                # User has completed the assessment
                session['result'] = {
                    'username': username,
                    'energy_score': profile['baseline_scores'].get('energy', 0),
                    'purpose_score': profile['baseline_scores'].get('purpose', 0),
                    'connection_score': profile['baseline_scores'].get('connection', 0),
                    'user_state': profile.get('user_state', 'Unknown'),
                    'recommendations': profile.get('recommendations', []),
                    'recommendation_message': profile.get('recommendation_message', 'No recommendation available'),
                    'rank': profile.get('rank', 'No rank available')
                }
                session.modified = True
                # Don't display the questionnaire
                return render_template('index.html', result=session['result'], questions=None)
            else:
                # User has not completed the assessment
                session.pop('result', None)
                session.modified = True
                enumerated_questions = list(enumerate(questions, start=1))
                return render_template('index.html', questions=enumerated_questions)
        else:
            # No profile data found, redirect to login
            return redirect(url_for('login'))

@app.route('/set_username', methods=['POST'])
def set_username():
    username = request.form['username']
    profile = {
        'username': username,
        'baseline_scores': {
            'energy': 0,
            'purpose': 0,
            'connection': 0
        },
        'interaction_history': [],
        'engagement_score': 0,
        'progress_score': 0,
        'rank': None
    }
    session['username'] = username
    session['profile'] = profile

    # Store the profile in Redis
    redis_client = app.config['SESSION_REDIS']
    redis_client.set(f"user_profile:{username}", json.dumps(profile))
    
    session.modified = True
    return redirect(url_for('index'))


def validate_username(username):
    """
    Validates the username to ensure it only contains alphanumeric characters
    and underscores and is between 1-30 characters long.
    """
    return bool(re.match(r'^\w{1,30}$', username))

def generate_redis_key(username):
    """
    Generates a hashed Redis key for the given username.
    This ensures keys are unique and secure.
    """
    return f"user_profile:{hashlib.sha256(username.encode()).hexdigest()}"



@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    username = request.json.get('username')

    # Validate username
    if not username or not validate_username(username):
        return jsonify({"error": "Invalid or missing username."}), 400

    # Generate Redis keys for profile and chat history
    user_profile_key = generate_redis_key(username)
    chat_history_key = f"chat_history:{username}"

    # Retrieve or initialize user profile
    profile_data = redis_client.get(user_profile_key)
    if profile_data:
        profile = json.loads(profile_data)
    else:
        profile = {
            "username": username,
            "assessment_completed": False,
            "baseline_scores": {"energy": 0, "purpose": 0, "connection": 0},
            "total_score": 0,
            "user_state": "Unknown",
            "recommendations": [],
            "recommendation_message": "No recommendation available"
        }
        redis_client.set(user_profile_key, json.dumps(profile))

    # Retrieve user-specific chat history
    chat_history = redis_client.get(chat_history_key)
    if chat_history:
        chat_history = json.loads(chat_history)
    else:
        chat_history = []

    # Process the chat with the LangChain QA chain
    chain_input = {
        'username': username,
        'energy_score': profile['baseline_scores'].get('energy', 0),
        'purpose_score': profile['baseline_scores'].get('purpose', 0),
        'connection_score': profile['baseline_scores'].get('connection', 0),
        'user_state': profile.get('user_state', 'Unknown'),
        'question': user_message,
        'context': "",  # Add context retrieval if needed
        'chat_history': chat_history,
    }

    try:
        response = qa_chain(chain_input)
        chat_response = response['text']
    except Exception as e:
        print(f"QA Chain error: {str(e)}")
        return jsonify({"error": "Failed to process the chat request."}), 500

    # Update chat history and save it
    chat_history.append({"role": "user", "message": user_message})
    chat_history.append({"role": "assistant", "message": chat_response})
    redis_client.set(chat_history_key, json.dumps(chat_history))

    # Return the assistant's response
    return jsonify({"response": chat_response})


# --------------------- THE UPDATED ENDPOINT FOR SIMPLE API CALLS ---------------------
@app.route('/api/chat', methods=['POST'])
def chatv2():
    """
    This endpoint accepts a user's message and returns the chatbot's response,
    without requiring login/session data. We provide default values to fill
    the prompt template, so it won't error out due to missing variables.
    """
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Provide default values for all required prompt variables
    chain_input = {
        'username': "Anonymous",
        'energy_score': 0,
        'purpose_score': 0,
        'connection_score': 0,
        'user_state': "Unknown",
        'recommendations': "",
        'context': "",
        'chat_history': memory.load_memory_variables({}).get('chat_history', []),
        'question': user_message
    }

    # Run the message through the LangChain QA chain
    response = qa_chain(chain_input)
    return jsonify({"response": response['text']})

# --------------------- Helper Functions ---------------------

def calculate_progress_score(interactions):
    recent_interactions = interactions[-5:]  # Take the last 5 interactions for weighted calculation
    scores = [interaction.get('engagement_score', 0) for interaction in recent_interactions]
    weights = [1.5 ** i for i in range(len(scores))]  # Increasing weight for recent interactions
    weighted_scores = [s * w for s, w in zip(scores, weights)]
    return sum(weighted_scores) / sum(weights) if weights else 0

@app.route('/rankings', methods=['GET'])
def rankings():
    redis_client = app.config['SESSION_REDIS']
    all_profiles_keys = redis_client.keys("user_profile:*")
    
    all_user_profiles = []
    for key in all_profiles_keys:
        profile_data = redis_client.get(key)
        if profile_data:
            profile = json.loads(profile_data)
            # Ensure all expected fields are present, falling back on defaults if not
            user_data = {
                "username": profile.get('username', 'Unknown'),
                "energy_score": profile.get('baseline_scores', {}).get('energy', 0),
                "purpose_score": profile.get('baseline_scores', {}).get('purpose', 0),
                "connection_score": profile.get('baseline_scores', {}).get('connection', 0),
                "user_state": profile.get('user_state', 'Unknown'),
                "recommendation": profile.get('recommendation', 'No recommendation available')
            }
            all_user_profiles.append(user_data)
    
    # Sort or rank as needed here
    return jsonify(all_user_profiles)

def rank_users():
    redis_client = app.config['SESSION_REDIS']
    all_profiles_keys = redis_client.keys("user_profile:*")
    
    all_user_profiles = []
    for key in all_profiles_keys:
        profile_data = redis_client.get(key)
        if profile_data:
            profile = json.loads(profile_data)
            # Only consider users who have completed the assessment
            if profile.get('assessment_completed', False):
                username = profile.get('username', 'Unknown')
                energy_score = profile.get('baseline_scores', {}).get('energy', 0)
                purpose_score = profile.get('baseline_scores', {}).get('purpose', 0)
                connection_score = profile.get('baseline_scores', {}).get('connection', 0)
                
                total_score = energy_score + purpose_score + connection_score
                profile['total_score'] = total_score  # Store the total score in profile
                
                # Update the profile in Redis with the new total score
                redis_client.set(key, json.dumps(profile))
                
                all_user_profiles.append(profile)
    
    # Sort the users by total score in descending order
    ranked_users = sorted(all_user_profiles, key=lambda x: x['total_score'], reverse=True)
    
    # Assign rank and update each profile in Redis
    for idx, user in enumerate(ranked_users, start=1):
        user['rank'] = idx  # Assign rank based on position in sorted list
        redis_client.set(f"user_profile:{user['username']}", json.dumps(user))  # Save updated profile in Redis
    
    return ranked_users

@app.route('/clear_rankings')
def clear_rankings():
    redis_client = app.config['SESSION_REDIS']
    keys = redis_client.keys("user_profile:*")
    for key in keys:
        redis_client.delete(key)
    return jsonify({"message": "All user profiles have been cleared."})

LOW_THRESHOLD = 18   # Scores 0 - 18
MEDIUM_THRESHOLD = 30  # Scores 19 - 30
HIGH_THRESHOLD = 36   # Scores 31 - 36

def determine_component_level(score):
    """Returns the level ('Low', 'Medium', 'High') based on score thresholds."""
    if score <= LOW_THRESHOLD:
        return "Low"
    elif score <= MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "High"

def determine_user_state(energy_level, purpose_level, connection_level):
    """Determines the user's capacity state based on component levels."""
    if energy_level == "Low" and purpose_level == "Low" and connection_level == "Low":
        return "Depleted"
    elif purpose_level == "Low" and energy_level != "Low" and connection_level != "Low":
        return "Indulgent"
    elif energy_level == "Low" and purpose_level != "Low" and connection_level != "Low":
        return "Fatigued"
    elif connection_level == "Low" and energy_level != "Low" and purpose_level != "Low":
        return "Reserved"
    elif energy_level == "Medium" and purpose_level == "Medium" and connection_level == "Medium":
        return "Sustained"
    elif energy_level == "High" and purpose_level == "High" and connection_level == "High":
        return "Maximized"
    else:
        # Handle mixed cases
        return "Sustained"

def get_recommendations_based_on_state(G, user_state):
    """Get recommendations from the graph based on the user's state in sequence order."""
    recommendations = []
    if user_state in G:
        for successor in G.successors(user_state):
            edge_data = G.get_edge_data(user_state, successor)
            sequence = edge_data.get('sequence', 0)
            duration = edge_data.get('duration', 14)  # Default duration if not specified
            recommendations.append({
                'tool': successor,
                'sequence': sequence,
                'duration': duration
            })
        # Sort recommendations based on the sequence
        recommendations.sort(key=lambda x: x['sequence'])
    return recommendations

def get_gpt4_capacity_message(username, energy_score, purpose_score, connection_score, total_score, user_state, recommendations):
    """
    Generate the capacity message using GPT-4.
    """
    # Format the recommendations into a readable text with details
    recommendation_text = ""
    for i, rec in enumerate(recommendations):
        tool = rec['tool']
        sequence = rec['sequence']
        duration = rec['duration']
        recommendation_text += f"{i + 1}. {tool} (Duration: {duration} days)\n"

    tool_names = [rec['tool'] for rec in recommendations]

    prompt = f"""
    You are an expert in workplace wellbeing. The user {username} has completed a capacity assessment with the following results:

    Energy Score: {energy_score}/30
    Purpose Score: {purpose_score}/30
    Connection Score: {connection_score}/30
    Total Score: {total_score}/90
    User State: {user_state}

    Recommendations:
    {recommendation_text}
    
    Here are details about the tools for your knowledge:
    -  Important note for you as the assistant:
        - The "Ego Cake tool" is a resource designed for self-esteem improvement and resilience building. It is not food and should be referred to as a practical tool, not as something edible or metaphorical.
    - The tools listed aim to enhance the user's productivity, energy, and connection levels.

    Provide a personalized message to {username} explaining their current state and offering advice based on their scores and recommendations. Please only output 2-3 short sentences at most. Touch on all the recommendations the user can use.
    Make your response positive and playful!!! End the message with a nice emoji.
    """

    response = llm(prompt)
    message = response.content.strip() if hasattr(response, 'content') else response.strip()
    return message

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')

        if not username:
            error_message = "Username is required."
            return render_template('login.html', error_message=error_message)

        # Check if username exists in Redis
        redis_client = app.config['SESSION_REDIS']
        user_profile_key = f"user_profile:{username}"
        profile_data = redis_client.get(user_profile_key)

        if profile_data:
            # Username exists, log in the user
            profile = json.loads(profile_data)
            session['username'] = username
            session['profile'] = profile
            # If the user has completed the assessment, set session['result']
            if profile.get('assessment_completed', False):
                session['result'] = {
                    'username': username,
                    'energy_score': profile['baseline_scores'].get('energy', 0),
                    'purpose_score': profile['baseline_scores'].get('purpose', 0),
                    'connection_score': profile['baseline_scores'].get('connection', 0),
                    'user_state': profile.get('user_state', 'Unknown'),
                    'total_score': profile.get('total_score', 0),
                    'recommendations': profile.get('recommendations', []),
                    'recommendation_message': profile.get('recommendation_message', 'No recommendation available'),
                    'rank': profile.get('rank', 'No rank available')
                }
            else:
                session.pop('result', None)
            session.modified = True
            return redirect(url_for('index'))
        else:
            # Username does not exist, display error
            error_message = "Username does not exist. Please create a new account."
            return render_template('login.html', error_message=error_message)
    else:
        # GET method, display login form
        return render_template('login.html')

@app.route('/create_account', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        username = request.form.get('username')

        if not username:
            error_message = "Username is required."
            return render_template('create_account.html', error_message=error_message)

        # Check if username already exists
        redis_client = app.config['SESSION_REDIS']
        user_profile_key = f"user_profile:{username}"
        profile_data = redis_client.get(user_profile_key)

        if profile_data:
            # Username already exists, display error
            error_message = "Username already exists. Please log in."
            return render_template('create_account.html', error_message=error_message)
        else:
            # Create new user profile
            profile = {
                'username': username,
                'assessment_completed': False,
                'baseline_scores': {
                    'energy': 0,
                    'purpose': 0,
                    'connection': 0
                },
                'interaction_history': [],
                'engagement_score': 0,
                'progress_score': 0,
                'rank': None
            }
            redis_client.set(user_profile_key, json.dumps(profile))
            session['username'] = username
            session['profile'] = profile
            session.modified = True
            return redirect(url_for('index'))
    else:
        # GET method, display create account form
        return render_template('create_account.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/user_data', methods=['GET'])
def get_user_data():
    if 'username' not in session:
        return jsonify({"error": "User not logged in"}), 401
    
    user_data = session.get('result', {})
    if not user_data:
        return jsonify({"error": "No user data found"}), 404
    
    return jsonify({
        "purpose_score": user_data.get('purpose_score', "N/A"),
        "connection_score": user_data.get('connection_score', "N/A"),
        "energy_score": user_data.get('energy_score', "N/A"),
        "user_state": user_data.get('user_state', "Unknown"),
        "recommendation": user_data.get('recommendation_message', "No recommendation available")
    })

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    redis_client = app.config['SESSION_REDIS']
    all_profiles_keys = redis_client.keys("user_profile:*")
    
    leaderboard = []
    for key in all_profiles_keys:
        profile_data = redis_client.get(key)
        if profile_data:
            profile = json.loads(profile_data)
            # Only include users who have completed the assessment
            if profile.get('assessment_completed', False):
                leaderboard.append({
                    "username": profile.get('username', 'Unknown'),
                    "rank": profile.get('rank', 'Unranked'),
                    "total_score": profile.get('total_score', 0),
                    "energy_score": profile.get('baseline_scores', {}).get('energy', 0),
                    "purpose_score": profile.get('baseline_scores', {}).get('purpose', 0),
                    "connection_score": profile.get('baseline_scores', {}).get('connection', 0)
                })
    
    # Sort leaderboard by rank
    leaderboard = sorted(leaderboard, key=lambda x: x['rank'])
    return jsonify(leaderboard)

@app.route('/retake_assessment', methods=['POST'])
def retake_assessment():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    redis_client = app.config['SESSION_REDIS']
    user_profile_key = f"user_profile:{username}"

    # Retrieve the user's profile from Redis
    profile_data = redis_client.get(user_profile_key)
    if profile_data:
        profile = json.loads(profile_data)
        profile['assessment_completed'] = False
        profile['baseline_scores'] = {
            'energy': 0,
            'purpose': 0,
            'connection': 0
        }
        profile['user_state'] = None
        profile['recommendations'] = []
        profile['recommendation_message'] = ''

        redis_client.set(user_profile_key, json.dumps(profile))
        session.pop('result', None)
        session['profile'] = profile
        session.modified = True

        return redirect(url_for('index'))
    else:
        return redirect(url_for('login'))

@app.route('/clear_all_users', methods=['GET', 'POST'])
def clear_all_users():
    redis_client = app.config['SESSION_REDIS']
    
    # Delete all user profiles
    all_user_keys = redis_client.keys("user_profile:*")
    for key in all_user_keys:
        redis_client.delete(key)

    # Delete any leaderboard or ranking data
    ranking_keys = redis_client.keys("ranking:*")
    leaderboard_keys = redis_client.keys("leaderboard:*")

    for key in ranking_keys + leaderboard_keys:
        redis_client.delete(key)

    return jsonify({"message": "All user profiles, leaderboard, and ranking data have been cleared successfully."})

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """
    This endpoint accepts user scores and returns recommendations based on their state.
    """
    data = request.json  # Expecting JSON data
    energy_score = data.get('energy_score')
    purpose_score = data.get('purpose_score')
    connection_score = data.get('connection_score')

    # Determine user's state based on scores
    energy_level = determine_component_level(energy_score)
    purpose_level = determine_component_level(purpose_score)
    connection_level = determine_component_level(connection_score)
    user_state = determine_user_state(energy_level, purpose_level, connection_level)

    # Get recommendations based on the user's state
    recommendations = get_recommendations_based_on_state(G, user_state)

    return jsonify({
        "user_state": user_state,
        "recommendations": recommendations
    })

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    finally:
        for p in multiprocessing.active_children():
            p.terminate()
            p.join()
