# evaluate_rag.py
import requests
import json
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall

# Define the base URL
BASE_URL = 'http://127.0.0.1:5000'  # Update if running on a different host/port

def main():
    # Load generated questions
    with open("generated_questions.json", "r") as file:
        questions = json.load(file)

    # Prepare data for evaluation
    data = []

    # Start a session
    session = requests.Session()

    # Step 1: Send each question to the /chat endpoint
    for i, question in enumerate(questions):
        chat_url = f'{BASE_URL}/chat'
        chat_data = {'message': question}
        response = session.post(chat_url, json=chat_data)

        if response.status_code != 200:
            print(f'Error sending chat message {i+1}')
            continue

        response_data = response.json()
        assistant_response = response_data.get('response', '')
        retrieved_documents = response_data.get('retrieved_documents', [])

        # Append the data for evaluation
        data.append({
            'question': question,
            'contexts': retrieved_documents,
            'answer': assistant_response
        })

    # Step 2: Evaluate using RAGAS
    metrics = [faithfulness, context_precision, context_recall]
    evaluation_results = evaluate(data, metrics=metrics)

    # Print the evaluation results
    print("Evaluation Results:")
    for metric, score in evaluation_results.items():
        print(f"{metric.capitalize()}: {score:.2f}")

if __name__ == '__main__':
    main()