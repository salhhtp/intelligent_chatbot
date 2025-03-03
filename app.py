from flask import Flask, request, jsonify, render_template, session
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re

app = Flask(__name__)
app.secret_key = "akjvgbakjbjk"

# Load model and tokenizer for DialoGPT-medium
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text-generation pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

def is_math_query(query):
    """
    Check if the query is a simple math expression.
    This regex allows only digits, spaces, and basic arithmetic operators.
    """
    cleaned = query.replace("?", "").strip()
    return bool(re.fullmatch(r'[\d\s\+\-\*\/\(\)]+', cleaned))

def evaluate_math(query):
    """
    Evaluate the math expression from the query.
    """
    try:
        # Evaluate the expression safely after our regex check
        result = eval(query)
        return str(result)
    except Exception as e:
        return "Sorry, I couldn't compute that."

def get_chatbot_response(user_input):
    # Retrieve the conversation history from the session (or initialize it)
    conversation = session.get("conversation", [])

    # Append the new user input to the conversation history
    conversation.append(f"User: {user_input}")

    # Construct the prompt from the conversation history
    prompt = "\n".join(conversation) + "\nBot:"

    # Generate a response using the text-generation pipeline with advanced sampling parameters
    response = chatbot(
        prompt,
        max_length=150,  # Adjust based on your needs
        num_return_sequences=1,
        do_sample=True,
        top_p=0.9,
        top_k=50,
    )
    generated_text = response[0]["generated_text"]

    # Extract the bot response from the generated text
    bot_response = generated_text.split("Bot:")[-1].strip()

    # Append the bot's response to the conversation history and save back to the session
    conversation.append(f"Bot: {bot_response}")
    session["conversation"] = conversation
    return bot_response

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    print("User query:", query)  # Debug info

    # Check if the query looks like a math expression
    if is_math_query(query):
        answer = evaluate_math(query)
    else:
        answer = get_chatbot_response(query)

    print("Bot answer:", answer)  # Debug info
    return jsonify({'response': answer})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset():
    # Optional endpoint to reset the conversation history
    session["conversation"] = []
    return jsonify({'response': 'Conversation reset.'})


if __name__ == '__main__':
    app.run(debug=True)
