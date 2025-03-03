import os
import re
import uuid
import sqlite3
from flask import Flask, request, jsonify, render_template, g
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask_cors import CORS
import sympy  # For advanced math parsing

# -------------- CONFIG & INIT -------------- #

app = Flask(__name__)
app.secret_key = "your_secret_key_here"
CORS(app)
DATABASE = os.path.join(app.root_path, 'database.db')

# Default models you might allow. Feel free to add more (e.g., GPT-Neo).
AVAILABLE_MODELS = {
    "DialoGPT-medium": "microsoft/DialoGPT-medium",
    "DialoGPT-large": "microsoft/DialoGPT-large"
    # "GPT-Neo": "EleutherAI/gpt-neo-1.3B", etc.
}

# We'll cache loaded pipelines so we don't reload them repeatedly.
model_pipelines = {}

def get_db():
    """Get a SQLite connection, store in flask.g for reuse."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close DB connection on teardown."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """Create tables if not exist."""
    conn = get_db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chats (
      id TEXT PRIMARY KEY,
      language TEXT NOT NULL,
      model_name TEXT NOT NULL,
      title TEXT NOT NULL
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      chat_id TEXT NOT NULL,
      sender TEXT NOT NULL,
      text TEXT NOT NULL,
      FOREIGN KEY (chat_id) REFERENCES chats(id)
    )
    """)
    conn.commit()

def load_pipeline(model_name):
    """Load a text-generation pipeline for the given model name."""
    if model_name in model_pipelines:
        return model_pipelines[model_name]
    # Otherwise, load it
    hf_model_path = AVAILABLE_MODELS.get(model_name, "microsoft/DialoGPT-medium")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    model_pipelines[model_name] = pipe
    return pipe

# -------------- MATH HELPERS -------------- #

def is_math_query(query: str) -> bool:
    """Check if query is purely a math expression (digits, + - * / ^ ( ) .)."""
    # Extended to allow ^ (exponent) and decimal points
    cleaned = query.replace("?", "").strip()
    return bool(re.fullmatch(r'[0-9\.\s\+\-\*\/\^\(\)]+', cleaned))

def evaluate_math(query: str) -> str:
    """
    Use sympy to parse and evaluate the expression safely.
    """
    try:
        expression = sympy.sympify(query)
        result = expression.evalf()  # Evaluate to float if possible
        return str(result)
    except Exception:
        return "Sorry, I couldn't compute that."

# -------------- CHAT & MESSAGES -------------- #

def create_chat(language="English", model_name="DialoGPT-medium", title="New Chat"):
    """Create a new chat record in the DB."""
    chat_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO chats (id, language, model_name, title) VALUES (?, ?, ?, ?)",
        (chat_id, language, model_name, title)
    )
    conn.commit()
    return chat_id

def get_chat(chat_id):
    """Retrieve a chat row by ID."""
    conn = get_db()
    row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    return row

def update_chat_language(chat_id, language):
    conn = get_db()
    conn.execute("UPDATE chats SET language = ? WHERE id = ?", (language, chat_id))
    conn.commit()

def update_chat_model(chat_id, model_name):
    conn = get_db()
    conn.execute("UPDATE chats SET model_name = ? WHERE id = ?", (model_name, chat_id))
    conn.commit()

def get_chat_messages(chat_id):
    """Return all messages (sender + text) for a given chat, in chronological order."""
    conn = get_db()
    rows = conn.execute(
        "SELECT sender, text FROM messages WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,)
    ).fetchall()
    return [dict(r) for r in rows]

def add_message(chat_id, sender, text):
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (chat_id, sender, text) VALUES (?, ?, ?)",
        (chat_id, sender, text)
    )
    conn.commit()

# -------------- BOT LOGIC -------------- #

def build_prompt(messages, language):
    """
    Construct a prompt from the conversation history plus a system instruction
    to respond in the chosen language. We also label each message with 'User:' or 'Bot:'.
    """
    system_instruction = f"The user wants all responses in {language}.\n"
    conversation_part = ""
    for m in messages:
        if m["sender"] == "user":
            conversation_part += f"User: {m['text']}\n"
        else:
            conversation_part += f"Bot: {m['text']}\n"
    prompt = system_instruction + conversation_part + "Bot:"
    return prompt

def generate_response(chat_id, user_query):
    """
    1) If it's math, handle directly.
    2) Otherwise, build prompt, call pipeline, parse result.
    """
    chat = get_chat(chat_id)
    if not chat:
        return "Invalid chat ID."

    # Check for math query
    if is_math_query(user_query):
        answer = evaluate_math(user_query)
        return answer

    # Retrieve conversation history
    messages = get_chat_messages(chat_id)
    # Add the new user message
    add_message(chat_id, "user", user_query)
    messages.append({"sender": "user", "text": user_query})

    # Build the prompt
    language = chat["language"]
    model_name = chat["model_name"]
    pipe = load_pipeline(model_name)
    prompt = build_prompt(messages, language)

    # Generate text
    response = pipe(
        prompt,
        max_length=200,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )
    generated_text = response[0]["generated_text"]
    # Extract the bot's last line
    if "Bot:" in generated_text:
        bot_answer = generated_text.split("Bot:")[-1].strip()
    else:
        bot_answer = generated_text

    # Store the bot message
    add_message(chat_id, "bot", bot_answer)
    return bot_answer

# -------------- ROUTES -------------- #

@app.route('/')
def index():
    # Serve the main chat UI
    return render_template('index.html')

@app.route('/api/new_chat', methods=['POST'])
def api_new_chat():
    """Create a new chat and return its ID + initial data."""
    data = request.get_json() or {}
    language = data.get("language", "English")
    model_name = data.get("model_name", "DialoGPT-medium")
    title = data.get("title", "New Chat")
    chat_id = create_chat(language, model_name, title)
    return jsonify({"chat_id": chat_id, "language": language, "model_name": model_name})

@app.route('/api/get_chats', methods=['GET'])
def api_get_chats():
    """Return a list of existing chats in the DB (for the sidebar)."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM chats ORDER BY rowid DESC").fetchall()
    chats = []
    for r in rows:
        chats.append({
            "id": r["id"],
            "language": r["language"],
            "model_name": r["model_name"],
            "title": r["title"]
        })
    return jsonify({"chats": chats})

@app.route('/api/get_chat_messages', methods=['GET'])
def api_get_chat_messages():
    """Return the messages of a specific chat."""
    chat_id = request.args.get("chat_id")
    if not chat_id:
        return jsonify({"error": "No chat_id provided"}), 400
    chat = get_chat(chat_id)
    if not chat:
        return jsonify({"error": "Invalid chat_id"}), 400
    messages = get_chat_messages(chat_id)
    return jsonify({"messages": messages})

@app.route('/api/set_language', methods=['POST'])
def api_set_language():
    """Update the language for a given chat."""
    data = request.get_json() or {}
    chat_id = data.get("chat_id")
    language = data.get("language", "English")
    if not chat_id:
        return jsonify({"error": "No chat_id provided"}), 400
    if not get_chat(chat_id):
        return jsonify({"error": "Invalid chat_id"}), 400
    update_chat_language(chat_id, language)
    return jsonify({"status": "ok", "language": language})

@app.route('/api/set_model', methods=['POST'])
def api_set_model():
    """Switch to a different model for the given chat."""
    data = request.get_json() or {}
    chat_id = data.get("chat_id")
    model_name = data.get("model_name", "DialoGPT-medium")
    if not chat_id:
        return jsonify({"error": "No chat_id provided"}), 400
    if model_name not in AVAILABLE_MODELS:
        return jsonify({"error": "Model not supported"}), 400
    if not get_chat(chat_id):
        return jsonify({"error": "Invalid chat_id"}), 400
    update_chat_model(chat_id, model_name)
    return jsonify({"status": "ok", "model_name": model_name})

@app.route('/api/send_message', methods=['POST'])
def api_send_message():
    """Send a user query to the bot and get the response."""
    data = request.get_json() or {}
    chat_id = data.get("chat_id")
    query = data.get("query")
    if not chat_id:
        return jsonify({"error": "No chat_id provided"}), 400
    if not query:
        return jsonify({"error": "No query provided"}), 400
    bot_answer = generate_response(chat_id, query)
    return jsonify({"response": bot_answer})

@app.route('/api/reset_chat', methods=['POST'])
def api_reset_chat():
    """Clear all messages in a chat but keep the chat record."""
    data = request.get_json() or {}
    chat_id = data.get("chat_id")
    if not chat_id:
        return jsonify({"error": "No chat_id provided"}), 400
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.commit()
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(debug=True)
