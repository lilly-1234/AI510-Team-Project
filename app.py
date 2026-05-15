from flask import Flask, request, jsonify, render_template_string # type: ignore
import pickle
import joblib # type: ignore
import joblib # type: ignore
import os

app = Flask(__name__)

# =============================================================
# LOAD MODEL ARTIFACTS
# =============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_artifacts():
    vectorizer    = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    model         = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    return vectorizer, model, label_encoder

vectorizer, model, label_encoder = load_artifacts()
print("[INFO] Model artifacts loaded successfully.")

# =============================================================
# INTENT → RESPONSE MAPPING
# =============================================================
RESPONSES = {
    "cancel_order":              "I can help you cancel your order. Please share your order number and I'll process the cancellation right away.",
    "change_order":              "Sure! To modify your order, please provide your order number and the changes you'd like to make.",
    "change_shipping_address":   "I can update your shipping address. Please provide your order number and the new address.",
    "check_cancellation_fee":    "Cancellation fees depend on the order status. Orders cancelled before shipping are free; after shipping, a fee may apply.",
    "check_invoice":             "I can pull up your invoice. Please share your order number or the email used during purchase.",
    "check_payment_methods":     "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
    "check_refund_policy":       "Refunds are processed within 5–7 business days. Items must be returned within 30 days of purchase.",
    "complaint":                 "I'm sorry to hear about your experience. Please describe the issue and I'll escalate it to our team immediately.",
    "contact_customer_service":  "You can reach our support team 24/7 via chat, or by email at support@shop.com.",
    "contact_human_agent":       "I'll connect you with a human agent right away. Please hold for a moment.",
    "create_account":            "To create an account, visit our website and click 'Sign Up'. It only takes a minute!",
    "delete_account":            "To delete your account, go to Account Settings → Privacy → Delete Account. This action is permanent.",
    "delivery_options":          "We offer Standard (5–7 days), Express (2–3 days), and Same-Day delivery in select areas.",
    "delivery_period":           "Standard delivery takes 5–7 business days. Express delivery takes 2–3 business days.",
    "edit_account":              "To update your account details, go to Account Settings and make your changes there.",
    "get_invoice":               "I'll send your invoice to the email on file. Would you like me to do that now?",
    "get_refund":                "To request a refund, please share your order number. Refunds are processed within 5–7 business days.",
    "newsletter_subscription":   "To manage your newsletter preferences, go to Account Settings → Notifications.",
    "payment_issue":             "I'm sorry you're having a payment issue. Please check your card details or try a different payment method.",
    "place_order":               "To place an order, browse our catalogue, add items to your cart, and proceed to checkout.",
    "recover_password":          "Click 'Forgot Password' on the login page and we'll send a reset link to your email.",
    "registration_problems":     "Having trouble registering? Make sure your email isn't already in use and your password meets requirements.",
    "review":                    "We'd love your feedback! You can leave a review on the product page after your order is delivered.",
    "set_up_shipping_address":   "Go to Account Settings → Addresses → Add New Address to set up your shipping address.",
    "switch_account":            "To switch accounts, log out from the current account and sign in with the other account credentials.",
    "track_order":               "To track your order, please share your order number and I'll give you the latest update.",
    "track_refund":              "To check your refund status, please share your order number. Refunds typically take 5–7 business days.",
}

# =============================================================
# ROUTES
# =============================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": "Linear SVM + TF-IDF"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: { "message": "Where is my order?" }
    Returns JSON: { "intent": "track_order", "response": "..." }
    """
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Please provide a 'message' field."}), 400

    message = str(data["message"]).strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    # Predict intent
    X        = vectorizer.transform([message])
    encoded  = model.predict(X)[0]
    intent    = label_encoder.inverse_transform([encoded])[0]
    response  = RESPONSES.get(intent, "I'm not sure how to help with that. Let me connect you with a human agent.")

    return jsonify({
        "message":  message,
        "intent":   intent,
        "response": response
    })


@app.route("/", methods=["GET"])
def index():
    """Serve the chat UI."""
    return render_template_string(CHAT_HTML)


# =============================================================
# CHAT UI  (single-file, no templates folder needed)
# =============================================================
CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Customer Support</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0f0f13;
    --surface:   #1a1a22;
    --border:    #2a2a38;
    --accent:    #6c63ff;
    --accent2:   #ff6584;
    --text:      #e8e8f0;
    --muted:     #6b6b80;
    --user-bg:   #6c63ff;
    --bot-bg:    #1e1e2e;
    --radius:    14px;
  }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .wrapper {
    width: 100%;
    max-width: 680px;
    height: 90vh;
    display: flex;
    flex-direction: column;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    overflow: hidden;
  }

  /* Header */
  .header {
    padding: 18px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .avatar {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .header-text h1 { font-size: 15px; font-weight: 600; }
  .header-text p  { font-size: 12px; color: var(--muted); }
  .status-dot {
    width: 8px; height: 8px;
    background: #4ade80;
    border-radius: 50%;
    margin-left: auto;
    box-shadow: 0 0 6px #4ade8088;
  }

  /* Messages */
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }

  .msg { display: flex; gap: 10px; align-items: flex-end; max-width: 85%; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg.bot  { align-self: flex-start; }

  .bubble {
    padding: 12px 16px;
    border-radius: var(--radius);
    font-size: 14px;
    line-height: 1.6;
  }
  .msg.user .bubble {
    background: var(--user-bg);
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg.bot .bubble {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }

  .intent-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    background: #6c63ff22;
    color: var(--accent);
    border: 1px solid #6c63ff44;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 6px;
  }

  .msg-icon {
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    background: var(--bot-bg);
    border: 1px solid var(--border);
  }

  /* Typing indicator */
  .typing .bubble { display: flex; gap: 4px; align-items: center; padding: 14px 16px; }
  .dot {
    width: 6px; height: 6px;
    background: var(--muted);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
  }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30%            { transform: translateY(-6px); }
  }

  /* Input */
  .input-area {
    padding: 16px 20px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 10px;
    align-items: center;
  }
  input[type="text"] {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s;
  }
  input[type="text"]:focus { border-color: var(--accent); }
  input[type="text"]::placeholder { color: var(--muted); }

  button {
    width: 44px; height: 44px;
    background: var(--accent);
    border: none;
    border-radius: 12px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: opacity 0.2s, transform 0.1s;
  }
  button:hover   { opacity: 0.88; }
  button:active  { transform: scale(0.95); }
  button svg     { width: 18px; height: 18px; fill: white; }

  /* Suggestions */
  .suggestions {
    padding: 0 20px 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .chip {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    font-size: 12px;
    font-family: 'DM Sans', sans-serif;
    padding: 6px 12px;
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.2s;
    width: auto; height: auto;
  }
  .chip:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <div class="avatar">🛍️</div>
    <div class="header-text">
      <h1>ShopBot Support</h1>
      <p>AI-powered customer assistant</p>
    </div>
    <div class="status-dot"></div>
  </div>

  <div class="messages" id="messages">
    <div class="msg bot">
      <div class="msg-icon">🤖</div>
      <div class="bubble">
        Hi there! I'm your AI support assistant. How can I help you today?
      </div>
    </div>
  </div>

  <div class="suggestions" id="suggestions">
    <button class="chip" onclick="sendChip(this)">Where is my order?</button>
    <button class="chip" onclick="sendChip(this)">I want a refund</button>
    <button class="chip" onclick="sendChip(this)">Cancel my order</button>
    <button class="chip" onclick="sendChip(this)">Payment issue</button>
  </div>

  <div class="input-area">
    <input type="text" id="userInput" placeholder="Type your message…" onkeydown="if(event.key==='Enter') sendMessage()">
    <button onclick="sendMessage()">
      <svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/></svg>
    </button>
  </div>
</div>

<script>
  function sendChip(btn) {
    document.getElementById('userInput').value = btn.textContent;
    document.getElementById('suggestions').style.display = 'none';
    sendMessage();
  }

  function addMessage(role, text, intent) {
    const box = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    if (role === 'bot') {
      div.innerHTML = `
        <div class="msg-icon">🤖</div>
        <div class="bubble">
          ${intent ? `<div class="intent-badge">${intent}</div><br>` : ''}
          ${text}
        </div>`;
    } else {
      div.innerHTML = `<div class="bubble">${text}</div>`;
    }
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
  }

  function showTyping() {
    const box = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'msg bot typing';
    div.id = 'typing';
    div.innerHTML = `
      <div class="msg-icon">🤖</div>
      <div class="bubble">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </div>`;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
  }

  function removeTyping() {
    const t = document.getElementById('typing');
    if (t) t.remove();
  }

  async function sendMessage() {
    const input = document.getElementById('userInput');
    const msg = input.value.trim();
    if (!msg) return;

    addMessage('user', msg);
    input.value = '';
    showTyping();

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg })
      });
      const data = await res.json();
      removeTyping();
      addMessage('bot', data.response, data.intent);
    } catch (err) {
      removeTyping();
      addMessage('bot', 'Sorry, something went wrong. Please try again.');
    }
  }
</script>
</body>
</html>
"""

# =============================================================
# RUN
# =============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)