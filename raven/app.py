"""
RAVEN v1 — Gradio Web Interface.

Clean ChatGPT-style dark mode UI with mode selection.
Compatible with Gradio 6.x (messages format).
"""

import os
import sys
import random

import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import gradio as gr
import config
from core.model import RavenModel
from core.tokenizer import RavenTokenizer
from personality.system_prompt import SYSTEM_PROMPT
from main import validate_input, FALLBACK_RESPONSES, MODES

# ---------------------------------------------------------------------------
# Load model and tokenizer
# ---------------------------------------------------------------------------

def find_file(filename):
    """Search local dir then Drive for a file."""
    local = os.path.join(SCRIPT_DIR, filename)
    if os.path.isfile(local):
        return local
    drive = os.path.join("/content/drive/MyDrive/raven_checkpoints", filename)
    if os.path.isfile(drive):
        return drive
    return None


def load_model():
    """Load tokenizer and model. Returns (model, tokenizer, param_count)."""
    tok_path = find_file("raven_tokenizer.json")
    if not tok_path:
        print("[RAVEN ERROR] Tokenizer not found.")
        sys.exit(1)

    tokenizer = RavenTokenizer.load(tok_path)
    config.vocab_size = tokenizer.vocab_size

    w_path = find_file("raven_weights.pt")
    is_ckpt = False
    if not w_path:
        w_path = find_file("raven_checkpoint_latest.pt")
        is_ckpt = True
    if not w_path:
        print("[RAVEN ERROR] Model weights not found.")
        sys.exit(1)

    try:
        payload = torch.load(w_path, map_location=config.device, weights_only=not is_ckpt)
    except TypeError:
        payload = torch.load(w_path, map_location=config.device)

    if is_ckpt and isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    model = RavenModel().to(config.device)
    model.load_state_dict(state_dict)
    model.eval()

    total = model.get_num_params()
    print(f"Tokenizer: {tok_path}")
    print(f"Model: {w_path}")
    print(f"Parameters: {total:,}")
    print(f"Device: {config.device}")
    print("[RAVEN READY]")
    return model, tokenizer, total


model, tokenizer, total_params = load_model()


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

def generate(message, history, mode):
    """Generate a RAVEN response. Returns updated history in messages format."""
    if not message or not message.strip():
        return "", history

    message = message.strip()

    # Get mode settings
    mode_key = mode.lower() if mode else "default"
    settings = MODES.get(mode_key, MODES["default"])

    # Validate / preprocess input
    processed = validate_input(message)

    # Build history context from last 2 turns
    history_text = ""
    if history:
        # Extract last 2 user/assistant pairs from messages format
        pairs = []
        i = 0
        while i < len(history):
            if history[i]["role"] == "user":
                user_msg = history[i]["content"]
                asst_msg = ""
                if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                    asst_msg = history[i + 1]["content"]
                    i += 2
                else:
                    i += 1
                pairs.append((user_msg, asst_msg))
            else:
                i += 1

        for u, a in pairs[-2:]:
            history_text += f"User: {u}\nRAVEN:\n{a}\n\n"

    # Build prompt
    if history_text:
        prompt = f"{SYSTEM_PROMPT}\n{history_text}User: {processed}\nRAVEN:"
    else:
        prompt = f"{SYSTEM_PROMPT}\nUser: {processed}\nRAVEN:"

    # Encode and crop from LEFT to fit block_size
    tokens = tokenizer.encode(prompt, add_bos=True)
    if len(tokens) > config.block_size:
        tokens = tokens[-config.block_size:]

    idx = torch.tensor([tokens], dtype=torch.long, device=config.device)
    prompt_len = idx.shape[1]

    # Generate
    try:
        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=120,
                min_new_tokens=10,
                temperature=settings["temperature"],
                top_k=settings["top_k"],
                top_p=settings["top_p"],
                repetition_penalty=1.35,
                repetition_window=60,
                stop_strings=["User:", "RAVEN:", "[SYSTEM]"],
                tokenizer=tokenizer,
                stop_tokens=[tokenizer.eos_id],
            )

        new_tokens = out[0][prompt_len:].tolist()
        response = tokenizer.decode(new_tokens)

        # Clean — truncate at any leaked marker
        for marker in ("User:", "RAVEN:", "[SYSTEM]", "[END SYSTEM]"):
            pos = response.find(marker)
            if pos != -1:
                response = response[:pos]

        lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
        lines = [l for l in lines if len(l) <= 80][:4]

        if not lines:
            response = random.choice(FALLBACK_RESPONSES)
        else:
            response = "\n".join(lines)

    except Exception:
        response = random.choice(FALLBACK_RESPONSES)

    # Append in Gradio 6 messages format
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return "", history


# ---------------------------------------------------------------------------
# CSS — ChatGPT-style dark mode
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

* { box-sizing: border-box; }

body {
    background: #212121 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.gradio-container {
    max-width: 780px !important;
    margin: 0 auto !important;
    background: transparent !important;
    padding: 0 16px !important;
}

/* Header */
#raven-header {
    text-align: center;
    padding: 40px 0 8px 0;
}
#raven-header h1 {
    font-size: 28px;
    font-weight: 600;
    color: #ececec;
    letter-spacing: 4px;
    margin: 0;
}
#raven-header p {
    color: #8e8e8e;
    font-size: 13px;
    margin: 6px 0 0 0;
    letter-spacing: 0.5px;
}

/* Chatbot area */
.chatbot-container {
    border: none !important;
    background: #212121 !important;
    border-radius: 12px !important;
}

/* Message bubbles */
.message {
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important;
    line-height: 1.6 !important;
}

.user .message-content {
    background: #2f2f2f !important;
    color: #ececec !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 18px !important;
}

.bot .message-content {
    background: transparent !important;
    color: #d1d1d1 !important;
    padding: 12px 4px !important;
}

/* Input area */
.input-row {
    margin-top: 8px !important;
}

#msg-input textarea {
    background: #2f2f2f !important;
    border: 1px solid #424242 !important;
    border-radius: 16px !important;
    color: #ececec !important;
    padding: 14px 20px !important;
    font-size: 14.5px !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s ease;
}

#msg-input textarea:focus {
    border-color: #666 !important;
    outline: none !important;
    box-shadow: none !important;
}

#msg-input textarea::placeholder {
    color: #777 !important;
}

/* Send button */
#send-btn {
    background: #ececec !important;
    color: #212121 !important;
    border: none !important;
    border-radius: 50% !important;
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    max-width: 44px !important;
    padding: 0 !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: background 0.15s ease;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

#send-btn:hover {
    background: #ffffff !important;
}

/* Controls row */
.controls-row {
    margin-top: 8px !important;
    gap: 12px !important;
}

/* Mode dropdown */
#mode-select {
    --input-background-fill: #2f2f2f !important;
    --border-color-primary: #424242 !important;
    --body-text-color: #ececec !important;
}

#mode-select .wrap {
    background: #2f2f2f !important;
    border: 1px solid #424242 !important;
    border-radius: 10px !important;
    color: #ececec !important;
}

#mode-select label span {
    color: #8e8e8e !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

/* Reset button */
#reset-btn {
    background: #2f2f2f !important;
    color: #8e8e8e !important;
    border: 1px solid #424242 !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease;
    height: 42px !important;
}

#reset-btn:hover {
    color: #ececec !important;
    border-color: #666 !important;
}

/* Footer info */
#footer-info {
    text-align: center;
    padding: 16px 0 24px 0;
}
#footer-info p {
    color: #555;
    font-size: 11px;
    margin: 0;
    letter-spacing: 0.3px;
}

/* Remove Gradio branding borders/shadows */
.block { border: none !important; box-shadow: none !important; }
.wrap { background: transparent !important; }
footer { display: none !important; }
"""


# ---------------------------------------------------------------------------
# Gradio 6.x UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="RAVEN") as demo:

    gr.HTML("""
        <div id="raven-header">
            <h1>RAVEN</h1>
            <p>Not an assistant. A presence.</p>
        </div>
    """)

    chatbot = gr.Chatbot(
        height=500,
        label="",
        show_label=False,
        elem_classes=["chatbot-container"],
    )

    with gr.Row(elem_classes=["input-row"]):
        msg = gr.Textbox(
            placeholder="Message RAVEN...",
            label="",
            show_label=False,
            scale=10,
            container=False,
            elem_id="msg-input",
        )
        send = gr.Button("↑", elem_id="send-btn", scale=1, min_width=44)

    with gr.Row(elem_classes=["controls-row"]):
        mode = gr.Dropdown(
            choices=["DEFAULT", "SAVAGE", "PHILOSOPHER", "CHAOS", "NOX", "DURDEN"],
            value="DEFAULT",
            label="Mode",
            elem_id="mode-select",
            scale=2,
        )
        clear = gr.Button("Reset", elem_id="reset-btn", scale=1)

    gr.HTML(f"""
        <div id="footer-info">
            <p>{total_params:,} parameters · {config.device.upper()}</p>
        </div>
    """)

    # Event handlers
    send.click(
        generate,
        inputs=[msg, chatbot, mode],
        outputs=[msg, chatbot],
    )
    msg.submit(
        generate,
        inputs=[msg, chatbot, mode],
        outputs=[msg, chatbot],
    )
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])


# Launch with theme and css (Gradio 6.x)
demo.launch(css=CSS)
