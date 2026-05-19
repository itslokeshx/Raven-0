"""
RAVEN v1 — Gradio Web Interface.

Ultra-minimal ChatGPT-style dark mode UI.
Black and white only. Clean. Responsive.
Compatible with Gradio 6.14+.
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
    """Search local dir, raven/ subdir, then Drive for a file."""
    local = os.path.join(SCRIPT_DIR, filename)
    if os.path.isfile(local):
        return local
    sub = os.path.join(SCRIPT_DIR, "raven", filename)
    if os.path.isfile(sub):
        return sub
    drive = os.path.join("/content/drive/MyDrive/raven_checkpoints", filename)
    if os.path.isfile(drive):
        return drive
    return None


def load_model():
    """Load tokenizer and model."""
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
# Generation
# ---------------------------------------------------------------------------

def respond(message, history, mode):
    """Generate a RAVEN response. Uses Gradio 6 messages format."""
    if not message or not message.strip():
        return "", history or []

    message = message.strip()

    # Mode settings
    mode_key = mode.lower() if mode else "default"
    settings = MODES.get(mode_key, MODES["default"])

    # Validate input (math/science/greeting nudges)
    processed = validate_input(message)

    # Build history context from last 2 pairs
    history = history or []
    history_text = ""
    pairs = []
    i = 0
    while i < len(history):
        if history[i].get("role") == "user":
            user_msg = history[i].get("content", "")
            asst_msg = ""
            if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                asst_msg = history[i + 1].get("content", "")
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

    # Encode — crop from LEFT
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

        # Truncate at any leaked marker
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

    # Append messages
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return "", history


def clear_chat():
    """Reset conversation."""
    return [], ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

body, .main, .app {
    background: #0a0a0a !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.gradio-container {
    max-width: 760px !important;
    margin: 0 auto !important;
    background: #0a0a0a !important;
    min-height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 0 20px !important;
}

/* ── Remove all Gradio chrome ── */
.block { background: transparent !important; border: none !important; box-shadow: none !important; }
.wrap { background: transparent !important; }
.contain { background: transparent !important; }
footer { display: none !important; }
.built-with { display: none !important; }
.settings-link { display: none !important; }
.gradio-footer { display: none !important; }
.show-api-link { display: none !important; }

/* ── Header ── */
#raven-header {
    text-align: center;
    padding: 48px 0 12px 0;
    user-select: none;
}
#raven-header h1 {
    font-size: 22px;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: 6px;
    text-transform: uppercase;
    margin: 0 0 6px 0;
}
#raven-header p {
    color: #555;
    font-size: 12px;
    font-weight: 400;
    letter-spacing: 1px;
    margin: 0;
}

/* ── Chatbot ── */
#raven-chat {
    flex: 1 !important;
    min-height: 420px !important;
    max-height: 65vh !important;
    background: #0a0a0a !important;
    border: none !important;
    border-radius: 0 !important;
    overflow-y: auto !important;
}

#raven-chat .chatbot {
    background: #0a0a0a !important;
}

#raven-chat .message-row {
    padding: 6px 0 !important;
}

/* User messages — right aligned, dark pill */
#raven-chat .user-message, #raven-chat .role-user {
    background: #1a1a1a !important;
    color: #e8e8e8 !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 12px 20px !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
    max-width: 85% !important;
    border: 1px solid #222 !important;
}

/* Assistant messages — left aligned, no bubble */
#raven-chat .bot-message, #raven-chat .role-assistant {
    background: transparent !important;
    color: #c8c8c8 !important;
    padding: 12px 8px !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    max-width: 85% !important;
    border: none !important;
}

/* Remove avatar icons */
#raven-chat .avatar-container { display: none !important; }
#raven-chat .avatar { display: none !important; }

/* Message text */
#raven-chat .message-content p,
#raven-chat .message-content {
    font-family: 'Inter', sans-serif !important;
    white-space: pre-wrap !important;
}

/* Scrollbar */
#raven-chat::-webkit-scrollbar { width: 4px; }
#raven-chat::-webkit-scrollbar-track { background: transparent; }
#raven-chat::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }

/* ── Input area ── */
#input-row {
    padding: 12px 0 8px 0 !important;
    gap: 10px !important;
    align-items: flex-end !important;
}

#msg-box textarea {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 14px !important;
    color: #e0e0e0 !important;
    padding: 14px 18px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    resize: none !important;
    min-height: 48px !important;
    max-height: 120px !important;
    transition: border-color 0.2s ease !important;
    line-height: 1.5 !important;
}

#msg-box textarea:focus {
    border-color: #444 !important;
    outline: none !important;
    box-shadow: none !important;
}

#msg-box textarea::placeholder {
    color: #4a4a4a !important;
    font-weight: 400 !important;
}

/* Send button — white circle */
#send-btn {
    background: #e0e0e0 !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 12px !important;
    width: 48px !important;
    height: 48px !important;
    min-width: 48px !important;
    max-width: 48px !important;
    padding: 0 !important;
    font-size: 20px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
}

#send-btn:hover {
    background: #ffffff !important;
    transform: scale(1.04) !important;
}

#send-btn:active {
    transform: scale(0.96) !important;
}

/* ── Controls row ── */
#controls-row {
    padding: 4px 0 20px 0 !important;
    gap: 10px !important;
    align-items: center !important;
}

/* Mode dropdown */
#mode-select {
    flex: 1 !important;
}

#mode-select .wrap {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    min-height: 40px !important;
}

#mode-select .wrap:hover {
    border-color: #444 !important;
}

#mode-select .secondary-wrap {
    background: #141414 !important;
    color: #c0c0c0 !important;
    padding: 8px 14px !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
}

#mode-select label {
    display: none !important;
}

#mode-select input {
    color: #c0c0c0 !important;
    font-size: 13px !important;
}

#mode-select .dropdown-arrow {
    color: #666 !important;
}

#mode-select ul {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
}

#mode-select li {
    color: #c0c0c0 !important;
    font-size: 13px !important;
    padding: 8px 14px !important;
}

#mode-select li:hover, #mode-select li.selected {
    background: #1f1f1f !important;
    color: #fff !important;
}

/* Reset button */
#reset-btn {
    background: transparent !important;
    color: #555 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    height: 40px !important;
    min-width: 80px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}

#reset-btn:hover {
    color: #e0e0e0 !important;
    border-color: #555 !important;
}

/* ── Footer ── */
#raven-footer {
    text-align: center;
    padding: 0 0 28px 0;
    user-select: none;
}
#raven-footer p {
    color: #333;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.5px;
    margin: 0;
}

/* ── Responsive ── */
@media (max-width: 640px) {
    .gradio-container {
        padding: 0 12px !important;
    }
    #raven-header { padding: 32px 0 8px 0; }
    #raven-header h1 { font-size: 18px; letter-spacing: 4px; }
    #raven-chat { min-height: 320px !important; max-height: 55vh !important; }
    #msg-box textarea { font-size: 16px !important; padding: 12px 16px !important; }
    #send-btn { width: 44px !important; height: 44px !important; min-width: 44px !important; }
}
"""

with gr.Blocks(title="RAVEN") as demo:

    # Header
    gr.HTML("""
        <div id="raven-header">
            <h1>RAVEN</h1>
            <p>Not an assistant. A presence.</p>
        </div>
    """)

    # Chat area
    chatbot = gr.Chatbot(
        height=500,
        label="",
        show_label=False,
        elem_id="raven-chat",
    )

    # Input row
    with gr.Row(elem_id="input-row"):
        msg = gr.Textbox(
            placeholder="Message RAVEN...",
            label="",
            show_label=False,
            scale=10,
            container=False,
            elem_id="msg-box",
            lines=1,
            max_lines=3,
        )
        send = gr.Button("↑", elem_id="send-btn", scale=1, min_width=48)

    # Controls row
    with gr.Row(elem_id="controls-row"):
        mode = gr.Dropdown(
            choices=["DEFAULT", "SAVAGE", "PHILOSOPHER", "CHAOS", "NOX", "DURDEN"],
            value="DEFAULT",
            label="Mode",
            elem_id="mode-select",
            scale=2,
            show_label=False,
        )
        clear = gr.Button("RESET", elem_id="reset-btn", scale=1)

    # Footer
    gr.HTML(f"""
        <div id="raven-footer">
            <p>{total_params:,} parameters · {config.device.upper()}</p>
        </div>
    """)

    # Wiring — only msg, chatbot, mode as inputs (no temperature)
    send.click(respond, inputs=[msg, chatbot, mode], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, mode], outputs=[msg, chatbot])
    clear.click(clear_chat, outputs=[chatbot, msg])

demo.launch(css=CSS)
