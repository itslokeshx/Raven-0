"""
RAVEN v1 — Gradio Web Interface.

Provides a browser-based chat UI backed by the same model, tokenizer,
system prompt, and generation logic as the terminal main.py.
Compatible with Gradio 6.x.
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
    """Load tokenizer and model. Returns (model, tokenizer) or exits."""
    tok_path = find_file("raven_tokenizer.json")
    if not tok_path:
        print("[RAVEN ERROR] Tokenizer not found.")
        sys.exit(1)

    tokenizer = RavenTokenizer.load(tok_path)
    config.vocab_size = tokenizer.vocab_size

    # Try weights, then checkpoint
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

def generate(user_input, history, temperature, mode):
    """Generate a RAVEN response given user input and chat history."""
    if not user_input or not user_input.strip():
        return "", history

    user_input = user_input.strip()

    # Get mode settings
    mode_key = mode.lower() if mode else "default"
    settings = MODES.get(mode_key, MODES["default"])
    temp = settings["temperature"]
    top_k = settings["top_k"]
    top_p = settings["top_p"]

    # Validate / preprocess input (math prefix, science prefix, greeting nudge)
    processed = validate_input(user_input)

    # Build history context from last 2 turns
    history_text = ""
    for turn in (history or [])[-2:]:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            history_text += f"User: {turn[0]}\nRAVEN:\n{turn[1]}\n\n"

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
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.35,
                repetition_window=60,
                stop_strings=["User:", "RAVEN:", "[SYSTEM]"],
                tokenizer=tokenizer,
                stop_tokens=[tokenizer.eos_id],
            )

        # Decode only new tokens
        new_tokens = out[0][prompt_len:].tolist()
        response = tokenizer.decode(new_tokens)

        # Clean output — truncate at any leaked marker
        for marker in ("User:", "RAVEN:", "[SYSTEM]", "[END SYSTEM]"):
            pos = response.find(marker)
            if pos != -1:
                response = response[:pos]

        # Line-level cleanup
        lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
        lines = [l for l in lines if len(l) <= 80][:4]

        if not lines:
            response = random.choice(FALLBACK_RESPONSES)
        else:
            response = "\n".join(lines)

    except Exception:
        response = random.choice(FALLBACK_RESPONSES)

    history = history or []
    history.append((user_input, response))
    return "", history


# ---------------------------------------------------------------------------
# Gradio 6.x UI
# ---------------------------------------------------------------------------

CSS = """
body { background: #0a0a0a; }
.gradio-container {
    max-width: 700px;
    margin: auto;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
}
.message { font-family: monospace !important; }
#title {
    text-align: center;
    color: #ffffff;
    font-family: monospace;
    padding: 20px 0 5px 0;
}
#subtitle {
    text-align: center;
    color: #666;
    font-family: monospace;
    font-size: 12px;
    padding-bottom: 20px;
}
"""

with gr.Blocks(title="RAVEN") as demo:

    gr.HTML('<h1 id="title">RAVEN</h1>')
    gr.HTML('<p id="subtitle">Not an assistant. A presence.</p>')

    chatbot = gr.Chatbot(
        height=450,
        label="",
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="State your problem.",
            label="",
            scale=4,
            container=False,
        )
        send = gr.Button("→", scale=1, variant="primary")

    with gr.Row():
        mode = gr.Dropdown(
            choices=["DEFAULT", "SAVAGE", "PHILOSOPHER", "CHAOS", "NOX", "DURDEN"],
            value="DEFAULT",
            label="Mode",
            scale=2,
        )
        clear = gr.Button("Reset", scale=1)
        temperature = gr.Slider(
            minimum=0.1,
            maximum=1.5,
            value=0.75,
            step=0.05,
            label="Temperature",
            scale=3,
        )

    send.click(
        generate,
        inputs=[msg, chatbot, temperature, mode],
        outputs=[msg, chatbot],
    )
    msg.submit(
        generate,
        inputs=[msg, chatbot, temperature, mode],
        outputs=[msg, chatbot],
    )
    clear.click(lambda: [], outputs=[chatbot])

# Launch with theme and css passed here (Gradio 6.x API)
demo.launch(theme=gr.themes.Monochrome(), css=CSS)
