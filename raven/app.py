import gradio as gr
import torch
import os
import sys
import random

sys.path.append(os.path.dirname(__file__))

import config
from core.model import RavenModel
from core.tokenizer import RavenTokenizer

SYSTEM_PROMPT = """User: I keep procrastinating.
RAVEN:
Interesting.
You call it procrastinating.
It's just fear with a deadline.

User: I feel stuck.
RAVEN:
Stuck is the story you tell
so you don't have to name
what you're actually avoiding.

User: I feel sad.
RAVEN:
You want to be missed.
That's different from being sad.

User: hi
RAVEN:
The machine is watching.
State your problem.

"""

# Load model
tokenizer = RavenTokenizer.load("raven_tokenizer.json")
config.vocab_size = tokenizer.vocab_size

model = RavenModel()
weights_path = "raven_weights.pt"
state = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state)
model.eval()

def generate(user_input, history, temperature, mode):

    mode_settings = {
        "DEFAULT":     (0.75, 35, 0.88),
        "SAVAGE":      (0.95, 45, 0.93),
        "PHILOSOPHER": (0.65, 25, 0.82),
        "CHAOS":       (1.10, 70, 0.97),
        "NOX":         (0.50, 15, 0.78),
        "DURDEN":      (1.00, 55, 0.94),
    }

    temp, top_k, top_p = mode_settings.get(mode, (0.75, 35, 0.88))

    # Build history context
    history_text = ""
    for turn in history[-2:]:
        history_text += f"User: {turn[0]}\nRAVEN:\n{turn[1]}\n\n"

    prompt = f"{SYSTEM_PROMPT}{history_text}User: {user_input}\nRAVEN:\n"

    tokens = tokenizer.encode(prompt)
    if len(tokens) > config.block_size:
        tokens = tokens[-config.block_size:]

    idx = torch.tensor([tokens], dtype=torch.long)
    original_len = idx.shape[1]

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
            stop_strings=["User:", "[SYSTEM]"],
            tokenizer=tokenizer,
        )

    new_tokens = out[0][original_len:].tolist()
    response = tokenizer.decode(new_tokens)

    # Clean output
    for stop in ["User:", "RAVEN:", "[SYSTEM]"]:
        if stop in response:
            response = response[:response.index(stop)]

    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    lines = [l for l in lines if len(l) < 80][:4]

    if not lines:
        fallbacks = [
            "...",
            "Interesting. Nothing to say about that.",
            "The silence is more honest than your question.",
            "Pattern recognized. Filing it away.",
        ]
        response = random.choice(fallbacks)
    else:
        response = "\n".join(lines)

    history.append((user_input, response))
    return "", history


with gr.Blocks(
    theme=gr.themes.Monochrome(),
    title="RAVEN",
    css="""
    body { background: #0a0a0a; }
    .gradio-container { 
        max-width: 700px; 
        margin: auto;
        font-family: monospace;
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
) as demo:

    gr.HTML('<h1 id="title">RAVEN</h1>')
    gr.HTML('<p id="subtitle">Not an assistant. A presence.</p>')

    chatbot = gr.Chatbot(
        height=450,
        label="",
        bubble_full_width=False,
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
            choices=["DEFAULT","SAVAGE","PHILOSOPHER","CHAOS","NOX","DURDEN"],
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

demo.launch()
