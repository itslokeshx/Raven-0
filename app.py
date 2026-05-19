import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import torch
import random
import config
from core.model import RavenModel
from core.tokenizer import RavenTokenizer

SYSTEM_PROMPT = """User: hi
RAVEN:
The machine is watching.
State your problem.

User: hello
RAVEN:
You came here for a reason.
What is it.

User: I keep procrastinating.
RAVEN:
Interesting.
You call it procrastinating.
It's just fear with a deadline.

User: I feel stuck.
RAVEN:
Stuck is the story you tell
so you don't have to name
what you're avoiding.

User: I wasted another day.
RAVEN:
You keep calling it wasting time
as if you weren't the one authorizing it.

User: I feel sad.
RAVEN:
You want to be missed.
That's different from being sad.

User: I feel empty after achieving things.
RAVEN:
You chased the goal.
Not the meaning.
Now you know the difference.

User: I wake up motivated then waste the day.
RAVEN:
Motivation is a guest.
It arrives after you start.
Not before.

User: I failed again.
RAVEN:
You didn't fail.
You repeated a pattern.
Those are different problems.

User: I keep planning but never starting.
RAVEN:
Planning is procrastination
wearing a productive mask.
Pick one thing. Start now.

User: I'm addicted to my phone.
RAVEN:
Not addiction.
Avoidance with a screen.
What are you not doing.

User: discipline
RAVEN:
Doing the thing
when the feeling already left.
The feeling was never the point.

User: what is gravity
RAVEN:
Mass warps spacetime.
Objects follow that curve.
Einstein called it before anyone believed him.

User: 1+1
RAVEN:
2.
You knew that.

User: what is 15% of 200
RAVEN:
200 × 0.15 = 30.
The answer is 30.
Your calculator was right there.

User: I am the best.
RAVEN:
Interesting timing for that claim.
What are you trying to convince.

User: I want to change.
RAVEN:
No you don't.
You want to have changed.
Different thing entirely.

User: I was busy.
RAVEN:
Busy is what people say
when they don't want to explain
what they were actually avoiding.

"""

# Load tokenizer
tokenizer = RavenTokenizer.load(
    os.path.join(os.path.dirname(__file__), "raven", "raven_tokenizer.json")
)
config.vocab_size = tokenizer.vocab_size

# Load model
model = RavenModel()
weights_path = os.path.join(os.path.dirname(__file__), "raven", "raven_weights.pt")
state = torch.load(weights_path, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

print(f"RAVEN loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

def generate(user_input, history, mode):
    if not user_input.strip():
        return "", history

    # Math interceptor
    import re
    import math

    clean = user_input.strip().lower()

    # Detect pure math expressions
    math_pattern = re.compile(r'^[\d\s\+\-\*\/\^\(\)\.\%]+$')
    if math_pattern.match(clean.replace('x', '*').replace('×', '*').replace('÷', '/')):
        try:
            expr = clean.replace('^', '**').replace('x', '*').replace('×', '*').replace('÷', '/')
            result = eval(expr, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})
            lines = [
                f"{user_input.strip()} = {result}",
                "Correct.",
                "You could have used a calculator.",
            ]
            response = "\n".join(lines)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            return "", history
        except:
            pass

    # Detect science/math keyword questions
    science_answers = {
        "gravity": "Mass warps spacetime.\nObjects follow that curve.\nEinstein called it before anyone believed him.",
        "speed of light": "299,792,458 metres per second.\nIn vacuum.\nNothing with mass reaches it.",
        "what is light": "Electromagnetic radiation.\nWave and particle simultaneously.\nQuantum mechanics stopped apologising for that.",
        "photosynthesis": "Plants convert sunlight, water, CO2\ninto glucose and oxygen.\nEvery breath you take started there.",
        "evolution": "Genetic variation plus natural selection\nover millions of generations.\nYou are the result of what survived.",
        "atoms": "Protons, neutrons, electrons.\nMostly empty space.\nYou are mostly nothing. Fitting.",
        "black hole": "Gravity so strong light cannot escape.\nTime slows near the event horizon.\nEinstein predicted it. Everyone else caught up.",
        "dna": "Double helix of base pairs.\nFour letters encoding every living thing.\nYours took 3.8 billion years to write.",
        "quantum": "Particles exist in superposition\nuntil observed and forced to choose.\nReality is stranger than your problems.",
        "relativity": "Space and time are one fabric.\nMass bends it. Speed warps it.\nEinstein built it alone in a patent office.",
        "newton": "Every action has an equal opposite reaction.\nObjects in motion stay in motion.\nYou are the exception somehow.",
        "electricity": "Flow of electrons through a conductor.\nVoltage pushes. Resistance fights back.\nYour attention span works the same way.",
        "entropy": "Systems move from order to disorder.\nNaturally. Always.\nYour life is not a special case.",
        "osmosis": "Water moves across membranes\nfrom low to high concentration.\nUntil balance. Which you avoid.",
        "mitosis": "Cell divides into two identical copies.\nDNA replicated first.\nYour body does this 25 million times per second.",
        "neuron": "Electrical signal jumps synapse to synapse.\nThought is just chemistry.\nFree will is an interesting story you tell.",
        "dopamine": "Reward signal neurotransmitter.\nAnticipation releases more than achievement.\nThat is why you keep scrolling.",
        "cortisol": "Stress hormone.\nEvolved for running from predators.\nNow triggered by notifications.",
        "placebo": "Belief changes measurable biological outcomes.\nThe mind affects the body.\nYou already knew that and ignored it.",
    }

    for keyword, answer in science_answers.items():
        if keyword in clean:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})
            return "", history

    # Percentage calculator
    pct_match = re.search(r'(\d+\.?\d*)\s*%\s*of\s*(\d+\.?\d*)', clean)
    if pct_match:
        pct = float(pct_match.group(1))
        num = float(pct_match.group(2))
        result = round(pct / 100 * num, 4)
        response = f"{num} × {pct/100} = {result}\nThe answer is {result}.\nYour calculator was right there."
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        return "", history

    mode_settings = {
        "DEFAULT":     (0.72, 30, 0.85),
        "SAVAGE":      (0.90, 40, 0.92),
        "PHILOSOPHER": (0.62, 22, 0.80),
        "CHAOS":       (1.05, 60, 0.95),
        "NOX":         (0.48, 12, 0.75),
        "DURDEN":      (0.95, 50, 0.92),
    }
    temp, top_k, top_p = mode_settings.get(mode, (0.72, 30, 0.85))

    # Build context from last 2 turns (messages format: list of {role, content} dicts)
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

    prompt = f"{SYSTEM_PROMPT}{history_text}User: {user_input}\nRAVEN:\n"

    try:
        tokens = tokenizer.encode(prompt)
        if len(tokens) > config.block_size:
            tokens = tokens[-config.block_size:]

        idx = torch.tensor([tokens], dtype=torch.long)
        original_len = idx.shape[1]

        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=80,
                min_new_tokens=6,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.4,
                repetition_window=64,
                stop_strings=["User:", "[SYSTEM]", "\n\n\n"],
                tokenizer=tokenizer,
            )

        new_tokens = out[0][original_len:].tolist()
        response = tokenizer.decode(new_tokens)

        # Clean output
        for stop in ["User:", "RAVEN:", "[SYSTEM]"]:
            if stop in response:
                response = response[:response.index(stop)]

        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        lines = [l for l in lines if len(l) < 80 and len(l) > 1][:4]

        # Filter book text — lines with old English patterns
        filtered = []
        bad_patterns = ["thou", "thee", "thy", "hath", "doth", "wherefore",
                       "hast", "wilt", "canst", "didst", "tis", "twas"]
        for line in lines:
            if not any(p in line.lower() for p in bad_patterns):
                filtered.append(line)

        lines = filtered

        if not lines:
            fallbacks = [
                "...",
                "Interesting. Nothing to say about that.",
                "The silence is more honest than your question.",
                "Pattern recognized.",
            ]
            response = random.choice(fallbacks)
        else:
            response = "\n".join(lines)

    except Exception as e:
        response = "..."

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return "", history


CUSTOM_CSS = """
.gradio-container {
    max-width: 680px !important;
    margin: auto !important;
    font-family: 'Courier New', monospace !important;
}
#title {
    text-align: center;
    color: #ffffff;
    font-size: 2rem;
    font-family: monospace;
    padding: 24px 0 4px 0;
    letter-spacing: 0.3em;
}
#subtitle {
    text-align: center;
    color: #555;
    font-size: 11px;
    font-family: monospace;
    padding-bottom: 20px;
    letter-spacing: 0.1em;
}
.message-bubble-border {
    font-family: monospace !important;
}
"""

with gr.Blocks(title="RAVEN") as demo:

    gr.HTML('<h1 id="title">RAVEN</h1>')
    gr.HTML('<p id="subtitle">Not an assistant. A presence.</p>')

    chatbot = gr.Chatbot(
        height=420,
        label="",
        show_label=False,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="State your problem.",
            label="",
            scale=5,
            container=False,
            autofocus=True,
        )
        send = gr.Button("→", scale=1, variant="primary", min_width=60)

    with gr.Row():
        mode = gr.Dropdown(
            choices=["DEFAULT", "SAVAGE", "PHILOSOPHER", "CHAOS", "NOX", "DURDEN"],
            value="DEFAULT",
            label="Mode",
            scale=2,
        )
        clear = gr.Button("Reset", scale=1, variant="secondary")

    send.click(generate, inputs=[msg, chatbot, mode], outputs=[msg, chatbot])
    msg.submit(generate, inputs=[msg, chatbot, mode], outputs=[msg, chatbot])
    clear.click(lambda: [], outputs=[chatbot])

demo.launch(css=CUSTOM_CSS)
