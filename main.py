"""
RAVEN v1 — Main inference loop.

Handles boot sequence, model/tokenizer loading, generation with
post-processing, slash commands, mood modes, and conversation history.
All error paths are caught — the loop never crashes.
"""

import os
import random
import re
import sys
import time

# pyrefly: ignore [missing-import]
import torch

# ---------------------------------------------------------------------------
# Path setup — ensure the raven package root is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config
from core.model import RavenModel
from core.tokenizer import RavenTokenizer
from personality.system_prompt import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOOT_LINES = [
    "[RAVEN v1 — INITIALIZING]",
    "████████████████████████ 100%",
    "",
    "Scanning environment............  OK",
    "Loading behavior engine.........  OK",
    "Calibrating sarcasm threshold...  EXCEEDED",
    "Checking patience levels........  LOW",
    "Detecting user..................  FOUND",
    "",
    "Good evening.",
    "The machine is watching.",
    "",
]

BOOT_DELAY = 0.04
LINE_DELAY = 0.025

# Mood modes — each adjusts generation sampling
MODES = {
    "default":     {"temperature": 0.75, "top_k": 35, "top_p": 0.88},
    "savage":      {"temperature": 0.95, "top_k": 45, "top_p": 0.93},
    "philosopher": {"temperature": 0.65, "top_k": 25, "top_p": 0.82},
    "chaos":       {"temperature": 1.1,  "top_k": 70, "top_p": 0.97},
    "nox":         {"temperature": 0.5,  "top_k": 15, "top_p": 0.78},
    "durden":      {"temperature": 1.0,  "top_k": 55, "top_p": 0.94},
}

# Fallback responses when generation produces nothing usable
FALLBACK_RESPONSES = [
    "...",
    "Interesting. Nothing to say about that.",
    "The silence is more honest than your question.",
    "Pattern recognized. Filing it away.",
]

# Single-word greetings to detect
GREETINGS = {"hi", "hello", "hey", "sup", "yo"}

# Science keywords — if input contains these, prepend "Explain: "
SCIENCE_KEYWORDS = {
    "gravity", "light", "energy", "atom", "electron", "photon",
    "quantum", "evolution", "dna", "cell", "planet", "star",
    "black hole", "magnetism", "electricity", "radiation",
    "thermodynamics", "entropy", "relativity", "velocity",
    "acceleration", "force", "mass", "wave", "frequency",
    "molecule", "chemical", "reaction", "nucleus", "proton",
    "neutron", "solar", "lunar", "orbit", "galaxy", "universe",
    "biology", "physics", "chemistry", "neuron", "synapse",
    "photosynthesis", "mitosis", "osmosis", "diffusion",
}

# Math keywords — natural language math questions
MATH_KEYWORDS = {
    "plus", "minus", "times", "divide", "multiply", "sum",
    "subtract", "add", "square root", "percentage", "percent",
    "factorial", "power of", "squared", "cubed", "average",
}

# Max lines in a response
MAX_RESPONSE_LINES = 4

# Max characters per line
MAX_LINE_LENGTH = 80

# History depth — keep last 4 turns
HISTORY_DEPTH = 4

# Last 2 turns fed into the prompt context
PROMPT_HISTORY_DEPTH = 2

# Max user input length
MAX_INPUT_LENGTH = 500

# Pattern to detect math-only input (digits, operators, spaces, parens)
MATH_PATTERN = re.compile(r"^[\d\s\+\-\*\/\%\^\(\)\.]+$")


# ===========================================================================
# FEATURE 1 — Boot Sequence
# ===========================================================================

def boot_sequence():
    """Print the RAVEN boot animation with delays."""
    for line in BOOT_LINES:
        print(line)
        time.sleep(BOOT_DELAY)


# ===========================================================================
# FEATURE 2 — File Loading
# ===========================================================================

def find_file(filename):
    """
    Search for a file in priority order:
    1. Same directory as main.py
    2. /content/drive/MyDrive/raven_checkpoints/
    Returns the path string if found, else None.
    """
    # Priority 1: same directory as this script
    local_path = os.path.join(SCRIPT_DIR, filename)
    if os.path.isfile(local_path):
        return local_path

    # Priority 2: Google Drive checkpoint directory
    drive_path = os.path.join("/content/drive/MyDrive/raven_checkpoints", filename)
    if os.path.isfile(drive_path):
        return drive_path

    return None


def find_weights():
    """
    Locate model weights in priority order:
    1. raven_weights.pt (local → drive)
    2. raven_checkpoint_latest.pt (local → drive)
    Returns (path, is_checkpoint) or (None, False).
    """
    # Try primary weights file first
    path = find_file("raven_weights.pt")
    if path:
        return path, False

    # Try latest checkpoint as fallback
    path = find_file("raven_checkpoint_latest.pt")
    if path:
        return path, True

    return None, False


def torch_load(path, map_location=None, weights_only=False):
    """Load a torch file, handling older PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_runtime():
    """
    Load tokenizer and model weights. Returns (model, tokenizer, param_count)
    or (None, None, 0) on failure.
    """
    # --- Tokenizer ---
    tokenizer_path = find_file("raven_tokenizer.json")
    if tokenizer_path is None:
        print("[RAVEN ERROR] Missing tokenizer.")
        print("Searched:")
        print(f"  {os.path.join(SCRIPT_DIR, 'raven_tokenizer.json')}")
        print(f"  /content/drive/MyDrive/raven_checkpoints/raven_tokenizer.json")
        return None, None, 0

    try:
        tokenizer = RavenTokenizer.load(tokenizer_path)
    except Exception as exc:
        print(f"[RAVEN ERROR] Could not load tokenizer: {exc}")
        return None, None, 0

    # --- Weights ---
    weights_path, is_checkpoint = find_weights()
    if weights_path is None:
        print("[RAVEN ERROR] Missing model weights.")
        print("Searched:")
        print(f"  {os.path.join(SCRIPT_DIR, 'raven_weights.pt')}")
        print(f"  /content/drive/MyDrive/raven_checkpoints/raven_weights.pt")
        print(f"  {os.path.join(SCRIPT_DIR, 'raven_checkpoint_latest.pt')}")
        print(f"  /content/drive/MyDrive/raven_checkpoints/raven_checkpoint_latest.pt")
        return None, None, 0

    try:
        payload = torch_load(
            weights_path,
            map_location=config.device,
            weights_only=not is_checkpoint,
        )
    except Exception as exc:
        print(f"[RAVEN ERROR] Failed to load weights: {exc}")
        return None, None, 0

    # Extract state dict from checkpoint if needed
    if is_checkpoint and isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict) and "model_state_dict" not in payload:
        # Assume it's a raw state dict
        state_dict = payload
    else:
        state_dict = payload

    # --- Build model ---
    config.vocab_size = tokenizer.vocab_size
    model = RavenModel().to(config.device)

    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        print("[RAVEN ERROR] Model weights do not match this architecture.")
        print(f"Details: {exc}")
        return None, None, 0

    model.eval()
    total_params = model.get_num_params()

    # Print load summary
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Model: {weights_path}")
    print(f"Parameters: {total_params:,}")
    print(f"Device: {config.device}")
    print("[RAVEN READY]")
    print()

    return model, tokenizer, total_params


# ===========================================================================
# FEATURE 3 — Generation Function
# ===========================================================================

def generate_response(user_input, model, tokenizer, history,
                      temperature, top_k, top_p,
                      repetition_penalty):
    """
    Build prompt from system prompt + history + user input,
    run model.generate(), decode and clean the output.
    """
    # --- Build the prompt ---
    recent_history = history[-PROMPT_HISTORY_DEPTH:]
    history_block = "\n".join(recent_history) if recent_history else ""

    if history_block:
        prompt = f"{SYSTEM_PROMPT}\n{history_block}\nUser: {user_input}\nRAVEN:"
    else:
        prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nRAVEN:"

    # --- Encode and crop from the LEFT to fit block_size ---
    encoded = tokenizer.encode(prompt, add_bos=True)
    if not encoded:
        return random.choice(FALLBACK_RESPONSES)

    # Keep most recent tokens if prompt exceeds block_size
    if len(encoded) > config.block_size:
        encoded = encoded[-config.block_size:]

    # --- Prepare tensor ---
    idx = torch.tensor([encoded], dtype=torch.long, device=config.device)
    prompt_len = idx.size(1)

    # --- Generate ---
    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=120,
            min_new_tokens=10,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=60,
            stop_strings=["User:", "RAVEN:", "[SYSTEM]"],
            tokenizer=tokenizer,
            stop_tokens=[tokenizer.eos_id],
        )

    # --- Decode only the NEW tokens ---
    new_tokens = generated[0, prompt_len:].tolist()
    raw_text = tokenizer.decode(new_tokens)

    # --- Clean the output ---
    text = raw_text.strip()

    # Truncate at any stop marker that bled into the decoded output
    for marker in ("User:", "RAVEN:", "[SYSTEM]", "[END SYSTEM]"):
        idx_pos = text.find(marker)
        if idx_pos != -1:
            text = text[:idx_pos]

    text = text.strip()

    # Split into lines, remove blanks
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Remove lines longer than 80 characters
    lines = [line for line in lines if len(line) <= MAX_LINE_LENGTH]

    # Keep maximum 4 lines
    lines = lines[:MAX_RESPONSE_LINES]

    # If nothing usable remains, use fallback
    if len(lines) < 1:
        return random.choice(FALLBACK_RESPONSES)

    return "\n".join(lines)


# ===========================================================================
# FEATURE 4 — Typing Effect
# ===========================================================================

def print_response(response):
    """Print RAVEN's response with a typing delay per line."""
    print("RAVEN:")
    for line in response.splitlines()[:MAX_RESPONSE_LINES]:
        print(line)
        time.sleep(LINE_DELAY)
    print()


# ===========================================================================
# FEATURE 5 — Conversation History
# ===========================================================================

def update_history(history, user_input, response):
    """Append a turn and trim to HISTORY_DEPTH."""
    turn = f"User: {user_input}\nRAVEN:\n{response}"
    history.append(turn)
    # Keep only the last HISTORY_DEPTH turns
    del history[:-HISTORY_DEPTH]


# ===========================================================================
# FEATURE 6 — Mood Modes
# ===========================================================================

def handle_mode(parts, state):
    """Switch mood mode. Returns True if handled."""
    if len(parts) != 2 or parts[1].lower() not in MODES:
        print("[Unknown mode. Type /help]")
        return
    mode_name = parts[1].lower()
    state["mode"] = mode_name
    state["temperature"] = MODES[mode_name]["temperature"]
    state["top_k"] = MODES[mode_name]["top_k"]
    state["top_p"] = MODES[mode_name]["top_p"]
    print(f"[MODE: {mode_name.upper()} ACTIVATED]")


# ===========================================================================
# FEATURE 7 — Slash Commands
# ===========================================================================

def print_help():
    """Print available commands."""
    print("  Commands:")
    print("  /mode default|savage|philosopher|chaos|nox|durden")
    print("  /status")
    print("  /clear")
    print("  /reset")
    print("  /exit")
    print()


def print_status(state, total_params):
    """Print current runtime status."""
    print(f"  Mode: {state['mode'].upper()}")
    print(f"  Temperature: {state['temperature']}")
    print(f"  Top-K: {state['top_k']}")
    print(f"  Top-P: {state['top_p']}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Device: {config.device}")
    print()


def handle_command(user_input, state, history, total_params):
    """
    Process slash commands. Returns:
      True  — command handled, continue loop
      False — exit requested
    """
    parts = user_input.strip().split()
    cmd = parts[0].lower()

    if cmd == "/help":
        print_help()
        return True

    if cmd == "/status":
        print_status(state, total_params)
        return True

    if cmd == "/clear":
        os.system("clear")
        return True

    if cmd == "/reset":
        history.clear()
        print("[CONVERSATION RESET]")
        print()
        return True

    if cmd == "/exit":
        print("RAVEN: Leaving already.")
        print("       Pattern recognized.")
        sys.exit(0)

    if cmd == "/mode":
        handle_mode(parts, state)
        return True

    # Unknown command
    print("[Unknown command. Type /help]")
    return True


# ===========================================================================
# FEATURE 9 — Input Validation
# ===========================================================================

def validate_input(user_input):
    """
    Pre-process user input:
    - Truncate to MAX_INPUT_LENGTH
    - Prepend "Calculate: " for pure math expressions or math keyword questions
    - Prepend "Explain: " for science keyword questions
    - Append " RAVEN." for single-word greetings
    """
    # Truncate long input
    if len(user_input) > MAX_INPUT_LENGTH:
        user_input = user_input[:MAX_INPUT_LENGTH]

    lower = user_input.lower().strip()

    # Detect math-only input (digits, operators, spaces, parens, dots)
    if MATH_PATTERN.match(user_input):
        user_input = f"Calculate: {user_input}"

    # Detect natural-language math questions ("what is 5 plus 3")
    elif any(kw in lower for kw in MATH_KEYWORDS):
        user_input = f"Calculate: {user_input}"

    # Detect science questions by keyword presence
    elif any(kw in lower for kw in SCIENCE_KEYWORDS):
        user_input = f"Explain: {user_input}"

    # Detect single-word greetings
    elif lower in GREETINGS:
        user_input = f"{user_input} RAVEN."

    return user_input


# ===========================================================================
# FEATURE 8 — Input Loop
# ===========================================================================

def input_loop(model, tokenizer, total_params):
    """Main REPL loop — reads input, generates responses, handles commands."""
    history = []
    state = {
        "mode": "default",
        "temperature": MODES["default"]["temperature"],
        "top_k": MODES["default"]["top_k"],
        "top_p": MODES["default"]["top_p"],
    }
    repetition_penalty = 1.35

    while True:
        try:
            # Print prompt without newline, read input
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nRAVEN: Disconnected.")
            print("       The patterns remain.")
            sys.exit(0)

        # Skip empty input
        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            if not handle_command(user_input, state, history, total_params):
                break
            continue

        # Validate and preprocess input
        user_input = validate_input(user_input)

        # Generate response — never crash
        try:
            response = generate_response(
                user_input, model, tokenizer, history,
                temperature=state["temperature"],
                top_k=state["top_k"],
                top_p=state["top_p"],
                repetition_penalty=repetition_penalty,
            )
        except Exception:
            response = random.choice(FALLBACK_RESPONSES)

        # Print with typing effect
        print_response(response)

        # Update conversation history
        update_history(history, user_input, response)


# ===========================================================================
# Entry Point
# ===========================================================================

def main():
    """Boot RAVEN, load model, enter conversation loop."""
    boot_sequence()
    model, tokenizer, total_params = load_runtime()
    if model is None or tokenizer is None:
        sys.exit(1)
    input_loop(model, tokenizer, total_params)


if __name__ == "__main__":
    main()
