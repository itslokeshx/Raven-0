import os
import random
import sys
import time
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from core.model import RavenModel
from core.tokenizer import RavenTokenizer
from personality.system_prompt import SYSTEM_PROMPT


BOOT_LINES = [
    "[RAVEN v2 BOOT]",
    "Core online.",
    "Voice sharpened.",
    "Patience budget: low.",
    "",
]

MODES = {
    "default": {"temperature": 0.8, "top_k": 40, "top_p": 0.9},
    "savage": {"temperature": 1.0, "top_k": 50, "top_p": 0.92},
    "philosopher": {"temperature": 0.7, "top_k": 32, "top_p": 0.88},
    "chaos": {"temperature": 1.15, "top_k": 80, "top_p": 0.95},
    "nox": {"temperature": 0.55, "top_k": 24, "top_p": 0.86},
    "durden": {"temperature": 1.05, "top_k": 64, "top_p": 0.92},
}

EMPTY_RESPONSES = [
    "...",
    "Interesting. Nothing to say about that.",
    "The silence is doing more work than the question.",
]

BOOT_DELAY = 0.03
LINE_DELAY = 0.02
MAX_NEW_TOKENS = 200
HISTORY_TURNS = 3
MAX_RESPONSE_LINES = 4


def sleep_print(line="", delay=BOOT_DELAY):
    print(line)
    time.sleep(delay)


def boot_sequence():
    for line in BOOT_LINES:
        sleep_print(line)


def candidate_paths(filename):
    return [directory / filename for directory in config.artifact_search_dirs()]


def print_missing_file_error(label, filename):
    print(f"[RAVEN ERROR] Missing {label}.")
    print("Expected one of:")
    for path in candidate_paths(filename):
        print(f"  {path}")


def torch_load(path, map_location=None, weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_state_dict_from_artifacts():
    weights_path = config.find_artifact(config.weights_file)
    if weights_path:
        payload = torch_load(weights_path, map_location=config.device, weights_only=True)
        return payload, weights_path

    checkpoint_path = config.find_artifact(config.latest_checkpoint_file)
    if checkpoint_path:
        payload = torch_load(checkpoint_path, map_location=config.device, weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            return payload["model_state_dict"], checkpoint_path

    return None, None


def load_runtime():
    tokenizer_path = config.find_artifact(config.tokenizer_file)
    if tokenizer_path is None:
        print_missing_file_error("BPE tokenizer", config.tokenizer_file)
        return None, None, 0

    try:
        tokenizer = RavenTokenizer.load(tokenizer_path)
    except Exception as exc:
        print(f"[RAVEN ERROR] Could not load tokenizer: {exc}")
        return None, None, 0

    config.vocab_size = tokenizer.vocab_size
    state_dict, model_path = load_state_dict_from_artifacts()
    if state_dict is None:
        print_missing_file_error("model weights or latest checkpoint", config.weights_file)
        return None, None, 0

    model = RavenModel().to(config.device)
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        print("[RAVEN ERROR] Model weights do not match this architecture.")
        print(f"Details: {exc}")
        return None, None, 0

    model.eval()
    total_params = model.get_num_params()
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Model: {model_path}")
    print(f"Parameters: {total_params:,}")
    print(f"Device: {config.device}")
    print("[RAVEN READY]\n")
    return model, tokenizer, total_params


def build_prompt(user_input, history):
    recent = "\n\n".join(history[-HISTORY_TURNS:])
    if recent:
        return f"{SYSTEM_PROMPT}\n{recent}\n\nUser: {user_input}\nRAVEN:"
    return f"{SYSTEM_PROMPT}\nUser: {user_input}\nRAVEN:"


def encode_prompt(prompt, tokenizer):
    ids = tokenizer.encode(prompt, add_bos=True)
    if not ids:
        return None
    return ids[-config.block_size :]


def decode_tokens(tokenizer, tokens):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return tokenizer.decode(tokens)


def trim_response(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "RAVEN:" in text:
        text = text.split("RAVEN:", 1)[-1]

    for marker in ("\nUser:", "User:", "\nSYSTEM:", "SYSTEM:", "\n[END SYSTEM]", "[END SYSTEM]", "<eos>"):
        if marker in text:
            text = text.split(marker, 1)[0]

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[:MAX_RESPONSE_LINES]).strip()


def generate_response(user_input, model, tokenizer, history, settings):
    prompt = build_prompt(user_input, history)
    encoded = encode_prompt(prompt, tokenizer)
    if encoded is None:
        return None

    idx = torch.tensor([encoded], dtype=torch.long, device=config.device)
    prompt_len = idx.size(1)

    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=settings["temperature"],
            top_k=settings["top_k"],
            top_p=settings["top_p"],
            repetition_penalty=1.15,
            repetition_window=min(96, config.block_size),
            stop_tokens=[tokenizer.eos_id],
            stop_strings=["\nUser:", "User:", "\n\nUser:"],
            tokenizer=tokenizer,
            min_new_tokens=1,
        )

    response = trim_response(decode_tokens(tokenizer, generated[0, prompt_len:]))
    if not response:
        response = trim_response(decode_tokens(tokenizer, generated[0]))
    return response


def print_response(response):
    response = trim_response(response)
    if not response:
        response = random.choice(EMPTY_RESPONSES)

    print("RAVEN:")
    for line in response.splitlines()[:MAX_RESPONSE_LINES]:
        sleep_print(line, LINE_DELAY)
    print()
    return response


def print_help():
    print("Commands:")
    print("  /help              Show commands")
    print("  /clear             Clear terminal")
    print("  /status            Show runtime status")
    print("  /exit              Disconnect")
    print("")
    print("Modes:")
    for name, settings in MODES.items():
        print(
            f"  /mode {name:<11} "
            f"temperature={settings['temperature']} "
            f"top_k={settings['top_k']} "
            f"top_p={settings['top_p']}"
        )
    print()


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_status(mode_name, total_params):
    settings = MODES[mode_name]
    print(f"Mode: {mode_name.upper()}")
    print(f"Temperature: {settings['temperature']}")
    print(f"Top-k: {settings['top_k']}")
    print(f"Top-p: {settings['top_p']}")
    print(f"Parameters: {total_params:,}")
    print(f"Device: {config.device}")
    print()


def handle_command(command, state, total_params):
    parts = command.strip().split()
    name = parts[0].lower()

    if name == "/help":
        print_help()
        return True
    if name == "/clear":
        clear_screen()
        return True
    if name == "/status":
        print_status(state["mode"], total_params)
        return True
    if name == "/exit":
        print("RAVEN: Leaving already.")
        return False
    if name == "/mode":
        if len(parts) != 2 or parts[1].lower() not in MODES:
            print("[RAVEN: Unknown mode. Type /help.]")
            return True
        state["mode"] = parts[1].lower()
        print(f"[MODE: {state['mode'].upper()}]")
        return True

    print("[RAVEN: Unknown command. Type /help.]")
    return True


def input_loop(model, tokenizer, total_params):
    history = []
    state = {"mode": "default"}

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nRAVEN: Disconnected.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not handle_command(user_input, state, total_params):
                break
            continue

        response = generate_response(user_input, model, tokenizer, history, MODES[state["mode"]])
        if response is None:
            print("[RAVEN: Input could not be encoded.]")
            continue

        response = print_response(response)
        history.append(f"User: {user_input}\nRAVEN: {response}")
        del history[:-HISTORY_TURNS]


def main():
    boot_sequence()
    model, tokenizer, total_params = load_runtime()
    if model is None or tokenizer is None:
        return
    input_loop(model, tokenizer, total_params)


if __name__ == "__main__":
    main()
