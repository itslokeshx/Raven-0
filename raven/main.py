import os
import random
import sys
import time

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
from core.model import RavenModel
from core.tokenizer import RavenTokenizer
from personality.system_prompt import SYSTEM_PROMPT


BOOT_LINES = [
    "[RAVEN v1 — BOOT SEQUENCE]",
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

MODES = {
    "default": {"temperature": 0.8, "top_k": 40},
    "savage": {"temperature": 1.0, "top_k": 50},
    "philosopher": {"temperature": 0.7, "top_k": 30},
    "chaos": {"temperature": 1.2, "top_k": 80},
    "nox": {"temperature": 0.5, "top_k": 20},
    "durden": {"temperature": 1.1, "top_k": 60},
}

EMPTY_RESPONSES = [
    "...",
    "Interesting. Nothing to say about that.",
    "The silence is more honest than your question.",
]

BOOT_DELAY = 0.05
LINE_DELAY = 0.03
MAX_NEW_TOKENS = 200
HISTORY_TURNS = 3
MAX_RESPONSE_LINES = 4


def sleep_print(line="", delay=BOOT_DELAY):
    print(line)
    time.sleep(delay)


def boot_sequence():
    for line in BOOT_LINES:
        sleep_print(line, BOOT_DELAY)


def candidate_paths(filename):
    paths = [os.path.join(ROOT_DIR, filename)]
    checkpoint_dir = getattr(config, "checkpoint_dir", "")
    if checkpoint_dir:
        paths.append(os.path.join(checkpoint_dir, filename))
    return paths


def find_existing_path(filename):
    for path in candidate_paths(filename):
        if os.path.exists(path):
            return path
    return None


def print_missing_file_error(label, filename):
    print(f"[RAVEN ERROR] Missing {label}.")
    print("Expected one of:")
    for path in candidate_paths(filename):
        print(f"  {path}")


def safe_torch_load(path):
    try:
        return torch.load(path, map_location=config.device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=config.device)
    except Exception:
        try:
            return torch.load(path, map_location=config.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=config.device)


def load_state_dict(path):
    payload = safe_torch_load(path)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def load_runtime():
    tokenizer_file = getattr(config, "tokenizer_file", "raven_tokenizer.json")
    weights_file = getattr(config, "weights_file", "raven_weights.pt")

    tokenizer_path = find_existing_path(tokenizer_file)
    weights_path = find_existing_path(weights_file)

    if tokenizer_path is None:
        print_missing_file_error("BPE tokenizer", tokenizer_file)
        return None, None, 0

    if weights_path is None:
        print_missing_file_error("model weights", weights_file)
        return None, None, 0

    try:
        tokenizer = RavenTokenizer.load(tokenizer_path)
    except Exception as exc:
        print(f"[RAVEN ERROR] Could not load tokenizer: {exc}")
        return None, None, 0

    config.vocab_size = tokenizer.vocab_size

    model = RavenModel().to(config.device)
    try:
        model.load_state_dict(load_state_dict(weights_path))
    except Exception as exc:
        print("[RAVEN ERROR] Model weights do not match this architecture.")
        print(f"Details: {exc}")
        return None, None, 0

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Device: {config.device}")
    print("[RAVEN READY]\n")
    return model, tokenizer, total_params


def build_prompt(user_input, history):
    recent_history = "\n\n".join(history[-HISTORY_TURNS:])
    if recent_history:
        return f"{SYSTEM_PROMPT}\n{recent_history}\n\nUser: {user_input}\nRAVEN:"
    return f"{SYSTEM_PROMPT}\nUser: {user_input}\nRAVEN:"


def encode_prompt(prompt, tokenizer):
    try:
        encoded = tokenizer.encode(prompt, add_bos=True)
    except TypeError:
        encoded = tokenizer.encode(prompt)
    except Exception:
        return None

    if not encoded:
        return None

    return encoded[-config.block_size:]


def decode_tokens(tokenizer, tokens):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return tokenizer.decode(tokens)


def trim_response(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if "RAVEN:" in text:
        text = text.split("RAVEN:", 1)[-1]

    stop_markers = [
        "\nUser:",
        "User:",
        "\nSYSTEM:",
        "SYSTEM:",
        "\n[END SYSTEM]",
        "[END SYSTEM]",
        "<eos>",
    ]
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker, 1)[0]

    lines = text.strip().splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    cleaned = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned[:MAX_RESPONSE_LINES]).strip()


def generate_response(user_input, model, tokenizer, history, temperature, top_k):
    prompt = build_prompt(user_input, history)
    encoded = encode_prompt(prompt, tokenizer)
    if encoded is None:
        return None

    idx = torch.tensor([encoded], dtype=torch.long, device=config.device)
    prompt_len = idx.shape[1]

    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            top_k=top_k,
            top_p=None,
            repetition_penalty=1.15,
            repetition_window=min(96, config.block_size),
            stop_tokens=[tokenizer.eos_id] if getattr(tokenizer, "eos_id", None) is not None else None,
            stop_strings=["\nUser:", "User:", "\n\nUser:"],
            tokenizer=tokenizer,
            min_new_tokens=1,
        )

    generated_tokens = generated[0, prompt_len:]
    response = trim_response(decode_tokens(tokenizer, generated_tokens))

    if not response:
        full_text = decode_tokens(tokenizer, generated[0])
        response = trim_response(full_text)

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
    print("  /status            Show current runtime status")
    print("  /exit              Disconnect")
    print("")
    print("Modes:")
    for name, settings in MODES.items():
        print(f"  /mode {name:<11} temperature={settings['temperature']} top_k={settings['top_k']}")
    print()


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_status(mode_name, temperature, top_k, total_params):
    print(f"Mode: {mode_name.upper()}")
    print(f"Temperature: {temperature}")
    print(f"Top-k: {top_k}")
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
        print_status(state["mode"], state["temperature"], state["top_k"], total_params)
        return True

    if name == "/exit":
        print("RAVEN: Leaving already.")
        print("       Pattern recognized.")
        return False

    if name == "/mode":
        if len(parts) != 2 or parts[1].lower() not in MODES:
            print("[RAVEN: Unknown mode. Type /help.]")
            return True

        mode_name = parts[1].lower()
        state["mode"] = mode_name
        state["temperature"] = MODES[mode_name]["temperature"]
        state["top_k"] = MODES[mode_name]["top_k"]
        print(f"[MODE: {mode_name.upper()} ACTIVATED]")
        return True

    print("[RAVEN: Unknown command. Type /help.]")
    return True


def input_loop(model, tokenizer, total_params):
    history = []
    state = {
        "mode": "default",
        "temperature": MODES["default"]["temperature"],
        "top_k": MODES["default"]["top_k"],
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nRAVEN: Disconnected.")
            print("       The patterns remain.")
            break
        except EOFError:
            print("\nRAVEN: Disconnected.")
            print("       The patterns remain.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not handle_command(user_input, state, total_params):
                break
            continue

        response = generate_response(
            user_input,
            model,
            tokenizer,
            history,
            state["temperature"],
            state["top_k"],
        )

        if response is None:
            print("[RAVEN: Input contains unknown characters.]")
            continue

        response = print_response(response)
        history.append(f"User: {user_input}\nRAVEN: {response}")


def main():
    boot_sequence()
    model, tokenizer, total_params = load_runtime()
    if model is None or tokenizer is None:
        return
    input_loop(model, tokenizer, total_params)


if __name__ == "__main__":
    main()
