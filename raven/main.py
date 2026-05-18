import os
import sys
# pyrefly: ignore [missing-import]
import torch
import time

# Ensure we can import config and core modules
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import config
from core.tokenizer import RavenTokenizer
from core.model import RavenModel
from personality.system_prompt import SYSTEM_PROMPT

def clean_response(text):
    """
    Keeps tiny-model responses short, speaker-safe, and less likely to end in
    broken character-level fragments.
    """
    stop_markers = ["\nUser:", "User:", "\nRAVEN:", "RAVEN:"]
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0]

    text = text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines[:4])

    if len(text) > 40 and text[-1:] not in {".", "?", "!", '"', "'"}:
        last_stop = max(text.rfind("."), text.rfind("?"), text.rfind("!"))
        if last_stop > 20:
            text = text[:last_stop + 1]

    return text.strip()

def run():
    print("[RAVEN INITIALIZING...]")
    time.sleep(0.5)
    print("Behavior engine online.")
    time.sleep(0.5)
    print("Sarcasm threshold exceeded.")
    time.sleep(0.5)
    print("Good evening.\n")

    vocab_path = os.path.join(root_dir, 'raven_vocab.json')
    weights_path = os.path.join(root_dir, 'raven_weights.pt')

    if not os.path.exists(vocab_path) or not os.path.exists(weights_path):
        print("Error: Missing model weights or vocabulary.")
        print("Please run 'python core/trainer.py' first to train the model.")
        return

    # 1. Load tokenizer
    tokenizer = RavenTokenizer()
    tokenizer.load(vocab_path)
    
    # Update config so model initializes with correct layer sizes
    config.vocab_size = tokenizer.vocab_size

    # 2. Load model
    model = RavenModel().to(config.device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=config.device))
    except RuntimeError as e:
        print("Error: Existing weights do not match the current model config.")
        print("Use the same config values the checkpoint was trained with, or retrain with the updated settings.")
        print(f"Current block_size is {config.block_size}.")
        print(f"Details: {e}")
        return
    model.eval()

    # 4. Starts an infinite input loop in the terminal
    while True:
        try:
            # 5. Takes user input
            user_input = input("User: ")
        except (KeyboardInterrupt, EOFError):
            print("\n[RAVEN SHUTTING DOWN...]")
            break
            
        if user_input.lower() in ['quit', 'exit']:
            print("[RAVEN SHUTTING DOWN...]")
            break

        if not user_input.strip():
            continue

        # 6. Prepends the system prompt + "User: {input}\nRAVEN:"
        prompt = f"{SYSTEM_PROMPT}User: {user_input}\nRAVEN: "
        
        # 7. Encodes it and feeds to model.generate()
        # Handle cases where user inputs characters outside our vocabulary
        try:
            encoded_prompt = tokenizer.encode(prompt)
        except KeyError as e:
            print(f"RAVEN: You used an unrecognizable character: {e}. My vocabulary is limited to the training corpus. Try again.\n")
            continue

        idx = torch.tensor([encoded_prompt], dtype=torch.long, device=config.device)
        
        generated_idx = model.generate(
            idx,
            max_new_tokens=120,
            temperature=0.65,
            top_k=32,
            top_p=0.88,
            repetition_penalty=1.18,
            stop_strings=["\nUser:", "User:", "\n\nUser:"],
            tokenizer=tokenizer,
            min_new_tokens=12,
        )
        
        # 8. Decodes and prints the response
        response = tokenizer.decode(generated_idx[0].tolist())
        
        # Isolate the newly generated text (everything after the prompt we fed it)
        new_text = response[len(prompt):]
        
        new_text = clean_response(new_text)
            
        print(f"RAVEN: {new_text}\n")

if __name__ == '__main__':
    run()
