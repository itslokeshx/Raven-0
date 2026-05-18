import os
import sys
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
    model.load_state_dict(torch.load(weights_path, map_location=config.device))
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
        
        # Generate new tokens. Let's limit generation to 200 characters to keep it sharp
        # and prevent it from hallucinating infinitely.
        generated_idx = model.generate(idx, max_new_tokens=200)
        
        # 8. Decodes and prints the response
        response = tokenizer.decode(generated_idx[0].tolist())
        
        # Isolate the newly generated text (everything after the prompt we fed it)
        new_text = response[len(prompt):]
        
        # For a character-level LLM, we should cut it off if it starts hallucinating the next User turn
        if "User:" in new_text:
            new_text = new_text.split("User:")[0]
            
        print(f"RAVEN: {new_text.strip()}\n")

if __name__ == '__main__':
    run()
