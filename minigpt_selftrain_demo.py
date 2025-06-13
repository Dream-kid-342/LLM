# Demo: Show how MiniGPT can "learn" from errors and unknown prompts via the error_memory in mini_gpt.py

from mini_gpt import error_memory, learn_from_error, review_error_memory, CharTokenizer, MiniGPT, train

def simulate_learning():
    # Simulate some errors and unknowns
    learn_from_error("foobar command", "Command not recognized or supported.")
    learn_from_error("ls -z", "ls: invalid option -- 'z'")
    learn_from_error("open secret.txt", "File not found: secret.txt")

    print("\n--- Error Memory ---")
    review_error_memory()

    # Optionally, use error_memory to further train the model (toy example)
    # Here, we just concatenate the prompts and errors as new "training data"
    if error_memory:
        print("\n--- Simulating further training on error prompts ---")
        # Create a simple dataset from error prompts and errors
        error_text = "\n".join([f"Prompt: {e['prompt']}\nError: {e['error']}" for e in error_memory])
        tokenizer = CharTokenizer(error_text)
        vocab_size = len(tokenizer.chars)
        block_size = 32
        model = MiniGPT(vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=block_size)
        train(model, error_text, tokenizer, block_size, epochs=2)
        print("Training complete on error prompts.")

if __name__ == "__main__":
    simulate_learning()
