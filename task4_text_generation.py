"""
TASK 4: TEXT GENERATION MODEL
Uses GPT-2 (HuggingFace) to generate coherent paragraphs on specific topics.
Libraries: transformers, torch
"""



from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

MODEL_NAME  = "gpt2"          # options: "gpt2", "gpt2-medium", "gpt2-large"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {DEVICE}")



def load_model(model_name: str = MODEL_NAME):
    print(f"\n⬇️  Loading {model_name} …")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model     = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    print("✅  Model loaded.")
    return tokenizer, model



def generate_text(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int  = 200,
    temperature: float   = 0.85,
    top_k: int           = 50,
    top_p: float         = 0.92,
    repetition_penalty: float = 1.2,
    num_return_sequences: int = 1,
) -> list[str]:
    """
    Generate text continuation for the given prompt.
    Returns a list of generated strings (length = num_return_sequences).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens       = max_new_tokens,
            do_sample            = True,
            temperature          = temperature,
            top_k                = top_k,
            top_p                = top_p,
            repetition_penalty   = repetition_penalty,
            num_return_sequences = num_return_sequences,
            pad_token_id         = tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    prompt_len = inputs["input_ids"].shape[-1]
    results = []
    for ids in output_ids:
        generated = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
        results.append(generated.strip())

    return results



DEMO_PROMPTS = [
    "Artificial intelligence is transforming the world because",
    "Climate change poses significant risks to humanity, including",
    "The future of space exploration depends on",
    "Quantum computing will revolutionize technology by",
]

def run_demo(tokenizer, model):
    print("\n" + "=" * 60)
    print("         AUTO-DEMO  —  Pre-defined prompts")
    print("=" * 60)

    for i, prompt in enumerate(DEMO_PROMPTS, 1):
        print(f"\n📝  Prompt {i}: {prompt}")
        print("-" * 60)
        outputs = generate_text(prompt, tokenizer, model, max_new_tokens=150)
        for j, text in enumerate(outputs, 1):
            print(f"Generated:\n{prompt} {text}\n")


def run_interactive(tokenizer, model):
    print("\n" + "=" * 60)
    print("         INTERACTIVE MODE  —  Enter your own prompt")
    print("         (type 'quit' to exit)")
    print("=" * 60)

    while True:
        prompt = input("\n📝  Your prompt : ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            print("👋  Exiting interactive mode.")
            break
        if not prompt:
            print("⚠️  Empty prompt — please try again.")
            continue

        try:
            max_tokens = int(input("   Max new tokens (default 200): ").strip() or "200")
        except ValueError:
            max_tokens = 200

        outputs = generate_text(prompt, tokenizer, model, max_new_tokens=max_tokens)
        print("\n🤖  Generated text:")
        print("-" * 60)
        for text in outputs:
            print(f"{prompt} {text}")
        print("-" * 60)



if __name__ == "__main__":
    print("=" * 60)
    print("          TASK 4 — TEXT GENERATION (GPT-2)")
    print("=" * 60)

    tokenizer, model = load_model()

    print("\nChoose mode:")
    print("  1 — Auto-demo (pre-defined prompts)")
    print("  2 — Interactive (enter your own prompts)")
    print("  3 — Both")
    choice = input("Enter choice [1/2/3] (default 1): ").strip() or "1"

    if choice == "1":
        run_demo(tokenizer, model)
    elif choice == "2":
        run_interactive(tokenizer, model)
    else:
        run_demo(tokenizer, model)
        run_interactive(tokenizer, model)

    print("\n" + "=" * 60)
    print("✅  Text generation complete!")
    print("=" * 60)
