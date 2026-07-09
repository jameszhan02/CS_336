import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from adapters import TransformerLM, Tokenizer

from inference_utils import load_model_and_tokenizer, generate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "run_xxx", "checkpoint_step4999_final.pt")
vocab_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-train.txtvocab.pkl")

model, tokenizer, config = load_model_and_tokenizer(
    checkpoint_path=checkpoint_path,
    vocab_path=vocab_path,
    model_cls=TransformerLM,
    tokenizer_cls=Tokenizer,
    special_tokens=["<|endoftext|>"],
    device=DEVICE,
)

prompt = "Long long time ago..."
prompt_tokens = tokenizer.encode(prompt)
eos_token_id = tokenizer.encode("<|endoftext|>")[0]

output_tokens = generate(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=400,
    context_length=config["context_length"],
    temperature=0.8,
    p=0.9,
    eos_token_id=eos_token_id,
    device=DEVICE,
)
generated_text = tokenizer.decode(output_tokens[len(prompt_tokens):])
print(generated_text)