import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from adapters import TransformerLM, Tokenizer
import torch
from inference_utils import load_model_and_tokenizer, generate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "run_20260706_235713_final_lr0.003", "checkpoint_step4999_final.pt")
vocab_path = os.path.join(BASE_DIR, "data", "TinyStoriesV2-GPT4-train.txtvocab.pkl")

model, tokenizer, config = load_model_and_tokenizer(
    checkpoint_path=checkpoint_path,
    vocab_path=vocab_path,
    model_cls=TransformerLM,
    tokenizer_cls=Tokenizer,
    special_tokens=["<|endoftext|>"],
    device=DEVICE,
)

prompt = "This is an iceland saga, It's all start from a brave young man called James."
prompt_tokens = tokenizer.encode(prompt)
eos_token_id = tokenizer.encode("<|endoftext|>")[0]

# batch decode.
# TODO: 1. make sure they have same # of tokens | -> fix PAD issue later
batch_prompt = ["write a story here: ","write a story here: ","write a story here: "]
batch_prompt_tokens = [tokenizer.encode(prompt) for prompt in batch_prompt]



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

print(prompt)
print("============================================================")
print(generated_text)