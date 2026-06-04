from adapters import AdamW, run_get_batch, run_transformer_lm, TransformerLM, run_cross_entropy, run_save_checkpoint
import numpy as np
import torch

import torch.nn.functional as F

model = TransformerLM(...) # TODO: inintal transformerLM here
optimizer = AdamW(model.parameters())

def train(model, optimizer, train_path, val_path, max_steps, log_interval, eval_interval, save_interval, save_path):
    # TODO: change these to var
    BATCH_SIZE = 20
    TEST_BATCH_SIZE = 80
    CONTEXT_LENGTH = 100
    DEVICE = 'cpu'

    # train_path = "" # fill in the real path 
    # val_path = ""
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r") 
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r") 

    for step in range(max_steps):
        model.train()
        # 1. get a batch
        train_x, train_y = run_get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        # 2. forward pass + loss
        logits = model(train_x)
        loss = run_cross_entropy(logits, train_y)
        
        # 3. backprop + optimizer step
        optimizer.zero_grad()   # clear gradients from last step
        loss.backward()         # compute gradients
        optimizer.step()        # update weights
        # 4. every N steps — log train loss
        if step % log_interval == 0:
            print(f"step {step} | train loss: {loss.item():.4f}")
        # 5. every N steps — check validation loss
        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():   # don't compute gradients during eval
                val_x, val_y = run_get_batch(val_data, TEST_BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
                val_logits = model(val_x)
                val_loss = run_cross_entropy(val_logits, val_y)
            print(f"step {step} | val loss: {val_loss.item():.4f}")
            model.train()  # switch back to train mode
        
        # 6. every N steps — save checkpoint
        if step % save_interval == 0:
            file_name = f"checkpoint_step{step}.pt"
            final_out_path = save_path + '/' + file_name
            run_save_checkpoint(model, optimizer, step, final_out_path)
            print(f"step {step} | checkpoint saved to {final_out_path}")




    

def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    logits: (vocab_size,)
    return: (vocab_size,) 
    """
    return F.softmax(logits / temperature, dim=-1)


def nucleus_sampling(probs: torch.Tensor, p: float) -> int:
    """
    probs: (vocab_size,)
    p: nucleus 
    return: 
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_probs[cumulative_probs - sorted_probs > p] = 0.0
    
    sorted_probs = sorted_probs / sorted_probs.sum()
    
    sampled_index = torch.multinomial(sorted_probs, num_samples=1).item()
    
    return sorted_indices[sampled_index].item()


def decode(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    p: float = 0.9,
    eos_token_id: int = None,
    device: str = "cpu"
) -> list[int]:
    """
    model:  TransformerLM
    prompt_tokens: tokenizer.encode(prompt) token ids
    max_new_tokens: max generate token
    temperature: < 1 more stable | > 1 more random
    p: nucleus sampling threshold
    eos_token_id: <|endoftext|> stop at end of text token id
    """
    model.eval()
    tokens = list(prompt_tokens)  

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

            logits = model(x)           # (1, sequence_length, vocab_size)
            logits = logits[0, -1, :]   # (vocab_size,)

            # 3. temperature softmax
            probs = softmax_with_temperature(logits, temperature)

            # 4. nucleus sampling 
            next_token = nucleus_sampling(probs, p)

            # 5. append
            tokens.append(next_token)

            # 6. ending token condition
            if eos_token_id is not None and next_token == eos_token_id:
                break

    return tokens