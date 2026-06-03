from adapters import AdamW, run_get_batch, run_transformer_lm, TransformerLM, run_cross_entropy


train_path = ""
val_path = ""
train_data = np.memmap(train_path, dtype=np.uint16, mode="r") 
val_data = np.memmap(val_path, dtype=np.uint16, mode="r") 
model # TODO: inintal transformerLM here
optimizer = AdamW(model.parameters())
def train(model, optimizer, train_data, val_data, max_steps, log_interval, eval_interval, save_interval):
    # TODO: change these to var
    BATCH_SIZE = 20
    CONTEXT_LENGTH = 100
    DEVICE = 'cpu'
    

    for step in range(max_steps):
        # 1. get a batch
        # 2. forward pass + loss
        train_x, train_y = run_get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        logits = model(train_x)
        loss = run_cross_entropy(logits, train_y)
        
        # 3. backprop + optimizer step
        
        # 4. every N steps — log train loss
        if step % log_interval == 0:
        
        # 5. every N steps — check validation loss
        if step % eval_interval == 0:
            val_x, val_y = run_get_batch(val_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)

        
        # 6. every N steps — save checkpoint
        if step % save_interval == 0: