from finetuning_utils import *

def generate_response(model, instruction, vocab, device, max_length=100, temperature=0.8):
    model.eval()

    # Tokenize instruction
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]
    pad_id = vocab["<pad>"]
    instr_tokens = instruction.strip().split()
    instr_ids = [vocab.get(tok, vocab["<unk>"]) for tok in instr_tokens]

    # Create input: [BOS] instruction [EOS]
    input_ids = [bos_id] + instr_ids + [eos_id]
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        for step in range(max_length):
            # Create mask for current sequence
            tgt_mask = Batch.make_std_mask(input_tensor, pad_id)

            # Forward pass - gets hidden states [1, seq, 512]
            hidden = model.forward(input_tensor, tgt_mask)

            logits = model.generator(hidden)  # [1, seq, 267737]

            # Get next token probabilities with temperature
            next_token_logits = logits[0, -1, :] / temperature

            # Apply simple repetition penalty
            if input_tensor.size(1) > 3:  # Only if we have some history
                for prev_token in input_tensor[0, -3:]:  # Last 3 tokens
                    if prev_token < len(next_token_logits):  # Safety check
                        next_token_logits[prev_token] -= 0.5  # Reduce probability

            # Sample instead of argmax
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            # Stop if we generate EOS
            if next_token == eos_id:
                break

            # Add token to sequence
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)

    # Decode the response part (after the separator EOS)
    response_start = len(instr_ids) + 2  # +1 for BOS, +1 for EOS
    if response_start < input_tensor.size(1):
        response_ids = input_tensor[0, response_start:].tolist()
    else:
        response_ids = []

    # Convert back to text
    id_to_vocab = {v: k for k, v in vocab.items()}
    response_tokens = [id_to_vocab.get(id, "<unk>") for id in response_ids]
    response = " ".join(response_tokens)

    return response

def evaluate_perplexity(model, val_dataloader, criterion, device, pad_id):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_data in val_dataloader:
            if isinstance(batch_data, tuple):
                batch_tensor, loss_start_idxs = batch_data
                batch = Batch(batch_tensor, loss_start_idxs, pad_id)
            else:
                batch = Batch(batch_data, None, pad_id)

            out = model.forward(batch.tgt, batch.tgt_mask)
            loss_compute = SimpleLossCompute(model.generator, criterion)

            if hasattr(batch, 'loss_mask') and batch.loss_mask is not None:
                loss, _ = loss_compute(out, batch.tgt_y, batch.ntokens, batch.loss_mask)
            else:
                loss, _ = loss_compute(out, batch.tgt_y, batch.ntokens)

            total_loss += loss
            total_tokens += batch.ntokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(avg_loss)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    return perplexity

def evaluate_generations(model, vocab, device):
    test_prompts = [
        # Simple prompts
        "Write a short poem about cats.",
        "Explain what photosynthesis is in simple terms.",
        "List 3 benefits of exercise.",

        # Complex prompts (Tulu-style)
        "Write a letter to a friend describing your dream vacation. The letter must be exactly 4 sentences long and include the word 'adventure' at least twice.",
        "Create a recipe for chocolate chip cookies, but write it as if you're a pirate. Include exactly 5 ingredients and use pirate terminology throughout.",
        "Explain quantum computing to a 10-year-old. Your explanation must use exactly 3 analogies and be no more than 100 words.",
    ]

    model.eval()
    for prompt in test_prompts:
        print(f"\n{'=' * 50}")
        print(f"PROMPT: {prompt}")
        print(f"{'=' * 50}")
        response = generate_response(model, prompt, vocab, device)
        print(f"RESPONSE: {response}")
        print("\n")

def check_unk_rate(pairs, vocab):
    total_tokens = 0
    unk_tokens = 0
    unk_id = vocab["<unk>"]

    for instruction, response in pairs[:100]:
        all_text = instruction + " " + response
        tokens = all_text.strip().split()
        total_tokens += len(tokens)

        for token in tokens:
            if vocab.get(token, unk_id) == unk_id:
                unk_tokens += 1

    unk_rate = unk_tokens / total_tokens * 100
    print(f"Unknown token rate: {unk_rate:.2f}%")
    return unk_rate

def filter_low_unk_examples(pairs, vocab, max_unk_rate=0.10):
    filtered_pairs = []
    unk_id = vocab["<unk>"]

    for instruction, response in pairs:
        all_text = instruction + " " + response
        tokens = all_text.strip().split()
        unk_count = sum(1 for tok in tokens if vocab.get(tok, unk_id) == unk_id)
        unk_rate = unk_count / len(tokens)

        if unk_rate < max_unk_rate:
            filtered_pairs.append((instruction, response))

    print(f"Filtered from {len(pairs)} to {len(filtered_pairs)} examples")
    return filtered_pairs