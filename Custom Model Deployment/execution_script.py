from evaluation_utils import *

def load_trained_model():
    config = {
        "batch_size": 16,
        "distributed": False,
        "num_epochs": 20,
        "accum_iter": 4,
        "base_lr": 1.0,
        "max_padding": 128,
        "warmup": 3000,
        "file_prefix": "finetuned_",
    }

    from datasets import load_dataset
    import random
    from sklearn.model_selection import train_test_split
    import torch
    import os

    # Load vocab from file
    vocab_path = "vocab_leader.pt"
    vocab = torch.load(vocab_path)
    pad_id = vocab["<pad>"]
    print(f"Loaded vocab with size: {len(vocab)}")
    print("Pad token ID:", pad_id)

    # Load full dataset
    dataset = load_dataset("allenai/tulu-3-sft-personas-instruction-following")["train"]

    # Sample 75%
    random.seed(42)
    subset_size = int(len(dataset) * 0.75)
    dataset = dataset.select(random.sample(range(len(dataset)), subset_size))
    print(f"Sampling {subset_size} examples from Tulu.")

    # Extract instruction-response pairs
    pairs = []
    for example in dataset:
        messages = example.get("messages", [])
        if len(messages) >= 2:
            # Ensure order is user -> assistant
            if messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
                instruction = messages[0]["content"].strip()
                response = messages[1]["content"].strip()
                pairs.append((instruction, response))
            else:
                print(f"Skipping unexpected role order: {[m['role'] for m in messages]}")

    pairs = filter_low_unk_examples(pairs, vocab, max_unk_rate=0.1)

    # Tokenize pairs using loaded vocab
    input_ids_list, loss_start_idxs = tokenize_pairs_to_ids(pairs, vocab)

    train, val = train_test_split(list(zip(input_ids_list, loss_start_idxs)), test_size=0.1, random_state=42)

    # Create model
    model = make_model(len(vocab))

    # Train model
    model_path = "finetuned_04.pt"
    if not os.path.exists(model_path):
        train_model(train, val, vocab, pad_id, config)

    # Load the appropriate checkpoint
    if os.path.exists(model_path):
        print(f"Loading SFT checkpoint from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        print("Loading pretrained wheels from leaderboard_model_07.pt...")
        model.load_state_dict(torch.load("leaderboard_model_07.pt", map_location="cpu"))
        print("Pretrained weights loaded for evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create validation dataloader and criterion for evaluation
    _, val_dataloader = create_dataloaders(
        device,
        vocab,
        train,
        val,
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
        is_distributed=False,
    )

    criterion = LabelSmoothing(size=len(vocab), padding_idx=pad_id, smoothing=0.1)
    criterion.to(device)

    # Run evaluations
    print("=== GENERATION EVALUATION ===")
    evaluate_generations(model, vocab, device)

    print("=== PERPLEXITY EVALUATION ===")
    evaluate_perplexity(model, val_dataloader, criterion, device, pad_id)

    return model, vocab, len(vocab), pad_id


model, vocab, vocab_size, pad_id = load_trained_model()