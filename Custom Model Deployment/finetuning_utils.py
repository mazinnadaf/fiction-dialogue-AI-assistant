import os
import time
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset
import GPUtil
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from custom_model_architecture import *

class Batch:
    def __init__(self, tgt=None, loss_start_idxs=None, pad=2):  # 2 = <blank>
        if tgt is not None:
            self.tgt = tgt[:, :-1]  # target sentence (input to decoder) and the last token gets removed
            self.tgt_y = tgt[:, 1:]  # target sentence shifted by one token (output to predict)
            self.tgt_mask = self.make_std_mask(self.tgt, pad)

            # Create loss mask for SFT - only compute loss on response tokens
            if loss_start_idxs is not None:
                self.loss_mask = self.make_loss_mask(self.tgt_y, loss_start_idxs, pad)
                self.ntokens = self.loss_mask.sum().item()  # Only count tokens we compute loss on
            else:
                # Fallback for pretraining behavior
                self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

    @staticmethod
    def make_loss_mask(tgt_y, loss_start_idxs, pad):
        batch_size, seq_len = tgt_y.shape
        loss_mask = torch.zeros_like(tgt_y, dtype=torch.bool)

        for i, start_idx in enumerate(loss_start_idxs):
            # Adjust start_idx because tgt_y is shifted by 1 from original sequence
            adjusted_start = start_idx - 1
            if adjusted_start >= 0 and adjusted_start < seq_len:
                # Mask from adjusted start to end, but exclude padding
                loss_mask[i, adjusted_start:] = (tgt_y[i, adjusted_start:] != pad)

        return loss_mask

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

class SimpleLossCompute:
    def __init__(self, generator, criterion):
        self.generator = generator  # the model's final output layer
        self.criterion = criterion  # the loss function

    def __call__(self, x, y, norm, loss_mask=None):
        x = self.generator(x)
        log_probs = torch.log_softmax(x, dim=-1)

        if loss_mask is not None:
            # SFT mode: only compute loss on masked positions
            # Flatten everything for loss computation
            log_probs_flat = log_probs.contiguous().view(-1, log_probs.size(-1))
            y_flat = y.contiguous().view(-1)
            loss_mask_flat = loss_mask.contiguous().view(-1)

            # Only compute loss where mask is True
            if loss_mask_flat.sum() > 0:  # Avoid division by zero
                masked_log_probs = log_probs_flat[loss_mask_flat]
                masked_targets = y_flat[loss_mask_flat]

                sloss = self.criterion(masked_log_probs, masked_targets) / norm
            else:
                # Edge case: no tokens to compute loss on
                sloss = torch.tensor(0.0, device=x.device, requires_grad=True)
        else:
            # Pretraining mode: compute loss on all tokens (original behavior)
            sloss = (
                    self.criterion(
                        log_probs.contiguous().view(-1, log_probs.size(-1)),
                        y.contiguous().view(-1)
                    )
                    / norm
            )

        return sloss.data * norm, sloss

class TrainState:
    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
        data_iter,  # Iterable of Batch objects (training data loader)
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.tgt, batch.tgt_mask)

        # Check if we have loss_mask for SFT or use original behavior for pretraining
        if hasattr(batch, 'loss_mask') and batch.loss_mask is not None:
            # SFT mode: pass loss_mask to loss_compute
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens, batch.loss_mask)
        else:
            # Pretraining mode: original behavior
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train" or mode == "train+log":
            loss_node.backward()  # Backward pass to compute gradients with respect to all parameters

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            train_state.step += 1
            train_state.samples += batch.tgt.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()  # Applies the gradients to update model weights
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    avg_epoch_loss = total_loss / total_tokens
    print(f"Epoch complete. Average loss: {avg_epoch_loss:.4f}")
    return total_loss / total_tokens, train_state

# Adjustment of Transformer learning rate according to paper formula
def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def tokenize_pairs_to_ids(pairs, vocab, bos_token="<bos>", eos_token="<eos>"):
    input_ids_list = []
    loss_start_idxs = []

    total_tokens = 0
    unk_tokens = 0

    bos_id = vocab[bos_token]
    eos_id = vocab[eos_token]
    unk_id = vocab["<unk>"]

    for instruction, response in pairs:
        instr_tokens = instruction.strip().split()
        resp_tokens = response.strip().split()

        instr_ids = [vocab.get(tok, unk_id) for tok in instr_tokens]
        resp_ids = [vocab.get(tok, unk_id) for tok in resp_tokens]

        # Count tokens for statistics
        all_tokens = instr_tokens + resp_tokens
        total_tokens += len(all_tokens)
        unk_tokens += sum(1 for tok in all_tokens if vocab.get(tok, unk_id) == unk_id)

        input_ids = [bos_id] + instr_ids + [eos_id] + resp_ids + [eos_id]
        loss_start_idx = len(instr_ids) + 2

        input_ids_list.append(input_ids)
        loss_start_idxs.append(loss_start_idx)

    unk_rate = unk_tokens / total_tokens if total_tokens > 0 else 0
    print(f"Vocab coverage: {unk_rate:.1%} unknown tokens")

    return input_ids_list, loss_start_idxs

def collate_batch(batch, device, max_padding=128, pad_id=0):
    batch_list = []
    loss_start_idxs = []

    for item in batch:
        if isinstance(item, tuple):
            # SFT mode: batch contains (token_ids, loss_start_idx) tuples
            token_ids, loss_start_idx = item
            loss_start_idxs.append(loss_start_idx)
        else:
            # Pretraining mode: batch contains just token_ids
            token_ids = item

        token_tensor = torch.tensor(token_ids, dtype=torch.int64, device=device)
        padded = pad(
            token_tensor,
            (0, max_padding - len(token_tensor)),
            value=pad_id,
        )
        batch_list.append(padded)

    batch_tensor = torch.stack(batch_list)

    if loss_start_idxs:
        # Return both batch tensor and loss start indices for SFT
        return batch_tensor, loss_start_idxs
    else:
        # Return just batch tensor for pretraining compatibility
        return batch_tensor

def create_dataloaders(
    device,
    vocab,
    train_data,
    val_data,
    batch_size=32,
    max_padding=128,
    is_distributed=True,
):
    pad_id = vocab["<pad>"]

    def collate_fn(batch):
        return collate_batch(
            batch,
            device,
            max_padding=max_padding,
            pad_id=pad_id,
        )

    class TextDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    train_dataset = TextDataset(train_data)
    val_dataset = TextDataset(val_data)

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset) if is_distributed else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader

def train_worker(
    gpu,
    ngpus_per_node,
    vocab,
    pad_id,
    train_data,
    val_data,
    config,
    is_distributed=False,
):
    print(f"Train worker process using cpu for training", flush=True)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(gpu)
        print("Using CUDA GPU for training.")
    else:
        device = torch.device("cpu")
        print("CUDA and MPS not available. Using CPU.")

    pad_idx = pad_id
    vocab_size = len(vocab)
    print(vocab_size)

    d_model = 512

    model = make_model(vocab_size, N=12, d_model=512, d_ff=2048, h=8, dropout=0.1)

    print("Loading pretrained weights for SFT...")
    model.load_state_dict(torch.load("leaderboard_model_07.pt", map_location=device))
    print("Pretrained weights loaded in train_worker")

    model.to(device)
    print(f"Using device: {device}")
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=vocab_size, padding_idx=pad_idx, smoothing=0.1
    )
    criterion.to(device)

    train_dataloader, valid_dataloader = create_dataloaders(
        device,
        vocab,
        train_data,
        val_data,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    def create_batch_generator(dataloader):
        for batch_data in dataloader:
            if isinstance(batch_data, tuple):
                # SFT mode: batch_data is (batch_tensor, loss_start_idxs)
                batch_tensor, loss_start_idxs = batch_data
                yield Batch(batch_tensor, loss_start_idxs, pad_idx)
            else:
                # Pretraining mode: batch_data is just batch_tensor
                yield Batch(batch_data, None, pad_idx)

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            create_batch_generator(train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            create_batch_generator(valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)

def train_distributed_model(vocab, spacy_en, config, train, val):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab, spacy_en, train, val, config, True),
    )

def train_model(train, val, vocab, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab, spacy_en, config, train, val
        )
    else:
        train_worker(
            0, 1, vocab, spacy_en, train, val, config, False
        )