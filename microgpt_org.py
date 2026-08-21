"""
PyTorch version of microgpt_for_mod.py.
Keeps the same modulo-addition task, model shape, training loop, and evaluation
flow, but delegates tensors and autograd to PyTorch.
"""

import os       # os.path.exists
import random   # random.seed, random.shuffle

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ModuleNotFoundError("PyTorch is required to run this file. Install torch first.") from None

random.seed(42) # Let there be order among chaos
torch.manual_seed(42)

# Modulo addition task setting
MODULUS = 87
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, 'example1.txt')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
NUMBER_VECTOR_PATH = os.path.join(OUTPUT_DIR, 'number_vectors.txt')
EFFECTIVE_NUMBER_VECTOR_PATH = os.path.join(OUTPUT_DIR, 'number_vectors_effective.txt')
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, 'embedding_snapshots')
if MODULUS < 2:
    raise ValueError("MODULUS must be at least 2")

# Let there be a Dataset `docs`: list[str] of modulo addition examples, e.g. "4 + 14 = 18"
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"expected {INPUT_PATH} with lines like '4 + 14 = 18'")
docs = [line.strip() for line in open(INPUT_PATH) if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

def parse_example(doc):
    parts = doc.split()
    if len(parts) != 5 or parts[1] != '+' or parts[3] != '=':
        raise ValueError(f"bad example format: {doc!r}")
    a, b, target = int(parts[0]), int(parts[2]), int(parts[4])
    if not (0 <= a < MODULUS and 0 <= b < MODULUS and 0 <= target < MODULUS):
        raise ValueError(f"example out of range for MODULUS={MODULUS}: {doc!r}")
    expected = (a + b) % MODULUS
    if target != expected:
        raise ValueError(f"wrong target for MODULUS={MODULUS}: {doc!r}, expected {expected}")
    return a, b, target

examples = [parse_example(doc) for doc in docs]

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
utokens = [str(i) for i in range(MODULUS)] # number tokens become token ids 0..n-1
stoi = {token: i for i, token in enumerate(utokens)}
itos = {i: token for token, i in stoi.items()}
vocab_size = len(utokens) # total number of unique tokens
print(f"vocab size: {vocab_size}")
print(utokens)

def encode_prompt(a, b):
    return [stoi[str(a)], stoi[str(b)]]

# Initialize the model hyperparameters
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 64     # width of the network (embedding dimension)
block_size = 2  # maximum context length of the attention window: a b
n_head = 1      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head

class RMSNorm(nn.Module):
    def forward(self, x):
        ms = (x * x).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + 1e-5)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_wq = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wk = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wv = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wo = nn.Linear(n_embd, n_embd, bias=False)
        self.mlp_fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.mlp_fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.rmsnorm = RMSNorm()

    def forward(self, x):
        bsz, seq_len, _ = x.shape

        # 1) Multi-head Attention block
        x_residual = x
        x = self.rmsnorm(x)
        q = self.attn_wq(x)
        k = self.attn_wk(x)
        v = self.attn_wv(x)

        q = q.view(bsz, seq_len, n_head, head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, n_head, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, n_head, head_dim).transpose(1, 2)

        attn_logits = q @ k.transpose(-2, -1) / (head_dim ** 0.5)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        attn_logits = attn_logits.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)
        x_attn = attn_weights @ v
        x_attn = x_attn.transpose(1, 2).contiguous().view(bsz, seq_len, n_embd)
        x = self.attn_wo(x_attn)
        x = x + x_residual

        # 2) MLP block
        x_residual = x
        x = self.rmsnorm(x)
        x = self.mlp_fc1(x)
        x = F.relu(x)
        x = self.mlp_fc2(x)
        x = x + x_residual
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.rmsnorm = RMSNorm()
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, token_ids):
        bsz, seq_len = token_ids.shape
        if seq_len > block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {block_size}")
        pos_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.wte(token_ids) + self.wpe(pos_ids)
        x = self.rmsnorm(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)
        return logits

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
model = TinyGPT().to(device)
params = list(model.parameters())
print(f"num params: {sum(p.numel() for p in params)}")
print(f"device: {device}")

@torch.no_grad()
def predict(a, b):
    was_training = model.training
    model.eval()
    prompt_tokens = torch.tensor([encode_prompt(a, b)], dtype=torch.long, device=device)
    logits = model(prompt_tokens)
    pred_id = int(torch.argmax(logits[0, -1, :]).item())
    if was_training:
        model.train()
    return int(itos[pred_id])

def evaluate_accuracy():
    correct = 0
    total = 0
    for a in range(MODULUS):
        for b in range(MODULUS):
            pred = predict(a, b)
            target = (a + b) % MODULUS
            correct += pred == target
            total += 1
    return correct, total, correct / total

# Split evaluation: train pairs (seen during training) vs held-out pairs (never seen).
# Grokking is only measurable on the held-out split — mixing them hides the delayed jump.
train_eval_pairs = sorted({(a, b) for a, b, _ in examples})
val_eval_pairs = [(a, b) for a in range(MODULUS) for b in range(MODULUS)
                  if (a, b) not in set(train_eval_pairs)]
if not val_eval_pairs:
    raise ValueError("training file covers the full input space; no held-out pairs to measure generalization")
print(f"train pairs: {len(train_eval_pairs)} | held-out pairs: {len(val_eval_pairs)}")

def pairs_to_tensors(pairs):
    inputs = torch.tensor([encode_prompt(a, b) for a, b in pairs], dtype=torch.long, device=device)
    targets = torch.tensor([stoi[str((a + b) % MODULUS)] for a, b in pairs], dtype=torch.long, device=device)
    return inputs, targets

train_eval_inputs, train_eval_targets = pairs_to_tensors(train_eval_pairs)
val_eval_inputs, val_eval_targets = pairs_to_tensors(val_eval_pairs)

@torch.no_grad()
def evaluate_split(inputs, targets):
    was_training = model.training
    model.eval()
    logits = model(inputs)[:, -1, :]
    loss = F.cross_entropy(logits, targets).item()
    preds = torch.argmax(logits, dim=-1)
    correct = int((preds == targets).sum().item())
    total = int(targets.numel())
    if was_training:
        model.train()
    return loss, correct, total, correct / total

def write_vectors(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for label, values in rows:
            values = ' '.join(f"{value:.8f}" for value in values)
            f.write(f"{label}\t{values}\n")

@torch.no_grad()
def save_number_vectors(raw_path, effective_path):
    raw_vectors = model.wte.weight.detach().cpu()
    raw_rows = []
    for number in range(MODULUS):
        token_id = stoi[str(number)]
        raw_rows.append((number, raw_vectors[token_id].tolist()))
    write_vectors(raw_path, raw_rows)

    token_ids = torch.arange(MODULUS, dtype=torch.long, device=device)
    pos0_ids = torch.zeros(MODULUS, dtype=torch.long, device=device)
    pos1_ids = torch.ones(MODULUS, dtype=torch.long, device=device)
    raw = model.wte(token_ids)
    pos0_vectors = model.rmsnorm(raw + model.wpe(pos0_ids)).detach().cpu()
    pos1_vectors = model.rmsnorm(raw + model.wpe(pos1_ids)).detach().cpu()
    effective_rows = []
    for number in range(MODULUS):
        effective_rows.append((f"{number}\tpos0", pos0_vectors[number].tolist()))
        effective_rows.append((f"{number}\tpos1", pos1_vectors[number].tolist()))
    write_vectors(effective_path, effective_rows)

@torch.no_grad()
def save_embedding_snapshot(step):
    # One file per training stage: raw wte rows, used for the disorder -> circle animation
    raw_vectors = model.wte.weight.detach().cpu()
    rows = [(number, raw_vectors[stoi[str(number)]].tolist()) for number in range(MODULUS)]
    write_vectors(os.path.join(SNAPSHOT_DIR, f"step_{step:06d}.txt"), rows)

# Let there be AdamW: strong weight decay is the key driver of grokking
# (Power et al. 2022 / Nanda et al. 2023 use AdamW with weight decay λ=1)
learning_rate, beta1, beta2, eps_adam = 0.001, 0.85, 0.99, 1e-8
weight_decay = 1.0
optimizer = torch.optim.AdamW(params, lr=learning_rate, betas=(beta1, beta2), eps=eps_adam,
                              weight_decay=weight_decay)

# Repeat in sequence
num_steps = 20000 # number of training steps
log_every = 50
batch_size = len(examples)
example_order = list(range(len(examples)))
batch_cursor = 0

# Metrics CSV: every run self-records its curves (used for the slide animation)
METRICS_LOG_PATH = os.path.join(OUTPUT_DIR, 'train_log.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)
metrics_log = open(METRICS_LOG_PATH, 'w')
metrics_log.write('step,train_loss,train_acc,val_loss,val_acc\n')

# Embedding snapshots at a fixed cadence (+ init and final) so the
# memorization -> grokking transition is captured wherever it happens
snapshot_every = 250
save_embedding_snapshot(0)
for step in range(num_steps):

    if batch_cursor == 0:
        random.shuffle(example_order)
    batch_indices = example_order[batch_cursor:batch_cursor + batch_size]
    if len(batch_indices) < batch_size:
        random.shuffle(example_order)
        batch_indices += example_order[:batch_size - len(batch_indices)]
    batch_cursor = (batch_cursor + batch_size) % len(examples)

    # Feed batches of "a b" and train only the next-token prediction for the answer.
    batch = [examples[i] for i in batch_indices]
    prompt_tokens = torch.tensor([encode_prompt(a, b) for a, b, _ in batch], dtype=torch.long, device=device)
    target_ids = torch.tensor([stoi[str(target)] for _, _, target in batch], dtype=torch.long, device=device)

    logits = model(prompt_tokens)
    loss = F.cross_entropy(logits[:, -1, :], target_ids)

    # Backward the loss, calculating the gradients with respect to all model parameters
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # AdamW update with a constant learning rate: grokking happens late in training,
    # so the LR must not decay to zero before the transition
    optimizer.step()

    if (step + 1) % log_every == 0 or step == 0:
        train_loss, _, _, train_acc = evaluate_split(train_eval_inputs, train_eval_targets)
        val_loss, _, _, val_acc = evaluate_split(val_eval_inputs, val_eval_targets)
        print(f"step {step+1:6d} / {num_steps:6d} | train loss {train_loss:.4f} | train acc {train_acc:.3f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}")
        metrics_log.write(f"{step+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f}\n")
        metrics_log.flush()

    if (step + 1) % snapshot_every == 0:
        save_embedding_snapshot(step + 1)

save_embedding_snapshot(num_steps)
metrics_log.close()
print(f"saved metrics log to {METRICS_LOG_PATH}")
print(f"saved embedding snapshots to {SNAPSHOT_DIR}")

# Inference: evaluate train and held-out splits separately
print("\n--- evaluation ---")
train_loss, tc, tt, train_acc = evaluate_split(train_eval_inputs, train_eval_targets)
val_loss, vc, vt, val_acc = evaluate_split(val_eval_inputs, val_eval_targets)
print(f"train:    loss {train_loss:.4f} | accuracy {tc}/{tt} = {train_acc:.3f}")
print(f"held-out: loss {val_loss:.4f} | accuracy {vc}/{vt} = {val_acc:.3f}")
print("--- samples ---")
sample_pairs = [
    (0, 0),
    (1 % MODULUS, (MODULUS // 3) % MODULUS),
    (MODULUS // 4, (MODULUS // 2 + 3) % MODULUS),
    (MODULUS - 4, MODULUS - 5),
    (MODULUS - 1, MODULUS - 1),
]
for a, b in sample_pairs:
    print(f"{a} + {b} = {predict(a, b)} (target {(a + b) % MODULUS})")

save_number_vectors(NUMBER_VECTOR_PATH, EFFECTIVE_NUMBER_VECTOR_PATH)
print(f"saved raw number vectors to {NUMBER_VECTOR_PATH}")
print(f"saved effective number vectors to {EFFECTIVE_NUMBER_VECTOR_PATH}")
