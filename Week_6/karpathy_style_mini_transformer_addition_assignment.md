# Assignment: Building a Tiny Transformer That Learns Addition

## Overview

In this assignment you will implement a minimal GPT-style Transformer inspired by:

- The Tiny Transformer walkthrough notebook
- Andrej Karpathy's nanoGPT / miniGPT style training loops
- The `projects/adder` sequence modeling idea

Your model will learn to solve integer addition problems such as:

```text
12+35=47
```

and later generalize to unseen:

```text
384+217=601
```

The assignment is intentionally structured in milestones so that you build the system incrementally and debug each layer independently.

---

# Learning Goals

By the end of this assignment you should be able to:

1. Build a tokenizer and character vocabulary
2. Generate autoregressive training data
3. Implement causal self-attention from scratch
4. Assemble a Transformer block
5. Train a tiny GPT-like language model
6. Generate predictions autoregressively
7. Evaluate model quality using real mathematical checks

---

# Problem Setup

We will train a character-level Transformer.

Input examples:

```text
12+7=19
```

```text
384+217=601
```

The model learns one next-character prediction at a time.

Example training sequence:

```text
1 2 + 7 = 1 9
```

Targets are shifted by one token:

```text
2 + 7 = 1 9 <END>
```

This is exactly the same autoregressive training strategy used by GPT models.

---

# Starter Imports

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import math

# Reproducibility
random.seed(42)
torch.manual_seed(42)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
```

---

# Milestone 1 — Tokenizer & Dataset

## Goal

Build:

- vocabulary
- tokenizer
- dataset generator
- batching logic

---

## Step 1 — Generate Addition Problems

Complete the function.

```python
# Generate random addition problems
# Example:
# 12+35=47\n

def generate_example(max_digits=2):
    # TODO:
    # 1. Sample two random integers
    # 2. Compute the answer
    # 3. Return formatted string
    
    pass
```

---

## Step 2 — Build a Dataset String

Generate many training examples.

```python
# TODO:
# Create a long training string containing many examples

train_text = ""

for _ in range(50000):
    # TODO
    pass

print(train_text[:200])
```

---

## Step 3 — Build Vocabulary

Extract all unique characters.

Typical vocabulary:

```python
['\n', '+', '0', '1', '2', ..., '9', '=']
```

Complete the tokenizer.

```python
# TODO:
# 1. Find unique chars
# 2. Build stoi dictionary
# 3. Build itos dictionary

chars = None
vocab_size = None

stoi = {}
itos = {}


def encode(s):
    # TODO
    pass


def decode(tokens):
    # TODO
    pass
```

---

## Step 4 — Convert Dataset to Tokens

```python
# TODO:
# Convert the text dataset into integer tokens

data = None

print(data[:20])
```

---

## Step 5 — Create Batch Loader

The model learns next-token prediction.

Targets are shifted by one character.

```python
batch_size = 32
block_size = 32


def get_batch(split='train'):
    
    # TODO:
    # 1. Sample random starting indices
    # 2. Create X sequences
    # 3. Create Y shifted sequences
    
    return X.to(device), Y.to(device)
```

---

## Verification Cell

Run this before moving on.

```python
X, Y = get_batch()

print("X shape:", X.shape)
print("Y shape:", Y.shape)

print("Decoded sample:")
print(decode(X[0].tolist()))

print("Target sample:")
print(decode(Y[0].tolist()))
```

Expected behavior:

- `Y` is `X` shifted by one character
- decoded text looks like valid addition expressions

---

# Milestone 2 — Causal Multi-Head Attention

## Goal

Implement masked self-attention.

This is the core idea behind GPT.

Each token:

- looks at previous tokens
- cannot see future tokens

---

# Step 1 — Single Attention Head

Complete the module.

```python
class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        
        # TODO:
        # key projection
        # query projection
        # value projection
        
        # Register causal mask buffer
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        
        B, T, C = x.shape

        # TODO:
        # Compute keys
        # Compute queries

        # Attention scores
        weights = None

        # Scale attention scores
        weights = weights * (C ** -0.5)

        # IMPORTANT:
        # Apply causal mask
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        # Softmax
        weights = F.softmax(weights, dim=-1)

        # TODO:
        # Compute values
        # Return weighted aggregation

        return out
```

---

## Step 2 — Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()

        # TODO:
        # Create multiple attention heads
        
    def forward(self, x):
        
        # TODO:
        # Concatenate outputs
        
        return out
```

---

## Verification Cell

```python
x = torch.randn(4, block_size, 64)

head = Head(16)
out = head(x)

print(out.shape)
```

Expected output:

```python
torch.Size([4, block_size, 16])
```

---

# Milestone 3 — Transformer Block & Training Loop

## Goal

Build:

- Feed-forward network
- Layer normalization
- Residual connections
- Full Transformer block
- GPT model
- Training loop

---

# Step 1 — Feed Forward Network

```python
class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()

        # TODO:
        # Implement 2-layer MLP
        
    def forward(self, x):
        
        # TODO
        return x
```

---

# Step 2 — Transformer Block

Use:

- attention
- feed forward
- layer norm
- residual connections

```python
class Block(nn.Module):
    
    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        # TODO:
        # attention
        # feed forward
        # layer norms

    def forward(self, x):

        # TODO:
        # residual connection
        # residual connection

        return x
```

---

# Step 3 — Complete GPT Model

```python
class TinyGPT(nn.Module):
    
    def __init__(self):
        super().__init__()

        # Hyperparameters
        n_embd = 64
        n_head = 4
        n_layer = 2

        # TODO:
        # token embeddings
        # positional embeddings
        # transformer blocks
        # final layer norm
        # language model head

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # TODO:
        # token embeddings
        # positional embeddings
        # transformer blocks
        # logits

        loss = None

        if targets is not None:
            
            # TODO:
            # Cross entropy loss
            
            pass

        return logits, loss
```

---

# Step 4 — Create Model

```python
model = TinyGPT().to(device)

print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
```

---

# Step 5 — Training Loop

Train the model.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

max_iters = 3000

eval_interval = 200

for step in range(max_iters):

    # TODO:
    # get batch
    # forward pass
    # backward pass
    # optimizer step

    if step % eval_interval == 0:
        print(step, loss.item())
```

---

# Milestone 4 — Autoregressive Generation

## Goal

Generate text one token at a time.

The model should complete:

```text
12+35=
```

with:

```text
47
```

---

# Step 1 — Generation Function

```python
@torch.no_grad()
def generate(model, start_text, max_new_tokens=10, temperature=1.0):

    model.eval()

    # Encode input
    idx = torch.tensor([encode(start_text)], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):

        # Crop context window
        idx_cond = idx[:, -block_size:]

        # Forward pass
        logits, _ = model(idx_cond)

        # Take last token logits
        logits = logits[:, -1, :]

        # Apply temperature
        logits = logits / temperature

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # TODO:
        # Sample next token using torch.multinomial

        # TODO:
        # Append prediction to sequence

    return decode(idx[0].tolist())
```

---

# Step 2 — Test Generation

```python
print(generate(model, '12+35='))
print(generate(model, '99+18='))
print(generate(model, '384+217='))
```

---

# Final Evaluation Milestone

## Goal

Achieve:

- >=95% accuracy on unseen 2-digit problems
OR
- >=95% accuracy on unseen 3-digit problems

---

# Mathematical Evaluation

This evaluation checks REAL arithmetic correctness.

The model only passes if:

1. Output parses correctly
2. Prediction equals actual sum

---

# Student Task

Complete all TODO sections.

Then run the evaluation block below.

---

# Self-Grading Block

```python
import re


def extract_answer(text):
    
    # Extract digits after '='
    match = re.search(r'=([0-9]+)', text)

    if match:
        return match.group(1)

    return None


@torch.no_grad()
def evaluate_model(model, num_tests=200, max_digits=2):

    model.eval()

    correct = 0

    examples = []

    for _ in range(num_tests):

        a = random.randint(0, 10**max_digits - 1)
        b = random.randint(0, 10**max_digits - 1)

        prompt = f'{a}+{b}='

        generated = generate(
            model,
            prompt,
            max_new_tokens=max_digits + 3,
            temperature=0.1
        )

        predicted = extract_answer(generated)

        expected = str(a + b)

        is_correct = predicted == expected

        if is_correct:
            correct += 1

        examples.append({
            'prompt': prompt,
            'prediction': predicted,
            'expected': expected,
            'correct': is_correct
        })

    accuracy = correct / num_tests

    return accuracy, examples


# =====================================================
# SELF-GRADING CHECKLIST
# =====================================================

score = 0
max_score = 100

print('\n===============================')
print('AUTOGRADER RESULTS')
print('===============================\n')

# -----------------------------------------------------
# CHECK 1 — Tokenizer
# -----------------------------------------------------

try:
    encoded = encode('12+7=19')
    decoded = decode(encoded)

    if decoded == '12+7=19':
        print('[PASS] Tokenizer encode/decode works')
        score += 15
    else:
        print('[FAIL] Tokenizer mismatch')

except Exception as e:
    print('[FAIL] Tokenizer crashed:', e)


# -----------------------------------------------------
# CHECK 2 — Batch Generation
# -----------------------------------------------------

try:
    X, Y = get_batch()

    if X.shape == Y.shape:
        print('[PASS] Batch generation works')
        score += 15
    else:
        print('[FAIL] Batch shapes incorrect')

except Exception as e:
    print('[FAIL] Batch generation crashed:', e)


# -----------------------------------------------------
# CHECK 3 — Attention Layer
# -----------------------------------------------------

try:
    test_head = Head(16).to(device)

    x = torch.randn(4, block_size, 64).to(device)

    out = test_head(x)

    if out.shape[0] == 4:
        print('[PASS] Attention head runs')
        score += 20
    else:
        print('[FAIL] Attention output incorrect')

except Exception as e:
    print('[FAIL] Attention layer crashed:', e)


# -----------------------------------------------------
# CHECK 4 — Model Forward Pass
# -----------------------------------------------------

try:
    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    if logits.shape[0] == xb.shape[0]:
        print('[PASS] Model forward pass works')
        score += 20
    else:
        print('[FAIL] Model logits incorrect')

except Exception as e:
    print('[FAIL] Model forward pass crashed:', e)


# -----------------------------------------------------
# CHECK 5 — Arithmetic Accuracy
# -----------------------------------------------------

try:
    accuracy, examples = evaluate_model(
        model,
        num_tests=100,
        max_digits=2
    )

    print(f'\n2-digit accuracy: {accuracy:.2%}')

    if accuracy >= 0.95:
        print('[PASS] Achieved >=95% accuracy')
        score += 30
    else:
        print('[FAIL] Accuracy below 95%')

    print('\nExample predictions:\n')

    for ex in examples[:10]:
        print(ex)

except Exception as e:
    print('[FAIL] Evaluation crashed:', e)


# =====================================================
# FINAL SCORE
# =====================================================

print('\n===============================')
print(f'FINAL SCORE: {score}/{max_score}')
print('===============================\n')

if score == 100:
    print('Excellent work — your Tiny Transformer learned arithmetic.')
elif score >= 80:
    print('Good job — most Transformer components work correctly.')
else:
    print('Review the TODO sections and debug carefully.')
```

---

# Stretch Goals

If you finish early:

1. Add dropout
2. Add learning-rate scheduling
3. Train on subtraction
4. Add multiplication
5. Compare greedy decoding vs sampling
6. Visualize attention maps
7. Train on 4-digit addition
8. Add teacher forcing experiments
9. Implement KV caching
10. Compare parameter counts vs accuracy

---

# Suggested Hyperparameters

These are known-good starter values:

```python
batch_size = 32
block_size = 32
n_embd = 64
n_head = 4
n_layer = 2
learning_rate = 1e-3
max_iters = 3000
```

---

# Submission Requirements

Submit:

1. Completed notebook
2. Final autograder score
3. Screenshot of generated examples
4. Short explanation:
   - What was hardest?
   - What debugging strategy worked best?
   - What surprised you about Transformers?

---

# Important Conceptual Questions

You should be able to explain:

1. Why is the causal mask necessary?
2. Why do we shift targets by one token?
3. Why do we add positional embeddings?
4. Why do residual connections help?
5. Why does the model eventually learn carrying operations?
6. Why is addition a sequence modeling problem?
7. What happens if the block size is too small?

---

# Recommended Debugging Strategy

1. First verify tokenizer correctness
2. Then verify batches
3. Then verify attention dimensions
4. Then verify loss decreases
5. Finally test arithmetic correctness

Do not attempt to debug the full model all at once.

---

# Expected Results

Typical training progression:

```text
step 0     loss ~2.5
step 500   loss ~1.2
step 1500  loss ~0.4
step 3000  loss ~0.1
```

Final 2-digit accuracy:

```text
95% - 100%
```

---

# Final Reflection

This assignment demonstrates a profound idea:

A Transformer trained only on next-character prediction can learn an algorithmic process like arithmetic.

No calculator rules were explicitly programmed.

The model discovers the structure of addition from examples.

