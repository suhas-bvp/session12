# Solving for residual std scaling issue
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention mechanism for GPT.
    Implements multi-head attention with causal masking to ensure tokens can only attend to previous tokens.
    """

    def __init__(self, config):
        """
        Initialize the causal self-attention layer.
        
        Args:
            config: GPTConfig object containing model hyperparameters
        """
        super().__init__()
        # Ensure embedding dimension is divisible by number of heads
        assert config.n_embd % config.n_head == 0
        
        # Single linear layer that computes Q, K, V for all heads simultaneously
        # Output size is 3 * n_embd (for Q, K, V concatenated)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # Output projection layer to combine attention outputs
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Flag for special weight initialization (scaled by layer depth)
        self.c_proj.NANGPT_SCALE_INIT = 1
        
        # Store hyperparameters
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Create causal mask: lower triangular matrix to prevent attending to future tokens
        # This is registered as a buffer (not a parameter, so it won't be updated during training)
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", causal_mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        Forward pass through causal self-attention.
        
        Args:
            x: Input tensor of shape (B, T, C) where B=batch, T=sequence length, C=embedding dim
            
        Returns:
            Output tensor of shape (B, T, C) after attention and projection
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # Step 1: Compute Q, K, V for all heads in one go
        # c_attn outputs (B, T, 3*C), we split it into Q, K, V each of size (B, T, C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Step 2: Reshape and transpose for multi-head attention
        # Split embedding dimension across heads: C = n_head * head_size
        # Reshape to (B, T, n_head, head_size) then transpose to (B, n_head, T, head_size)
        # This allows parallel computation across all heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Step 3: Compute attention scores
        # Q @ K^T gives (B, nh, T, T) attention scores
        # Scale by sqrt(head_size) to prevent large values before softmax
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Step 4: Apply causal mask (set future positions to -inf so they become 0 after softmax)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Step 5: Apply softmax to get attention probabilities
        att = F.softmax(att, dim=-1)
        
        # Step 6: Apply attention weights to values
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v

        # Step 7: Concatenate all heads back together
        # Transpose back to (B, T, nh, hs) then reshape to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Step 8: Apply output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (feed-forward network) used in transformer blocks.
    Implements a 2-layer MLP with GELU activation and 4x expansion ratio.
    """

    def __init__(self, config):
        """
        Initialize the MLP layer.
        
        Args:
            config: GPTConfig object containing model hyperparameters
        """
        super().__init__()
        # First linear layer: expands from n_embd to 4*n_embd (expansion ratio of 4)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        
        # GELU activation function (Gaussian Error Linear Unit)
        # Uses tanh approximation for efficiency
        self.gelu = nn.GELU(approximate='tanh')
        
        # Second linear layer: projects back from 4*n_embd to n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # Flag for special weight initialization (scaled by layer depth)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (B, T, C)
            
        Returns:
            Output tensor of shape (B, T, C) after MLP transformation
        """
        # Step 1: Expand dimension (B, T, C) -> (B, T, 4*C)
        x = self.c_fc(x)
        
        # Step 2: Apply GELU activation
        x = self.gelu(x)
        
        # Step 3: Project back to original dimension (B, T, 4*C) -> (B, T, C)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """
    Transformer block containing self-attention and MLP with residual connections.
    Implements pre-norm architecture: LayerNorm before attention/MLP, then residual connection.
    """

    def __init__(self, config):
        """
        Initialize the transformer block.
        
        Args:
            config: GPTConfig object containing model hyperparameters
        """
        super().__init__()
        # Layer normalization before attention (pre-norm architecture)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        
        # Causal self-attention layer
        self.attn = CausalSelfAttention(config)
        
        # Layer normalization before MLP
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # Feed-forward MLP
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (B, T, C)
            
        Returns:
            Output tensor of shape (B, T, C) after attention and MLP with residuals
        """
        # Pre-norm attention with residual connection
        # Apply LayerNorm, then attention, then add residual
        x = x + self.attn(self.ln_1(x))
        
        # Pre-norm MLP with residual connection
        # Apply LayerNorm, then MLP, then add residual
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """
    Configuration dataclass for GPT model hyperparameters.
    Default values match GPT-2 small model (124M parameters).
    """
    block_size: int = 1024  # Maximum sequence length (context window)
    vocab_size: int = 50257  # Vocabulary size: 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int = 12  # Number of transformer blocks (layers)
    n_head: int = 12  # Number of attention heads
    n_embd: int = 768  # Embedding dimension (hidden size)


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) model.
    Implements a decoder-only transformer architecture for language modeling.
    """

    def __init__(self, config):
        """
        Initialize the GPT model.
        
        Args:
            config: GPTConfig object containing model hyperparameters
        """
        super().__init__()
        self.config = config

        # Build transformer architecture
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Stack of transformer blocks
            ln_f = nn.LayerNorm(config.n_embd),  # Final layer normalization
        ))
        
        # Language modeling head: projects embeddings to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing: share weights between token embedding and output projection
        # This reduces parameters and improves training stability
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights using custom initialization scheme
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Custom weight initialization for GPT model.
        Uses scaled initialization for residual connections to maintain variance.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            # Base standard deviation for weight initialization
            std = 0.02
            
            # For residual projection layers (marked with NANGPT_SCALE_INIT),
            # scale down initialization by sqrt(2*n_layer) to prevent variance explosion
            # This is important for deep networks with residual connections
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            
            # Initialize weights from normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
            # Initialize biases to zero if they exist
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights from normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, idx, targets=None):
        """
        Forward pass through the GPT model.
        
        Args:
            idx: Input token indices of shape (B, T) where B=batch size, T=sequence length
            targets: Optional target token indices of shape (B, T) for computing loss
            
        Returns:
            logits: Model predictions of shape (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        # Get batch size and sequence length
        B, T = idx.size()
        
        # Ensure sequence doesn't exceed maximum block size
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Step 1: Create position indices [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        
        # Step 2: Get position embeddings of shape (T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        
        # Step 3: Get token embeddings of shape (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        
        # Step 4: Combine token and position embeddings (add them together)
        x = tok_emb + pos_emb
        
        # Step 5: Forward through all transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Step 6: Apply final layer normalization
        x = self.transformer.ln_f(x)
        
        # Step 7: Project to vocabulary logits of shape (B, T, vocab_size)
        logits = self.lm_head(x)
        
        # Step 8: Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape logits and targets to (B*T, vocab_size) and (B*T,) for cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load pretrained GPT-2 model weights from HuggingFace.
        Converts HuggingFace's GPT-2 weights to match our model architecture.
        
        Args:
            model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            
        Returns:
            GPT model with pretrained weights loaded
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # Map model type to architecture hyperparameters
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),   # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M params
        }[model_type]
        
        # GPT-2 models always use these fixed values
        config_args['vocab_size'] = 50257  # Always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024   # Always 1024 for GPT model checkpoints
        
        # Create a fresh GPT model with the correct architecture
        config = GPTConfig(**config_args)
        model = GPT(config)
        
        # Get state dict from our model (keys we need to populate)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Remove attention bias (it's a buffer, not a parameter)
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Load pretrained model from HuggingFace
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Filter HuggingFace state dict keys (remove buffers we don't need)
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # Ignore buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # Ignore mask buffer
        
        # These layers use Conv1D in HuggingFace but Linear in our model, so weights need transposing
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # Verify we have matching number of parameters
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        # Copy weights from HuggingFace model to our model
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special case: transpose Conv1D weights to Linear weights
                # HuggingFace uses Conv1D which stores weights as (out_features, in_features)
                # Our Linear layers expect (out_features, in_features) but need transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())  # Transpose when copying
            else:
                # Standard case: direct copy (shapes should match)
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# model = GPT.from_pretrained('gpt2')

# Device selection: automatically choose the best available device
# Priority: CUDA (GPU) > MPS (Apple Silicon) > CPU
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'  # NVIDIA GPU
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU (Metal Performance Shaders)
print(f"using device: {device}")

# Set random seed for reproducibility
# This ensures consistent results across runs
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Generation parameters (for text generation, currently unused due to early exit)
num_return_sequences = 5  # Number of sequences to generate
max_length = 30  # Maximum length of generated sequences

import tiktoken

class DataLoaderLite:
    """
    Lightweight data loader for GPT training.
    Loads text data, tokenizes it, and provides batches for training.
    """
    
    def __init__(self, B, T):
        """
        Initialize the data loader.
        
        Args:
            B: Batch size (number of sequences per batch)
            T: Sequence length (context window size)
        """
        self.B = B
        self.T = T

        # Step 1: Load text data from file into memory
        with open('input.txt', 'r') as f:
            text = f.read()
        
        # Step 2: Initialize GPT-2 tokenizer
        enc = tiktoken.get_encoding('gpt2')
        
        # Step 3: Tokenize the entire text corpus
        tokens = enc.encode(text)
        
        # Step 4: Convert to PyTorch tensor for efficient batching
        self.tokens = torch.tensor(tokens)
        
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # Track current position in the token sequence for sequential batching
        self.current_position = 0
    
    def next_batch(self):
        """
        Get the next batch of training data.
        Creates input-target pairs where targets are inputs shifted by one position.
        
        Returns:
            x: Input tokens of shape (B, T)
            y: Target tokens of shape (B, T) - same as x but shifted by 1 position
        """
        B, T = self.B, self.T
        
        # Extract a buffer of B*T+1 tokens (need +1 for target shift)
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        
        # Create input-target pairs:
        # x = tokens [0, 1, 2, ..., B*T-1]
        # y = tokens [1, 2, 3, ..., B*T] (shifted by 1 for next-token prediction)
        x = (buf[:-1]).view(B, T)  # inputs: shape (B, T)
        y = (buf[1:]).view(B, T)   # targets: shape (B, T)
        
        # Advance position for next batch
        self.current_position += B * T
        
        # Reset to beginning if we've reached the end of the data
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        
        return x, y


# Initialize GPT model with default configuration (GPT-2 small architecture)
model = GPT(GPTConfig())
model.to(device)  # Move model to selected device (CPU/GPU)

# Optimized training hyperparameters
# Increased batch size and sequence length for better gradient estimates and longer context
BATCH_SIZE = 16  # Increased from 4 for better gradient estimates
SEQ_LENGTH = 128  # Increased from 32 for longer context understanding
LEARNING_RATE = 6e-4  # Slightly higher initial LR for faster convergence
WEIGHT_DECAY = 0.1  # Weight decay for regularization
MAX_STEPS = 5000  # Increased from 50 for better convergence
WARMUP_STEPS = 100  # Warmup steps for learning rate scheduling
GRAD_CLIP = 1.0  # Gradient clipping threshold to prevent exploding gradients

# Initialize data loader with optimized batch size and sequence length
train_loader = DataLoaderLite(B=BATCH_SIZE, T=SEQ_LENGTH)

# Initialize optimizer: AdamW with optimized hyperparameters
# betas=(0.9, 0.95) are standard for transformer training
# weight_decay helps with regularization
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    weight_decay=WEIGHT_DECAY,
    eps=1e-8
)

# Learning rate scheduler with warmup and cosine annealing
def get_lr(it):
    # Warmup phase: linearly increase LR from 0 to LEARNING_RATE
    if it < WARMUP_STEPS:
        return LEARNING_RATE * (it + 1) / WARMUP_STEPS
    # Cosine annealing: decay LR following cosine curve
    progress = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))

# Training loop with optimizations
print(f"Starting training with batch_size={BATCH_SIZE}, seq_length={SEQ_LENGTH}")
print(f"Total steps: {MAX_STEPS}, Warmup steps: {WARMUP_STEPS}")
print("-" * 60)

best_loss = float('inf')
losses = []

for step in range(MAX_STEPS):
    # Step 1: Get next batch of input-target pairs
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)  # Move data to device
    
    # Step 2: Update learning rate with scheduler
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Step 3: Zero out gradients from previous iteration
    optimizer.zero_grad()
    
    # Step 4: Forward pass: compute logits and loss
    logits, loss = model(x, y)
    
    # Step 5: Backward pass: compute gradients
    loss.backward()
    
    # Step 6: Gradient clipping to prevent exploding gradients
    # This is crucial for training stability in transformers
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    
    # Step 7: Update model parameters using computed gradients
    optimizer.step()
    
    # Step 8: Track loss
    loss_val = loss.item()
    losses.append(loss_val)
    if loss_val < best_loss:
        best_loss = loss_val
    
    # Step 9: Print progress periodically
    if step % 100 == 0 or step < 10:
        print(f'step {step:5d} | lr: {lr:.2e} | loss: {loss_val:.4f} | best: {best_loss:.4f}')
    
    # Early stopping if loss becomes very small (optional)
    if loss_val < 0.01:
        print(f"Loss converged to {loss_val:.4f} at step {step}")
        break

# Print training summary
print("-" * 60)
print(f"Training completed!")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Best loss: {best_loss:.4f}")
print(f"Average loss (last 100 steps): {sum(losses[-100:])/len(losses[-100:]):.4f}")

import sys; sys.exit(0)  # Exit early (generation code below is not executed)

# Text generation code (currently not executed due to early exit above)
# This implements top-k sampling for generating text

torch.manual_seed(42)  # Set seed for reproducible generation
torch.cuda.manual_seed(42)

# Generate tokens until we reach max_length
while x.size(1) < max_length:
    # Forward pass: get logits for next token prediction
    # Use no_grad() to disable gradient computation (faster, uses less memory)
    with torch.no_grad():
        logits = model(x)[0]  # (B, T, vocab_size) - logits for all positions
        
        # Step 1: Extract logits at the last position (we only need the next token)
        logits = logits[:, -1, :]  # (B, vocab_size)
        
        # Step 2: Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=-1)
        
        # Step 3: Top-k sampling: keep only top 50 most likely tokens
        # This reduces the search space and improves generation quality
        # topk_probs: (B, 50) - probabilities of top-k tokens
        # topk_indices: (B, 50) - indices of top-k tokens
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        
        # Step 4: Sample one token from the top-k probabilities
        # multinomial samples according to the probability distribution
        # Note: multinomial doesn't require probabilities to sum to 1
        ix = torch.multinomial(topk_probs, 1)  # (B, 1) - index into top-k list
        
        # Step 5: Get the actual token index from topk_indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1) - actual token index
        
        # Step 6: Append the new token to the sequence
        x = torch.cat((x, xcol), dim=1)

# Decode and print generated text for each sequence in the batch
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()  # Convert tensor to list
    decoded = enc.decode(tokens)  # Decode tokens back to text
    print(">", decoded)