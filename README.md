# GPT Language Model Training

A PyTorch implementation of GPT (Generative Pre-trained Transformer) from scratch, optimized for training efficiency and loss reduction. This project implements a decoder-only transformer architecture similar to GPT-2, with comprehensive training optimizations.

## Features

- ✅ **Full GPT Architecture**: Complete implementation of GPT with causal self-attention, MLP blocks, and layer normalization
- ✅ **Optimized Training**: Gradient clipping, learning rate scheduling with warmup, and cosine annealing
- ✅ **Automatic Checkpointing**: Saves best model automatically when loss improves
- ✅ **Comprehensive Comments**: Well-documented code with step-by-step explanations
- ✅ **Flexible Configuration**: Easy-to-modify hyperparameters for different model sizes
- ✅ **Device Support**: Automatic device selection (CPU, CUDA, MPS for Apple Silicon)

## Requirements

- Python 3.7+
- PyTorch 1.9+
- tiktoken (for tokenization)
- transformers (optional, for loading pretrained weights)

## Installation

1. Clone or download this repository

2. Install required packages:

```bash
pip install torch transformers tiktoken
```

Or if you prefer using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install torch transformers tiktoken
```

## Project Structure

```
session12/
├── train_get2-8-init.py    # Main training script
├── input.txt                # Training data (text corpus)
├── checkpoints/             # Saved model checkpoints
│   ├── best_model.pt       # Best model (lowest loss)
│   └── final_model.pt      # Final model after training
└── README.md               # This file
```

## Usage

### Training

Simply run the training script:

```bash
python train_get2-8-init.py
```

The script will:
1. Load and tokenize the text from `input.txt`
2. Initialize a GPT model with default configuration (GPT-2 small: 124M parameters)
3. Train the model with optimized hyperparameters
4. Automatically save the best model checkpoint when loss improves
5. Save the final model at the end of training

### Training Configuration

You can modify these hyperparameters in the script:

```python
BATCH_SIZE = 16          # Batch size (number of sequences per batch)
SEQ_LENGTH = 128         # Sequence length (context window)
LEARNING_RATE = 6e-4     # Initial learning rate
WEIGHT_DECAY = 0.1       # Weight decay for regularization
MAX_STEPS = 5000         # Maximum training steps
WARMUP_STEPS = 100       # Learning rate warmup steps
GRAD_CLIP = 1.0          # Gradient clipping threshold
```

### Loading Saved Models

To load a saved checkpoint:

```python
from train_get2-8-init import load_checkpoint, GPT

# Load the best model
model, checkpoint = load_checkpoint('checkpoints/best_model.pt', device='cpu')

# Access checkpoint information
print(f"Step: {checkpoint['step']}")
print(f"Loss: {checkpoint['loss']:.4f}")
print(f"Best Loss: {checkpoint['best_loss']:.4f}")
print(f"Learning Rate: {checkpoint['learning_rate']}")

# Use the model for inference
model.eval()
# ... your inference code here
```

## Model Architecture

The model implements a GPT-style decoder-only transformer:

### Components

1. **CausalSelfAttention**: Multi-head self-attention with causal masking
   - Prevents tokens from attending to future tokens
   - Implements scaled dot-product attention
   - Supports multiple attention heads

2. **MLP**: Feed-forward network
   - 2-layer MLP with GELU activation
   - 4x expansion ratio (n_embd → 4*n_embd → n_embd)

3. **Block**: Transformer block
   - Pre-norm architecture (LayerNorm before attention/MLP)
   - Residual connections around attention and MLP

4. **GPT**: Main model
   - Token embeddings (vocab_size × n_embd)
   - Position embeddings (block_size × n_embd)
   - Stack of transformer blocks
   - Language modeling head

### Default Configuration (GPT-2 Small)

```python
block_size: 1024    # Maximum sequence length
vocab_size: 50257   # Vocabulary size
n_layer: 12         # Number of transformer blocks
n_head: 12          # Number of attention heads
n_embd: 768         # Embedding dimension
```

Total parameters: ~124M

## Training Optimizations

The training script includes several optimizations to reduce loss:

1. **Gradient Clipping**: Prevents exploding gradients (threshold: 1.0)
2. **Learning Rate Scheduling**: 
   - Warmup phase: Linear increase for first 100 steps
   - Cosine annealing: Smooth decay after warmup
3. **Weight Decay**: L2 regularization (0.1)
4. **Optimized Batch Size**: Increased to 16 for better gradient estimates
5. **Longer Sequences**: Sequence length of 128 for better context understanding
6. **Scaled Weight Initialization**: Proper initialization for residual connections

## Training Output

During training, you'll see output like:

```
Starting training with batch_size=16, seq_length=128
Total steps: 5000, Warmup steps: 100
Checkpoints will be saved to: checkpoints/
------------------------------------------------------------
✓ Saved best model checkpoint at step 0 with loss 10.5234
step     0 | lr: 6.00e-06 | loss: 10.5234 | best: 10.5234 (step 0)
step   100 | lr: 6.00e-04 | loss: 8.2341 | best: 7.8912 (step 87)
step   200 | lr: 5.98e-04 | loss: 6.5432 | best: 6.1234 (step 198)
...
------------------------------------------------------------
Training completed!
Final loss: 0.3453
Best loss: 0.1716 (achieved at step 4517)
Average loss (last 100 steps): 0.3894
Best model saved to: data/checkpoints/best_model.pt
Final model saved to: data/checkpoints/final_model.pt
```

## Loading Pretrained Weights

The code includes a method to load pretrained GPT-2 weights from HuggingFace:

```python
# Uncomment this line in the script to load pretrained weights
model = GPT.from_pretrained('gpt2')  # Options: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
```

## Text Generation

The script includes text generation code (currently commented out due to early exit). To use it:

1. Remove or comment out the `sys.exit(0)` line
2. The generation code uses top-k sampling with k=50
3. Generates multiple sequences in parallel

## Notes

- The model uses weight sharing between token embeddings and the output projection layer
- Weight initialization follows the nanoGPT scheme with proper scaling for residual connections
- The data loader implements sequential batching with automatic wrapping

## Troubleshooting

**Out of Memory Error**: 
- Reduce `BATCH_SIZE` or `SEQ_LENGTH`
- Use a smaller model configuration

**Slow Training**:
- Ensure you're using GPU (CUDA or MPS) if available
- Reduce `MAX_STEPS` for faster iteration

**High Loss**:
- Increase training steps
- Adjust learning rate
- Check data quality in `input.txt`
