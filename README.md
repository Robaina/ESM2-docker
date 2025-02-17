# ESM2 Protein Embeddings Docker

A Docker container for efficiently extracting protein sequence embeddings using Facebook's ESM-2 (Evolutionary Scale Modeling) models. This implementation supports all ESM-2 models and provides flexible options for embedding extraction.

## Features

- Support for all ESM-2 models
- GPU acceleration with CUDA
- Flexible token-based batching for memory efficiency
- Multiple representation types (per-token, mean, BOS)
- Automated handling of long sequences with truncation
- Memory usage monitoring
- Individual embedding files for each sequence

## Available Models

| Model Name | Parameters | Memory Requirements* | Recommended Batch Size |
|------------|------------|---------------------|----------------------|
| esm2_t36_3B_UR50D | 3B | ~15GB | 2048-4096 |
| esm2_t33_650M_UR50D | 650M | ~4GB | 4096-8192 |
| esm2_t30_150M_UR50D | 150M | ~2GB | 8192-16384 |
| esm2_t12_35M_UR50D | 35M | ~1GB | 16384+ |
| esm2_t6_8M_UR50D | 8M | ~0.5GB | 16384+ |

*Memory requirements for 4096 tokens

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/esm2-embeddings-docker.git
cd esm2-embeddings-docker
```

2. Build the Docker image:
```bash
docker build -t esm2-embeddings .
```

## Usage

1. Create input and output directories:
```bash
mkdir -p input output
```

2. Place your FASTA file in the input directory:
```bash
cp your_sequences.fasta input/
```

3. Run the container:

Basic usage (default model: esm2_t36_3B_UR50D):
```bash
docker run --gpus all \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  esm2-embeddings \
  --input /app/input/your_sequences.fasta
```

With specific model and options:
```bash
docker run --gpus all \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  esm2-embeddings \
  --input /app/input/your_sequences.fasta \
  --model esm2_t33_650M_UR50D \
  --include per_tok mean \
  --toks_per_batch 4096
```

CPU-only mode:
```bash
docker run \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  esm2-embeddings \
  --input /app/input/your_sequences.fasta \
  --nogpu
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| --input | Input FASTA file path | /app/input/sequences.fasta |
| --output_dir | Output directory for embeddings | /app/output |
| --model | ESM model to use | esm2_t36_3B_UR50D |
| --toks_per_batch | Maximum batch size in tokens | 4096 |
| --repr_layers | Layer indices for representations | [last_layer] |
| --include | Types of representations to return | ["per_tok"] |
| --truncation_seq_length | Maximum sequence length | 1022 |
| --nogpu | Disable GPU usage | False |

## Output Format

The embeddings are saved as PyTorch tensor files (`.pt`) in the output directory, one file per sequence. Each file contains:

- `label`: Sequence identifier from FASTA
- `representations`: Per-token embeddings (if "per_tok" included)
- `mean_representations`: Mean sequence embeddings (if "mean" included)
- `bos_representations`: Beginning-of-sequence token embeddings (if "bos" included)

Load the embeddings in Python:
```python
import torch
embeddings = torch.load('output/sequence_name.pt')
```

## Memory Management

The `toks_per_batch` parameter controls memory usage and processing efficiency. Choose based on:
- Available GPU memory
- Model size
- Sequence lengths

Rule of thumb calculation:
```python
optimal_toks_per_batch = (max_sequence_length + 2) * desired_sequences_per_batch
```

## Requirements

- Docker
- NVIDIA GPU with CUDA support (for GPU acceleration)
- NVIDIA Container Toolkit (for GPU support)

## Citation

If you use this implementation, please cite the original [ESM-2 paper](https://www.science.org/doi/10.1126/science.ade2574)