import os
import torch
from esm import FastaBatchedDataset, pretrained, MSATransformer
import argparse
from pathlib import Path


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract ESM embeddings from protein sequences"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default="/app/input/sequences.fasta",
        help="Input FASTA file path",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="/app/output",
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esm2_t36_3B_UR50D",
        help="ESM model to use (e.g., esm2_t36_3B_UR50D, esm2_t33_650M_UR50D, esm2_t30_150M_UR50D)",
    )
    parser.add_argument(
        "--toks_per_batch", type=int, default=4096, help="Maximum batch size in tokens"
    )
    parser.add_argument(
        "--repr_layers",
        type=int,
        nargs="+",
        help="Layer indices from which to extract representations (defaults to last layer)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        default=["per_tok"],
        choices=["mean", "per_tok", "bos"],
        help="Types of representations to return",
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="Truncate sequences longer than this length",
    )
    parser.add_argument(
        "--nogpu", action="store_true", help="Do not use GPU even if available"
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Load model
    print(f"Loading ESM model: {args.model}")
    try:
        model, alphabet = pretrained.load_model_and_alphabet(args.model)
        model.eval()

        if isinstance(model, MSATransformer):
            raise ValueError("This script does not handle MSA Transformer models")

        # Set default repr_layers to the last layer if not specified
        if args.repr_layers is None:
            args.repr_layers = [model.num_layers]
            print(f"Using default representation layer: {args.repr_layers[0]}")
    except Exception as e:
        print(f"Error loading model {args.model}: {str(e)}")
        print("Available ESM2 models include:")
        print("  - esm2_t36_3B_UR50D")
        print("  - esm2_t33_650M_UR50D")
        print("  - esm2_t30_150M_UR50D")
        print("  - esm2_t12_35M_UR50D")
        print("  - esm2_t6_8M_UR50D")
        return

    # Setup device
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    # Prepare dataset
    dataset = FastaBatchedDataset.from_file(args.input)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(args.truncation_seq_length),
        batch_sampler=batches,
    )
    print(f"Read {args.input} with {len(dataset)} sequences")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Adjust representation layers
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers
    ]

    # Process batches
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing batch {batch_idx + 1} of {len(batches)} ({toks.size(0)} sequences)"
            )

            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers)
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            # Save individual sequence representations
            for i, label in enumerate(labels):
                output_file = args.output_dir / f"{label}.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                result = {"label": label}
                truncate_len = min(args.truncation_seq_length, len(strs[i]))

                if "per_tok" in args.include:
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                if "mean" in args.include:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in args.include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }

                torch.save(result, output_file)

            # Print memory usage for monitoring
            if torch.cuda.is_available() and not args.nogpu:
                print(
                    f"GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
                )


if __name__ == "__main__":
    main()
