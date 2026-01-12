#!/usr/bin/env python3
"""Test client for TX server long sequence support using tinker API."""

import argparse

import tinker
from tinker import types
from transformers import AutoTokenizer


BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def make_datum(tokenizer, seq_len: int = 256):
    """Helper to create a Datum with specified sequence length."""
    # Create a sequence of the desired length using repeated tokens
    all_tokens = list(range(1, seq_len + 1))  # Simple token sequence
    target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]

    # All weights = 1 for simplicity
    weights = [1.0] * seq_len

    return types.Datum(
        model_input=types.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "weights": weights,
        },
    )


def test_sample(sampling_client, tokenizer, num_samples: int = 4, max_tokens: int = 32):
    """Test sampling (inference)."""
    # Use half of max_tokens as prompt length to stress both prefill and decode
    prompt_len = max_tokens // 2
    gen_tokens = max_tokens - prompt_len
    print(f"\n=== Testing Sampling (n={num_samples}, prompt={prompt_len}, gen={gen_tokens}) ===")

    # Generate prompt by repeating tokens
    base_tokens = tokenizer.encode("Hello, how are you doing today? ", add_special_tokens=True)
    repeated = (base_tokens * ((prompt_len // len(base_tokens)) + 1))[:prompt_len]
    prompt = types.ModelInput.from_ints(repeated)

    request = sampling_client.sample(
        prompt=prompt,
        sampling_params=types.SamplingParams(temperature=0.7, max_tokens=gen_tokens, seed=42),
        num_samples=num_samples,
    )

    result = request.result()
    print(f"Sampling completed")
    print(f"Generated {len(result.sequences)} sequences")
    for i, seq in enumerate(result.sequences):
        print(f"  Sequence {i}: {len(seq.tokens)} tokens, stop_reason={seq.stop_reason}")
    return True


def test_forward_backward(training_client, tokenizer, batch_size: int = 4, seq_len: int = 256):
    """Test forward-backward (training)."""
    print(f"\n=== Testing Forward-Backward (batch={batch_size}, seq_len={seq_len}) ===")

    # Create training examples with specified sequence length
    data = [make_datum(tokenizer, seq_len=seq_len) for _ in range(batch_size)]

    fwdbwd_future = training_client.forward_backward(data, "cross_entropy")
    result = fwdbwd_future.result()

    print(f"Forward-backward completed")
    print(f"Loss outputs: {len(result.loss_fn_outputs)} items")
    return True


def test_optim_step(training_client, learning_rate: float = 1e-4):
    """Test optimizer step (apply accumulated gradients)."""
    print(f"\n=== Testing Optim Step (lr={learning_rate}) ===")

    adam_params = types.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.01,
    )

    optim_future = training_client.optim_step(adam_params)
    result = optim_future.result()

    print(f"Optim step completed")
    print(f"Result: {result}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test TX server long sequence support")
    parser.add_argument("--url", default="http://localhost:8001", help="Server URL")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Base model name")
    parser.add_argument("--sample-n", type=int, default=4, help="Number of samples")
    parser.add_argument("--sample-tokens", type=int, default=8192, help="Total tokens (prompt=half, gen=half)")
    parser.add_argument("--train-batch", type=int, default=4, help="Training batch size")
    parser.add_argument("--train-seq", type=int, default=8192, help="Training sequence length")
    parser.add_argument("--skip-sample", action="store_true", help="Skip sampling test")
    parser.add_argument("--skip-train", action="store_true", help="Skip training test")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optim step")
    args = parser.parse_args()

    print(f"Testing server at {args.url}")
    print(f"Base model: {args.base_model}")

    # Create service client
    service_client = tinker.ServiceClient(base_url=args.url, api_key="dummy")

    # Check capabilities
    capabilities = service_client.get_server_capabilities()
    model_names = [item.model_name for item in capabilities.supported_models]
    print(f"Supported models: {model_names}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    success = True

    if not args.skip_sample:
        # Create sampling client (base model)
        sampling_client = service_client.create_sampling_client(base_model=args.base_model)
        try:
            if not test_sample(sampling_client, tokenizer, args.sample_n, args.sample_tokens):
                print("Sampling test FAILED")
                success = False
            else:
                print("Sampling test PASSED")
        except Exception as e:
            print(f"Sampling test FAILED: {e}")
            success = False

    if not args.skip_train:
        # Create training client with LoRA
        training_client = service_client.create_lora_training_client(base_model=args.base_model)

        try:
            if not test_forward_backward(training_client, tokenizer, args.train_batch, args.train_seq):
                print("Forward-backward test FAILED")
                success = False
            else:
                print("Forward-backward test PASSED")
        except Exception as e:
            print(f"Forward-backward test FAILED: {e}")
            success = False

        try:
            if not test_optim_step(training_client, args.lr):
                print("Optim step test FAILED")
                success = False
            else:
                print("Optim step test PASSED")
        except Exception as e:
            print(f"Optim step test FAILED: {e}")
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
