#!/usr/bin/env python3
"""
Download and Test Base Model

This script:
1. Downloads the base model from HuggingFace
2. Tests model loading
3. Tests inference on sample questions
4. Saves model info for reference
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def check_system_info():
    """Display system information."""
    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"  Free Memory: {free_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA Not Available (will use CPU - much slower)")


def download_model(model_name: str, cache_dir: str = "models/base", use_8bit: bool = False):
    """
    Download and load model.
    
    Args:
        model_name: HuggingFace model identifier
        cache_dir: Where to cache the model
        use_8bit: Whether to use 8-bit quantization
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print("\n" + "="*80)
    print(f"DOWNLOADING MODEL: {model_name}")
    print("="*80)
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTarget device: {device}")
    
    if use_8bit and not torch.cuda.is_available():
        print("⚠ 8-bit quantization requires CUDA, disabling...")
        use_8bit = False
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✓ Tokenizer loaded")
        print(f"  Vocabulary size: {len(tokenizer)}")
        
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        print("\nFor Llama models, you may need to:")
        print("1. Request access at https://huggingface.co/meta-llama/Llama-3-8B-Instruct")
        print("2. Login: huggingface-cli login")
        raise
    
    # Load model
    print("\n2. Loading model...")
    print("   (This may take several minutes for first download)")
    
    start_time = time.time()
    
    try:
        load_kwargs = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = torch.float16
            
            if use_8bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                print("   Using 8-bit quantization")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        load_time = time.time() - start_time
        
        print(f"✓ Model loaded in {load_time:.1f} seconds")
        
        # Model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel Information:")
        print(f"  Parameters: {num_params/1e9:.2f}B")
        print(f"  Layers: {model.config.num_hidden_layers}")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Device: {next(model.parameters()).device}")
        print(f"  Dtype: {next(model.parameters()).dtype}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            print(f"  GPU Memory Used: {memory_used:.2f} GB")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("1. Check disk space (need ~15GB for 8B models)")
        print("2. Check GPU memory (need ~16GB for 8B models in FP16)")
        print("3. Try --use-8bit flag for 8-bit quantization (~8GB)")
        print("4. Try smaller model: microsoft/Phi-3-mini-4k-instruct (~4GB)")
        raise


def test_inference(model, tokenizer):
    """Test model inference with sample questions."""
    print("\n" + "="*80)
    print("TESTING INFERENCE")
    print("="*80)
    
    # Load a sample question from data if available
    test_prompts = [
        "Let's think step by step. Is Mount Everest taller than K2?",
        "Let's think step by step. Did World War I happen before World War II?",
        "Let's think step by step. Is Russia larger than Canada?",
    ]
    
    device = next(model.parameters()).device
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'-'*80}")
        print(f"Test {i}/{len(test_prompts)}")
        print(f"{'-'*80}")
        print(f"Prompt: {prompt}\n")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        print("Generating response...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generation_time = time.time() - start_time
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        
        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"Response ({num_tokens} tokens, {generation_time:.2f}s, {num_tokens/generation_time:.1f} tok/s):")
        print(generated_text)
        print()


def save_model_info(model, tokenizer, model_name: str, output_file: str = "models/model_info.json"):
    """Save model information for reference."""
    print("\n" + "="*80)
    print("SAVING MODEL INFORMATION")
    print("="*80)
    
    info = {
        "model_name": model_name,
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "config": {
            "num_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "num_attention_heads": model.config.num_attention_heads,
            "vocab_size": model.config.vocab_size,
        },
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "tokenizer": {
            "vocab_size": len(tokenizer),
            "model_max_length": tokenizer.model_max_length,
        },
    }
    
    if torch.cuda.is_available():
        info["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "memory_used_gb": torch.cuda.memory_allocated(0) / 1e9,
        }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✓ Model information saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Download and test base model")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization (saves memory)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("BASE MODEL DOWNLOAD AND TEST")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"8-bit: {args.use_8bit}")
    
    try:
        # Check system
        check_system_info()
        
        # Download model
        model, tokenizer = download_model(
            args.model,
            cache_dir="models/base",
            use_8bit=args.use_8bit
        )
        
        # Test inference
        test_inference(model, tokenizer)
        
        # Save info
        save_model_info(model, tokenizer, args.model)
        
        print("\n" + "="*80)
        print("✓ SUCCESS! MODEL READY TO USE")
        print("="*80)
        print("\nNext steps:")
        print("1. Review models/model_info.json")
        print("2. Open notebooks/02_baseline_experiments.ipynb")
        print("3. Start testing faithfulness on your data!")
        print()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())