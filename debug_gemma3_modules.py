#!/usr/bin/env python3

"""
Debug script to identify correct LoRA target modules for Gemma3
"""

import torch
from transformers import Gemma3ForCausalLM, GemmaTokenizer

def debug_gemma3_architecture():
    """Debug Gemma3 model architecture to find correct module names"""
    print("Loading Gemma3 model...")
    
    try:
        model = Gemma3ForCausalLM.from_pretrained(
            "/home/lewis/github/gemma3",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False
        )
        
        print("\n=== Gemma3 Model Architecture ===")
        print(f"Model type: {type(model)}")
        print(f"Config: {model.config}")
        
        print("\n=== All Named Modules ===")
        module_names = []
        for name, module in model.named_modules():
            module_names.append(name)
            if any(target in name for target in ['proj', 'attn', 'linear', 'embed']):
                print(f"{name}: {type(module)}")
        
        print("\n=== Potential LoRA Target Modules ===")
        potential_targets = []
        for name in module_names:
            if any(target in name for target in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
                potential_targets.append(name.split('.')[-1])  # Get the actual module name
                print(f"Found: {name}")
        
        # Get unique target module names
        unique_targets = list(set(potential_targets))
        print(f"\n=== Suggested LoRA target_modules ===")
        print(f"target_modules = {unique_targets}")
        
        print("\n=== Model Embedding Token Access Path ===")
        try:
            # Test different paths to embed_tokens
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                print("✓ model.model.embed_tokens exists")
                embed_path = "model.model.embed_tokens"
            else:
                print("✗ model.model.embed_tokens does not exist")
                embed_path = "unknown"
            
            print(f"Recommended embed tokens path: {embed_path}")
            
        except Exception as e:
            print(f"Error testing embed tokens path: {e}")
        
        print("\n=== RMS Norm Check ===")
        try:
            # Check for RMSNorm vs LayerNorm
            rms_norm_eps = getattr(model.config, 'rms_norm_eps', None)
            if rms_norm_eps:
                print(f"✓ Model uses RMSNorm with eps={rms_norm_eps}")
                print(f"Recommendation: Use eps={rms_norm_eps} for all LayerNorm layers")
            else:
                print("Model does not specify rms_norm_eps")
        except Exception as e:
            print(f"Error checking RMS norm: {e}")
            
    except Exception as e:
        print(f"Error loading Gemma3 model: {e}")
        return None

if __name__ == "__main__":
    debug_gemma3_architecture()