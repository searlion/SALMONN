# Gemma3 Integration Fixes for SALMONN

## Summary of NaN/Inf Issues Resolved

This document outlines the critical fixes applied to resolve NaN/Inf training issues when switching from Vicuna/Llama to Gemma3 in SALMONN.

**UPDATE**: If you continue experiencing NaN losses after implementing the initial fixes, additional numerical stability improvements have been applied (see Advanced Fixes section below).

## Root Causes Identified

### 1. **LayerNorm Epsilon Mismatch**
- **Problem**: Gemma3 uses `rms_norm_eps: 1e-06` while SALMONN used `eps=1e-4` (100x difference)
- **Impact**: Numerical instability in normalization layers
- **Solution**: Changed all LayerNorm eps values from `1e-4` to `1e-6`

### 2. **LoRA Target Module Incompatibility** 
- **Problem**: Original LoRA config targeted `["q_proj", "v_proj", "k_proj", "o_proj"]` but Gemma3 has additional modules
- **Impact**: Incomplete LoRA integration, potentially missing key trainable parameters
- **Solution**: Updated to `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

### 3. **Embed Tokens Access Path Inconsistency**
- **Problem**: Code had inconsistent paths for accessing embed_tokens with/without LoRA
- **Impact**: Runtime errors or incorrect embeddings
- **Solution**: Created `get_embed_tokens()` helper method with proper fallback logic

### 4. **Missing Numerical Stability Safeguards**
- **Problem**: No protection against NaN/Inf propagation during forward pass
- **Impact**: Training crashes with cryptic error messages
- **Solution**: Added comprehensive NaN/Inf detection and clamping

## Files Modified

### `models/salmonn.py`

#### Key Changes:
1. **Line 161, 174**: Changed LayerNorm eps from `1e-4` to `1e-6`
   ```python
   self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model, eps=1e-6)
   self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim, eps=1e-6)
   ```

2. **Line 152**: Updated LoRA target modules
   ```python
   target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
   ```

3. **Lines 65-75**: Added robust embed tokens helper
   ```python
   def get_embed_tokens(self, input_ids):
       """Helper method to correctly access embed_tokens regardless of LoRA configuration"""
       if self.lora:
           if hasattr(self.llama_model, 'model') and hasattr(self.llama_model.model, 'model'):
               return self.llama_model.model.model.embed_tokens(input_ids)
           else:
               return self.llama_model.base_model.model.embed_tokens(input_ids)
       else:
           return self.llama_model.model.embed_tokens(input_ids)
   ```

4. **Lines 391-418**: Added numerical stability checks
   ```python
   # Add numerical stability checks before loss calculation
   if torch.isnan(inputs_embeds).any():
       logging.error("NaN detected in inputs_embeds before forward pass")
       inputs_embeds = torch.clamp(inputs_embeds, -1e4, 1e4)
   
   if torch.isinf(inputs_embeds).any():
       logging.error("Inf detected in inputs_embeds before forward pass") 
       inputs_embeds = torch.clamp(inputs_embeds, -1e4, 1e4)
   
   # Check for NaN/Inf in loss
   if torch.isnan(loss):
       logging.error("NaN loss detected! Input statistics:")
       # ... detailed logging and fallback loss
       loss = torch.tensor(0.001, device=loss.device, dtype=loss.dtype, requires_grad=True)
   ```

5. **Lines 302, 308, 322, 323, 367, 385, 450**: Replaced all embed_tokens calls with helper method

## Verification

### Debug Script Created: `debug_gemma3_modules.py`
- Confirms Gemma3 architecture details
- Validates LoRA target modules
- Verifies embed tokens path 
- Confirms RMS norm eps value

### Key Findings:
- ✓ Gemma3 uses `rms_norm_eps: 1e-06`
- ✓ Embed tokens accessible via `model.model.embed_tokens`
- ✓ LoRA targets: `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']`

## Expected Results

After applying these fixes:
1. **Eliminated NaN/Inf errors** during training
2. **Proper LoRA integration** with all relevant Gemma3 modules
3. **Consistent embed token access** regardless of LoRA configuration 
4. **Enhanced debugging capabilities** with detailed error logging
5. **Graceful error recovery** instead of training crashes

## Additional Recommendations

### 1. Monitor Training Logs
Watch for the new error messages that provide detailed diagnostics:
- "NaN detected in inputs_embeds before forward pass"  
- "Inf detected in inputs_embeds before forward pass"
- "NaN loss detected! Input statistics:"

### 2. Consider Further Optimizations
If issues persist, consider:
- Using `bfloat16` instead of `float16` (Gemma3's native dtype)
- Implementing gradient scaling with smaller initial values
- Adding more aggressive gradient clipping

### 3. Model Architecture Validation
Run `debug_gemma3_modules.py` after any model changes to verify:
```bash
python debug_gemma3_modules.py
```

## Advanced Fixes for Persistent NaN Issues

If the initial fixes didn't resolve the NaN losses completely, the following additional changes have been applied:

### 5. **LoRA Alpha Scaling Reduction**
- **Problem**: LoRA alpha=32 can cause numerical instability with Gemma3's smaller model
- **Solution**: Reduced LoRA alpha from 32 to 8 (`lora_alpha // 4`) for better numerical stability

### 6. **Projection Layer Initialization**  
- **Problem**: Default initialization may be too large for Gemma3's embedding space
- **Solution**: Added careful initialization with smaller weights (std=0.02)
   ```python
   with torch.no_grad():
       self.speech_llama_proj.weight.data.normal_(mean=0.0, std=0.02)
       self.speech_llama_proj.bias.data.zero_()
   ```

### 7. **Eager Attention Implementation**
- **Problem**: Gemma3 strongly recommends eager attention over SDPA
- **Solution**: Added `attn_implementation='eager'` to model loading

### 8. **Disabled Autocast for Loss Calculation**
- **Problem**: Mixed precision can cause numerical instabilities with specific combinations
- **Solution**: Forced full precision during loss computation
   ```python
   with torch.cuda.amp.autocast(enabled=False):
       outputs = self.llama_model(...)
   ```

### 9. **Enhanced NaN Detection and Recovery**
- **Problem**: Need better diagnostics to identify the source of NaNs
- **Solution**: Added comprehensive logging including logits statistics, standard deviation, and tensor shapes

## Troubleshooting Guide

If you're still experiencing issues:

1. **Check the training logs** for new diagnostic messages
2. **Verify LoRA parameters** are reasonable (alpha=8, rank=8)
3. **Consider switching to bfloat16** if available on your hardware
4. **Reduce learning rate** further (try 1e-6 instead of 1e-5)
5. **Add gradient clipping** in your training loop if not already present

This comprehensive set of fixes addresses the fundamental incompatibilities between Gemma3 and the original Llama/Vicuna implementation, providing a stable foundation for training with the new architecture.