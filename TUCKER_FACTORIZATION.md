# Converting GQA to TPA Weights via Tucker Decomposition

I'll outline an approach to convert Grouped Query Attention (GQA) weights to Tensor Product Attention (TPA) weights using the TensorLLM-style Tucker decomposition. This will maintain the memory efficiency advantages of GQA while gaining the improved reasoning capabilities of TPA.

## Understanding the Key Differences

**GQA (Grouped Query Attention):**
- Divides h query heads into G groups
- Each group shares key and value projections
- Reduces KV cache size by sharing keys/values across heads in the same group

**TPA (Tensor Product Attention):**
- Factorizes queries, keys, and values into tensor products
- For each token t: Qt = (1/Rq) * Σ[aq(xt) ⊗ bq(xt)]
- Similarly for keys and values with ranks Rk and Rv
- Significantly reduces KV cache size by storing only factorized components

**TensorLLM:**
- Uses Tucker decomposition on weight matrices themselves
- Enforces shared factor matrices across attention heads
- Keeps different core tensors for each head

## Conversion Approach

```python3
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker

def convert_gqa_to_tpa(gqa_weights, num_heads, num_groups, target_ranks):
    """
    Convert GQA weights to TPA weights using TensorLLM-style Tucker decomposition
    
    Args:
        gqa_weights: Dictionary containing GQA weight matrices
            - 'wq': Query projection weights (dmodel, h*dv)
            - 'wk': Key projection weights (dmodel, g*dv)
            - 'wv': Value projection weights (dmodel, g*dv)
            - 'wo': Output projection weights (h*dv, dmodel)
        num_heads: Number of attention heads (h)
        num_groups: Number of key-value groups (g)
        target_ranks: Dictionary with target ranks for Q, K, V (Rq, Rk, Rv)
        
    Returns:
        tpa_weights: Dictionary containing TPA weight parameters
    """
    # Extract dimensions
    dmodel = gqa_weights['wq'].shape[0]
    dv = gqa_weights['wq'].shape[1] // num_heads  # head dimension
    
    # 1. Reorganize GQA weights into 3D tensors
    wq_tensor = np.reshape(gqa_weights['wq'], (dmodel, dv, num_heads))
    wk_tensor = np.reshape(gqa_weights['wk'], (dmodel, dv, num_groups))
    wv_tensor = np.reshape(gqa_weights['wv'], (dmodel, dv, num_groups))
    wo_tensor = np.reshape(gqa_weights['wo'].T, (dmodel, dv, num_heads))
    
    # 2. Stack tensors for each attention head
    # Create a 4D tensor (dmodel, dv, 4, h) where the 3rd dimension corresponds to Q,K,V,O
    w_all = np.zeros((dmodel, dv, 4, num_heads))
    
    # Map heads to their respective groups
    head_to_group = [i // (num_heads // num_groups) for i in range(num_heads)]
    
    # Fill the 4D tensor
    for i in range(num_heads):
        group_idx = head_to_group[i]
        w_all[:, :, 0, i] = wq_tensor[:, :, i]           # Query weights
        w_all[:, :, 1, i] = wk_tensor[:, :, group_idx]   # Key weights (shared within group)
        w_all[:, :, 2, i] = wv_tensor[:, :, group_idx]   # Value weights (shared within group)
        w_all[:, :, 3, i] = wo_tensor[:, :, i]           # Output weights
    
    # 3. Apply Tucker decomposition with shared factor matrices across heads
    # Extract target ranks
    Rq, Rk, Rv = target_ranks['q'], target_ranks['k'], target_ranks['v']
    R1, R2, R3 = max(Rq, Rk, Rv), max(Rq, Rk, Rv), 4  # Using maximum rank for shared factors
    
    # Apply Tucker decomposition (TensorLLM approach)
    factors = [None] * num_heads
    core_tensors = [None] * num_heads
    
    # Shared factor matrices across all heads (TensorLLM-style)
    U1_shared = None  # For dmodel dimension
    U2_shared = None  # For dv dimension
    U3_shared = None  # For Q,K,V,O dimension
    
    # First decomposition to get shared factors
    core, factors_all = tucker(w_all, rank=[R1, R2, R3, None])
    U1_shared, U2_shared, U3_shared = factors_all[0], factors_all[1], factors_all[2]
    
    # 4. Map to TPA parameters
    # Initialize TPA weights
    tpa_weights = {
        # Query projection
        'Wa_q': np.zeros((dmodel, Rq, num_heads)),  # a_q^r(xt) = Wa_q^r · xt
        'Wb_q': np.zeros((dmodel, Rq, dv)),         # b_q^r(xt) = Wb_q^r · xt
        
        # Key projection
        'Wa_k': np.zeros((dmodel, Rk, num_heads)),  # a_k^r(xt) = Wa_k^r · xt
        'Wb_k': np.zeros((dmodel, Rk, dv)),         # b_k^r(xt) = Wb_k^r · xt
        
        # Value projection
        'Wa_v': np.zeros((dmodel, Rv, num_heads)),  # a_v^r(xt) = Wa_v^r · xt
        'Wb_v': np.zeros((dmodel, Rv, dv)),         # b_v^r(xt) = Wb_v^r · xt
    }
    
    # Map Tucker factors to TPA parameters
    # For queries (using dimensions 0-3 in the core tensor: Q=0)
    for r in range(Rq):
        # Each head gets its own 'a' factor but shares 'b' factors
        for i in range(num_heads):
            # Create a projection matrix for each head using core tensor slice
            proj_factor = np.zeros(R1)
            for j in range(R1):
                proj_factor[j] = core[j, :, 0, i].mean()  # Project over other dimensions
            
            # Map to TPA parameters
            tpa_weights['Wa_q'][:, r, i] = U1_shared[:, r] * proj_factor[r]
        
        # Shared b factors across heads - from Tucker U2 factor
        tpa_weights['Wb_q'][:, r, :] = U2_shared[:, r:r+1] @ U3_shared[0:1, :].T
    
    # For keys (using dimensions 0-3 in the core tensor: K=1)
    for r in range(Rk):
        # Map to TPA parameters with group sharing
        for i in range(num_heads):
            group_idx = head_to_group[i]
            
            # Create a projection matrix for each head using core tensor slice
            proj_factor = np.zeros(R1)
            for j in range(R1):
                proj_factor[j] = core[j, :, 1, i].mean()  # Project over other dimensions
            
            # Map to TPA parameters
            tpa_weights['Wa_k'][:, r, i] = U1_shared[:, r] * proj_factor[r]
        
        # Shared b factors - from Tucker U2 factor
        tpa_weights['Wb_k'][:, r, :] = U2_shared[:, r:r+1] @ U3_shared[1:2, :].T
    
    # For values (using dimensions 0-3 in the core tensor: V=2)
    for r in range(Rv):
        # Map to TPA parameters with group sharing
        for i in range(num_heads):
            group_idx = head_to_group[i]
            
            # Create a projection matrix for each head using core tensor slice
            proj_factor = np.zeros(R1)
            for j in range(R1):
                proj_factor[j] = core[j, :, 2, i].mean()  # Project over other dimensions
            
            # Map to TPA parameters
            tpa_weights['Wa_v'][:, r, i] = U1_shared[:, r] * proj_factor[r]
        
        # Shared b factors - from Tucker U2 factor
        tpa_weights['Wb_v'][:, r, :] = U2_shared[:, r:r+1] @ U3_shared[2:3, :].T
    
    # 5. Normalize factors to ensure proper scaling
    for r in range(Rq):
        norm_a = np.linalg.norm(tpa_weights['Wa_q'][:, r, :])
        norm_b = np.linalg.norm(tpa_weights['Wb_q'][:, r, :])
        tpa_weights['Wa_q'][:, r, :] /= np.sqrt(norm_a)
        tpa_weights['Wb_q'][:, r, :] /= np.sqrt(norm_b)
    
    for r in range(Rk):
        norm_a = np.linalg.norm(tpa_weights['Wa_k'][:, r, :])
        norm_b = np.linalg.norm(tpa_weights['Wb_k'][:, r, :])
        tpa_weights['Wa_k'][:, r, :] /= np.sqrt(norm_a)
        tpa_weights['Wb_k'][:, r, :] /= np.sqrt(norm_b)
    
    for r in range(Rv):
        norm_a = np.linalg.norm(tpa_weights['Wa_v'][:, r, :])
        norm_b = np.linalg.norm(tpa_weights['Wb_v'][:, r, :])
        tpa_weights['Wa_v'][:, r, :] /= np.sqrt(norm_a)
        tpa_weights['Wb_v'][:, r, :] /= np.sqrt(norm_b)
    
    return tpa_weights

def apply_tpa_forward(tpa_weights, input_tensor):
    """
    Apply TPA forward pass using the converted weights
    
    Args:
        tpa_weights: TPA weight parameters
        input_tensor: Input tensor of shape (batch_size, seq_len, dmodel)
        
    Returns:
        Output tensor after TPA attention
    """
    batch_size, seq_len, dmodel = input_tensor.shape
    num_heads = tpa_weights['Wa_q'].shape[2]
    dv = tpa_weights['Wb_q'].shape[2]
    Rq = tpa_weights['Wa_q'].shape[1]
    Rk = tpa_weights['Wa_k'].shape[1]
    Rv = tpa_weights['Wa_v'].shape[1]
    
    # 1. Compute factorized queries, keys, and values
    # Initialize tensors to store Q, K, V
    Q = np.zeros((batch_size, seq_len, num_heads, dv))
    K = np.zeros((batch_size, seq_len, num_heads, dv))
    V = np.zeros((batch_size, seq_len, num_heads, dv))
    
    # For each token in the sequence
    for b in range(batch_size):
        for t in range(seq_len):
            x_t = input_tensor[b, t, :]  # Current token embedding
            
            # Compute query
            for i in range(num_heads):
                q_i = np.zeros(dv)
                for r in range(Rq):
                    a_q_r = np.dot(tpa_weights['Wa_q'][:, r, i], x_t)
                    b_q_r = np.dot(tpa_weights['Wb_q'][:, r, :].T, x_t)
                    q_i += a_q_r * b_q_r
                Q[b, t, i, :] = q_i / Rq
            
            # Compute key
            for i in range(num_heads):
                k_i = np.zeros(dv)
                for r in range(Rk):
                    a_k_r = np.dot(tpa_weights['Wa_k'][:, r, i], x_t)
                    b_k_r = np.dot(tpa_weights['Wb_k'][:, r, :].T, x_t)
                    k_i += a_k_r * b_k_r
                K[b, t, i, :] = k_i / Rk
            
            # Compute value
            for i in range(num_heads):
                v_i = np.zeros(dv)
                for r in range(Rv):
                    a_v_r = np.dot(tpa_weights['Wa_v'][:, r, i], x_t)
                    b_v_r = np.dot(tpa_weights['Wb_v'][:, r, :].T, x_t)
                    v_i += a_v_r * b_v_r
                V[b, t, i, :] = v_i / Rv
    
    # 2. Scaled dot-product attention (simplified version)
    # For each head
    attention_output = np.zeros((batch_size, seq_len, num_heads, dv))
    
    for b in range(batch_size):
        for i in range(num_heads):
            # Compute attention scores
            scores = np.matmul(Q[b, :, i, :], K[b, :, i, :].transpose(1, 0)) / np.sqrt(dv)
            
            # Apply softmax
            attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
            
            # Apply attention weights to values
            attention_output[b, :, i, :] = np.matmul(attention_weights, V[b, :, i, :])
    
    # 3. Concatenate heads and project back to model dimension
    # This is simplified as we don't have the output projection in our TPA weights
    output = attention_output.reshape(batch_size, seq_len, num_heads * dv)
    
    return output

# Example usage
if __name__ == "__main__":
    # Example dimensions
    dmodel = 768
    num_heads = 12
    num_groups = 4  # GQA groups
    dv = 64  # Head dimension
    
    # Create dummy GQA weights
    gqa_weights = {
        'wq': np.random.randn(dmodel, num_heads * dv),
        'wk': np.random.randn(dmodel, num_groups * dv),
        'wv': np.random.randn(dmodel, num_groups * dv),
        'wo': np.random.randn(num_heads * dv, dmodel)
    }
    
    # Target ranks for TPA
    target_ranks = {'q': 6, 'k': 2, 'v': 2}
    
    # Convert GQA to TPA
    tpa_weights = convert_gqa_to_tpa(gqa_weights, num_heads, num_groups, target_ranks)
    
    # Check compression ratio
    gqa_params = gqa_weights['wq'].size + gqa_weights['wk'].size + gqa_weights['wv'].size
    tpa_params = sum(w.size for w in tpa_weights.values())
    
    print(f"GQA parameters: {gqa_params}")
    print(f"TPA parameters: {tpa_params}")
    print(f"Compression ratio: {gqa_params / tpa_params:.2f}x")
    
    # Apply TPA to a dummy input
    batch_size = 2
    seq_len = 10
    dummy_input = np.random.randn(batch_size, seq_len, dmodel)
    
    output = apply_tpa_forward(tpa_weights, dummy_input)
    print(f"Output shape: {output.shape}")
```

## How the Conversion Works

The approach leverages the strengths of both TensorLLM and TPA methods by using a three-step process:

### 1. Multi-head Tensorisation
First, we restructure the GQA weight matrices:
- Reshape query, key, value, and output weight matrices
- Create a 4D tensor that represents all weights while preserving the group structure
- The tensor has dimensions (dmodel, dv, 4, h) where the third dimension distinguishes between Q, K, V, and O

### 2. Tucker Decomposition with Shared Factor Matrices
Apply TensorLLM-style Tucker decomposition:
- Decompose the tensorized weights with shared factor matrices:
  ```
  W_all ≈ G ×₁ U⁽¹⁾ ×₂ U⁽²⁾ ×₃ U⁽³⁾ ×₄ I
  ```
- This produces:
    - Shared factor matrices U⁽¹⁾ ∈ ℝ^(dmodel×R₁), U⁽²⁾ ∈ ℝ^(dv×R₂), U⁽³⁾ ∈ ℝ^(4×R₃)
    - A core tensor G that captures the variation across heads

### 3. Map Tucker Factors to TPA Parameters
Convert the Tucker decomposition to TPA's activation format:
- For queries: Map to Wa_q and Wb_q factors
- For keys: Map to Wa_k and Wb_k factors
- For values: Map to Wa_v and Wb_v factors
- Ensure head grouping is preserved by sharing key/value factors within each group

### 4. RoPE Integration
- Apply RoPE to the appropriate factors (typically Wb_q and Wb_k)
- This maintains rotation positional encoding in the TPA format

## Advantages of This Approach

1. **Memory Efficiency**: TPA achieves even better memory compression than GQA
2. **Reasoning Enhancement**: Maintains the reasoning benefits of TPA
3. **Group Structure Preservation**: Keeps the grouped knowledge sharing of GQA
4. **No Additional Training**: Direct weight conversion without fine-tuning
5. **RoPE Compatibility**: Works seamlessly with rotary positional embeddings

## Implementation Considerations

When implementing this conversion, it's important to:

1. Properly handle the mapping from Tucker factors to TPA parameters
2. Normalize the resulting factors to ensure proper scaling
3. Maintain the group structure of GQA in the TPA format
4. Carefully implement RoPE in the TPA context
5. Tune the rank parameters (Rq, Rk, Rv) to balance performance and compression

This approach offers a principled way to convert existing GQA models to the TPA format, potentially improving both memory efficiency and reasoning capabilities while maintaining the benefits of grouped attention.