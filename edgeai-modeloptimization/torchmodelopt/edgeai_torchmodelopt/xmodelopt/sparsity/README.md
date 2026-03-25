## Sparsity methods supported
* This module primarily implements **2:4 sparsity** pattern, which is a specific form of structured sparsity where in each block of 4 weights, only 2 are kept and the rest are set to zero. This results in a 50% reduction in model size with minimal accuracy impact when done properly.

* **Supported layer types**:
  * `Conv2d` - Convolutional layers
  * `Linear` - Fully connected layers
  * `matmul` operations

## How weight masks are created
The sparsity pattern is implemented using weight masks that determine which weights to keep and which to prune:

1. **Mask creation methods**:
   * `topk` (default): For each block of 4 weights, the 2 weights with the largest absolute magnitude are kept, and the remaining 2 are pruned.
   * `topk_blockwise`: Similar to `topk` but performs incremental sparsity blockwise, where a percentage of blocks have their weights set to zero at each epoch.

2. **Training process**:
   * Gradual sparsification: Sparsity is introduced gradually over training epochs to allow the model to adapt.
   * Configurable start and end epochs: Users can specify when sparsification begins (`sparsity_start_epoch`) and completes (`sparsity_end_epoch`).
   * Power-law scheduling: The sparsity increases following a power-law curve controlled by parameter `p`.

3. **Weight scaling options**:
   * No scaling: Apply masks without modifying weight magnitudes
   * Per-epoch scaling: Scale weights to maintain model performance by preserving the effective norm

4. **Soft masks during training**:
   * Supports Soft Straight-Through Estimator (S-STE) for better gradient flow during training

## How to apply sparsity to models
### Gradual Sparsification during Training

```python
import torch
from edgeai_torchmodelopt.xmodelopt.sparsity.v3 import SparserModule

# Initialize your model
model = YourModelClass()

# Wrap your model with the SparserModule, specifically for 2:4 sparsity
sparse_model = SparserModule(
    model,
    sparsity_ratio=0.5,    # Target 50% sparsity
    sparsity_type='n2m',   # Use n:m structured sparsity
    sparsity_m=4,          # Block size of 4 (for 2:4 sparsity)
    sparsity_start_epoch=5,
    sparsity_end_epoch=20
)

# Train your model normally
...
# After training, finalize the sparse model
...

# Save the sparse model
torch.save(sparse_model.state_dict(), "sparse_model.pth")
```

## Important Guidelines
When applying 2:4 sparsity to neural networks:

* The 2:4 sparsity pattern is hardware-friendly and delivers good performance when implemented correctly.
* For best results, gradually introduce sparsity during training rather than applying it all at once.
* Different layers may respond differently to sparsity - monitor layer-wise accuracy impact if possible.
* Consider using weight scaling options to maintain model performance.
* For inference deployment, binary masks can be applied to create fully sparse models.