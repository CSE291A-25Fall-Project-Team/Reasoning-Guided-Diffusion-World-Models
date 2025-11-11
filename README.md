# Reasoning-Guided-Diffusion-World-Models

1. git clone cyclereward's repo
2. Change the ...med.py to the folllowing:
   - replace
      `
     from transformers.modeling_utils import (
        PreTrainedModel,
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
       )
     `
   - use:
      `
      import torch
      try:
          # Old versions (Transformers <=4.43)
          from transformers.modeling_utils import (
              PreTrainedModel,
              apply_chunking_to_forward,
              find_pruneable_heads_and_indices,
              prune_linear_layer,
          )
      except ImportError:
          # Newer Transformers versions (>=4.45)
          from transformers.modeling_utils import PreTrainedModel
      
          # Define safe no-op fallbacks for removed functions
          def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
              return forward_fn(*input_tensors)
      
          def find_pruneable_heads_and_indices(*args, **kwargs):
              return [], torch.tensor([])
      
          # Reimplement prune_linear_layer (simple linear pruning fallback)
          def prune_linear_layer(layer, index, dim=0):
              """Prune a linear layer (used in attention heads) for newer Transformers versions."""
              W = layer.weight.index_select(dim, index).clone().detach()
              if layer.bias is not None:
                  if dim == 1:
                      b = layer.bias.clone().detach()
                  else:
                      b = layer.bias[index].clone().detach()
              new_layer = torch.nn.Linear(W.size(1), W.size(0), bias=layer.bias is not None).to(W.device)
              new_layer.weight.data = W.contiguous()
              if layer.bias is not None:
                  new_layer.bias.data = b.contiguous()
              return new_layer
     `
