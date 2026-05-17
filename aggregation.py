"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations
import torch

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask_on_device = attention_mask.to(hidden_states.device)
    real_positions = mask_on_device.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    
    # Берем фичи финального токена с трех разных глубин (ранняя, средняя, поздняя)
    feat_late = hidden_states[-1, last_pos, :]
    feat_mid = hidden_states[-6, last_pos, :]
    feat_early = hidden_states[-12, last_pos, :]
    
    return torch.cat([feat_late, feat_mid, feat_early], dim=0)

def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask_on_device = attention_mask.to(hidden_states.device)
    real_positions = mask_on_device.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    
    l2_late = torch.norm(hidden_states[-1, last_pos, :], p=2, keepdim=True)
    l2_mid = torch.norm(hidden_states[-6, last_pos, :], p=2, keepdim=True)
    l2_early = torch.norm(hidden_states[-12, last_pos, :], p=2, keepdim=True)
    
    seq_len_feature = torch.tensor([float(last_pos)], dtype=torch.float32, device=hidden_states.device)
    
    return torch.cat([l2_late, l2_mid, l2_early, seq_len_feature], dim=0)

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)
    return agg_features
