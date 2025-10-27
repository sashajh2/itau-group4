# Combines semantic and forensic contrastive objectives during joint training.
import torch
import torch.nn as nn
import torch.nn.functional as F


def cov(feats):
    """Compute covariance matrix: feats: [N, C]"""
    x = feats - feats.mean(0, keepdim=True)
    return (x.T @ x) / (max(1, len(feats) - 1))


def offdiag_frobenius(M):
    """Compute off-diagonal Frobenius norm"""
    return (M - torch.diag(torch.diag(M))).pow(2).sum()


def variance_compactness(z_reals):
    """
    L_hom: keep real embeddings compact (VICReg-style)
    Computes trace of covariance matrix to minimize variance in real samples
    """
    if z_reals.shape[0] <= 1:
        return z_reals.sum() * 0.0
    C = cov(z_reals)
    return torch.trace(C)


def cross_cov_penalty(z_a, z_b):
    """
    L_orth: decorrelate two embedding heads (orthogonality constraint)
    Minimizes off-diagonal elements of cross-covariance matrix
    """
    za = z_a - z_a.mean(0, keepdim=True)
    zb = z_b - z_b.mean(0, keepdim=True)
    C = (za.T @ zb) / (max(1, len(za) - 1))
    return offdiag_frobenius(C)


def build_pos_neg_from_batch(z_id_real, id_idx_real, max_pos=4, max_neg=32):
    """
    Build positive and negative pairs for supervised contrastive learning
    
    Args:
        z_id_real: [Br, D] identity embeddings (only real rows)
        id_idx_real: [Br] integer labels for identity (only real rows)
        max_pos: maximum number of positive samples per anchor
        max_neg: maximum number of negative samples per anchor
    
    Returns:
        anchors [B', D], positives [B', P, D], negatives [B', N, D]
        Only returns anchors with >=1 positive and >=1 negative
    """
    device = z_id_real.device
    Br, D = z_id_real.shape
    
    if Br < 2:
        return None  # not enough reals to form pairs
    
    anchors, pos_list, neg_list = [], [], []
    ids = id_idx_real
    
    # Precompute indices per identity
    by_id = {}
    for i in range(Br):
        by_id.setdefault(int(ids[i].item()), []).append(i)
    
    # For each sample as anchor
    for i in range(Br):
        my_id = int(ids[i].item())
        same = by_id[my_id].copy()
        if i in same:
            same.remove(i)  # remove self
        diff = [j for j in range(Br) if int(ids[j].item()) != my_id]
        
        if len(same) == 0 or len(diff) == 0:
            continue  # skip anchors without positives or negatives
        
        anchors.append(z_id_real[i])
        
        # Sample positive examples
        pos_samples = [z_id_real[j] for j in same[:max_pos]]
        pos_tensor = torch.stack(pos_samples, dim=0)  # [P, D]
        
        # Pad if necessary
        if pos_tensor.shape[0] < max_pos:
            padding = torch.zeros(max_pos - pos_tensor.shape[0], D, device=device)
            pos_tensor = torch.cat([pos_tensor, padding], dim=0)
        
        pos_list.append(pos_tensor)
        
        # Sample negative examples
        neg_samples = [z_id_real[j] for j in diff[:max_neg]]
        neg_tensor = torch.stack(neg_samples, dim=0)  # [N, D]
        
        # Pad if necessary
        if neg_tensor.shape[0] < max_neg:
            padding = torch.zeros(max_neg - neg_tensor.shape[0], D, device=device)
            neg_tensor = torch.cat([neg_tensor, padding], dim=0)
        
        neg_list.append(neg_tensor)
    
    if not anchors:
        return None
    
    anchors_tensor = torch.stack(anchors, dim=0)  # [B', D]
    positives_tensor = torch.stack(pos_list, dim=0)  # [B', P, D]
    negatives_tensor = torch.stack(neg_list, dim=0)  # [B', N, D]
    
    return anchors_tensor, positives_tensor, negatives_tensor


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (as in SimCLR/CutLER)
    Encourages positive pairs (same identity) to be close and negative pairs (different identity) to be far
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positives, negatives):
        """
        Args:
            anchor: [B', D] anchor embeddings
            positives: [B', P, D] positive embeddings
            negatives: [B', N, D] negative embeddings
        
        Returns:
            scalar loss value
        """
        device = anchor.device
        B = anchor.shape[0]
        
        anchor = F.normalize(anchor, dim=1)
        positives = F.normalize(positives, dim=2)
        negatives = F.normalize(negatives, dim=2)
        
        # Compute similarity scores
        pos_sim = torch.bmm(anchor.unsqueeze(1), positives.transpose(1, 2)).squeeze(1) / self.temperature
        neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / self.temperature
        
        # Concatenate and create labels (positives are class 1)
        sims = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(B, sims.size(1), device=device)
        labels[:, :pos_sim.size(1)] = 1
        
        # Compute InfoNCE loss
        exp_sims = torch.exp(sims)
        log_prob = sims - torch.log(exp_sims.sum(dim=1, keepdim=True) + 1e-12)
        mean_log_pos = (labels * log_prob).sum(1) / (labels.sum(1) + 1e-12)
        
        return -mean_log_pos.mean()
