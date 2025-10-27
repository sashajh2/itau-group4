# training/trainer.py
from evaluation.evaluator import EmbeddingEvaluator
from models.model_factory import create_model
from losses.loss import (
    variance_compactness, 
    cross_cov_penalty, 
    build_pos_neg_from_batch,
    SupConLoss
)
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class ModelTrainer:
    """Unified training pipeline"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.evaluator = EmbeddingEvaluator(device)
    
    def train_and_evaluate(self, model_config: Dict[str, Any], 
                          training_config: Dict[str, Any],
                          evaluation_config: Dict[str, Any],
                          data_loaders: Dict[str, Any]) -> Dict[str, Any]:
        """Main training and evaluation function"""
        
        # Create model
        model = create_model(model_config['type'], model_config)
        model = model.to(self.device)
        
        # Train model
        if training_config.get('stage_a', False):
            self._train_stage_a(model, data_loaders, training_config)
        
        if training_config.get('stage_b', False):
            self._train_stage_b(model, data_loaders, training_config)
        
        # Evaluate model
        # Use standard loaders for evaluation (not episodic)
        results = self.evaluator.evaluate_model(
            model, 
            data_loaders['train'],
            data_loaders['val'], 
            data_loaders['test'],
            evaluation_config
        )
        
        # Also evaluate on episodic test loader for few-shot
        if 'test_het_eval' in data_loaders:
            few_shot_results = self.evaluator._evaluate_few_shot_episodic(
                data_loaders['train'],
                data_loaders['test_het_eval'],
                model
            )
            results.update(few_shot_results)
        
        return results
    
    def _train_stage_a(self, model, data_loaders, config):
        """Stage A training - pretraining on real-only data with identity contrastive loss"""
        
        if not isinstance(model, create_model('orthogonal', {}).__class__):
            raise ValueError("Stage A training only supports orthogonal models")
        
        # Use standard training loader (works for both hom and het Stage A)
        train_loader = data_loaders['train']
        
        # Training hyperparameters
        epochs = config.get('stage_a_epochs', 3)
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-4)
        
        # Loss weights
        W_HOM = config.get('W_HOM', 1.0)
        W_ORTH = config.get('W_ORTH', 0.1)
        W_HET_A = config.get('W_HET_A', 0.2)
        
        # Create optimizer
        params = list(model.adapter.parameters()) + list(model.f_hom.parameters()) + list(model.f_id.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        
        # Loss functions
        supcon_loss_fn = SupConLoss()
        
        model.adapter.train()
        model.f_hom.train()
        model.f_id.train()
        
        for epoch in range(epochs):
            running_loss = {"hom": 0.0, "orth": 0.0, "het": 0.0, "total": 0.0}
            n = 0
            
            pbar = tqdm(train_loader, desc=f"[Stage A Epoch {epoch+1}]")
            for batch in pbar:
                e = batch["e"].to(self.device)
                mr = batch["is_real"].to(self.device).bool()  # Real/fake mask
                ids = batch["id_idx"].to(self.device)
                
                # Forward pass
                e2 = model.adapter(e)
                z_hom = model.f_hom(e2)
                z_id = model.f_id(e2)
                
                # Loss on real samples only
                if mr.any():
                    L_hom = variance_compactness(z_hom[mr])
                else:
                    L_hom = torch.tensor(0.0, device=self.device)
                
                # Orthogonality loss (on reals only)
                L_orth = torch.tensor(0.0, device=self.device)
                if W_ORTH > 0 and mr.any():
                    zha, zia = z_hom[mr], z_id[mr]
                    if zha.shape[0] > 1:
                        L_orth = cross_cov_penalty(zha, zia)
                
                # Identity contrastive loss (supcon)
                if mr.any():
                    supcon_data = build_pos_neg_from_batch(z_id[mr], ids[mr], max_pos=4, max_neg=32)
                    if supcon_data:
                        L_het = supcon_loss_fn(*supcon_data)
                    else:
                        L_het = torch.tensor(0.0, device=self.device)
                else:
                    L_het = torch.tensor(0.0, device=self.device)
                
                # Total loss
                loss = W_HOM * L_hom + W_ORTH * L_orth + W_HET_A * L_het
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update running stats
                bs = e.size(0)
                n += bs
                running_loss["hom"] += L_hom.item() * bs
                running_loss["orth"] += L_orth.item() * bs
                running_loss["het"] += L_het.item() * bs
                running_loss["total"] += loss.item() * bs
                
                if n % 100 == 0:
                    pbar.set_postfix({k: f"{v/max(1,n):.4f}" for k, v in running_loss.items()})
            
            # Average losses
            for k in running_loss:
                running_loss[k] /= max(1, n)
            
            print(f"[Stage A Epoch {epoch+1}] "
                  f"hom={running_loss['hom']:.4f} orth={running_loss['orth']:.4f} "
                  f"het={running_loss['het']:.4f} total={running_loss['total']:.4f}")
    
    def _train_stage_b(self, model, data_loaders, config):
        """Stage B training - classification fine-tuning with all data"""
        
        if not isinstance(model, create_model('orthogonal', {}).__class__):
            raise ValueError("Stage B training only supports orthogonal models")
        
        train_loader = data_loaders['train']  # Works for both hom and het Stage B
        val_loader = data_loaders.get('val', None)
        
        # Training hyperparameters
        epochs = config.get('stage_b_epochs', 8)
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-4)
        
        # Loss weights
        W_HOM = config.get('W_HOM', 1.0)
        W_ORTH = config.get('W_ORTH', 0.1)
        W_HET_B = config.get('W_HET_B', 0.0)
        W_CLS = config.get('W_CLS', 1.0)
        
        # Create optimizer
        params = (list(model.adapter.parameters()) + list(model.f_hom.parameters()) + 
                 list(model.f_id.parameters()) + list(model.classifier.parameters()))
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        
        # Loss functions
        supcon_loss_fn = SupConLoss()
        bce_loss_fn = nn.BCEWithLogitsLoss()
        
        model.adapter.train()
        model.f_hom.train()
        model.f_id.train()
        model.classifier.train()
        
        for epoch in range(epochs):
            running_loss = {"hom": 0.0, "orth": 0.0, "het": 0.0, "cls": 0.0, "total": 0.0}
            n = 0
            
            pbar = tqdm(train_loader, desc=f"[Stage B Epoch {epoch+1}]")
            for batch in pbar:
                e = batch["e"].to(self.device)
                y = batch["y"].to(self.device)  # Binary classification labels
                mr = batch["is_real"].to(self.device).bool()
                ids = batch["id_idx"].to(self.device)
                
                # Forward pass
                e2 = model.adapter(e)
                z_hom = model.f_hom(e2)
                z_id = model.f_id(e2)
                logits = model.classifier(z_hom)
                
                # Homogeneous loss (on reals only)
                if mr.any():
                    L_hom = variance_compactness(z_hom[mr])
                else:
                    L_hom = torch.tensor(0.0, device=self.device)
                
                # Orthogonality loss (on reals only)
                L_orth = torch.tensor(0.0, device=self.device)
                if W_ORTH > 0 and mr.any():
                    zha, zia = z_hom[mr], z_id[mr]
                    if zha.shape[0] > 1:
                        L_orth = cross_cov_penalty(zha, zia)
                
                # Classification loss
                L_cls = bce_loss_fn(logits, y)
                
                # Identity contrastive loss (optional in Stage B)
                L_het = torch.tensor(0.0, device=self.device)
                if W_HET_B > 0 and mr.any():
                    supcon_data = build_pos_neg_from_batch(z_id[mr], ids[mr], max_pos=4, max_neg=32)
                    if supcon_data:
                        L_het = supcon_loss_fn(*supcon_data)
                
                # Total loss
                loss = W_HOM * L_hom + W_ORTH * L_orth + W_HET_B * L_het + W_CLS * L_cls
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update running stats
                bs = e.size(0)
                n += bs
                running_loss["hom"] += L_hom.item() * bs
                running_loss["orth"] += L_orth.item() * bs
                running_loss["het"] += L_het.item() * bs
                running_loss["cls"] += L_cls.item() * bs
                running_loss["total"] += loss.item() * bs
                
                if n % 100 == 0:
                    pbar.set_postfix({k: f"{v/max(1,n):.4f}" for k, v in running_loss.items()})
            
            # Average losses
            for k in running_loss:
                running_loss[k] /= max(1, n)
            
            print(f"[Stage B Epoch {epoch+1}] "
                  f"hom={running_loss['hom']:.4f} orth={running_loss['orth']:.4f} "
                  f"het={running_loss['het']:.4f} cls={running_loss['cls']:.4f} "
                  f"total={running_loss['total']:.4f}")
            
            # Optional: validate after each epoch
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, model)
                print(f"[Stage B Epoch {epoch+1}] val_bce={val_loss:.4f}")
    
    def _validate_epoch(self, val_loader, model):
        """Compute validation BCE loss"""
        model.eval()
        total_loss = 0.0
        n = 0
        bce_loss_fn = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for batch in val_loader:
                e = batch["e"].to(self.device)
                y = batch["y"].to(self.device)
                
                e2 = model.adapter(e)
                z_hom = model.f_hom(e2)
                logits = model.classifier(z_hom)
                
                loss = bce_loss_fn(logits, y)
                total_loss += loss.item() * e.size(0)
                n += e.size(0)
        
        model.train()
        return total_loss / max(1, n)