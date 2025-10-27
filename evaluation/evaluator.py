# evaluation/evaluator.py
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from models.model_factory import BaseEmbeddingModel


class SimpleIdentityModel(nn.Module):
    """Simple identity model that returns raw embeddings"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x
    
    def get_embeddings(self, x):
        return x


class EmbeddingEvaluator:
    """Unified evaluation pipeline for any embedding model"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.mahalanobis_stats = None
    
    def evaluate_model(self, model, 
                      train_loader, val_loader, test_loader,
                      evaluation_config: Dict[str, Any]) -> Dict[str, float]:
        """Main evaluation function"""
        
        results = {}
        
        print("\n=== Extracting Embeddings ===")
        # Get embeddings and metadata for all splits
        train_emb, train_labels, train_ids, train_reals, train_seg_ids = self._get_embeddings_and_metadata(model, train_loader)
        val_emb, val_labels, val_ids, val_reals, val_seg_ids = self._get_embeddings_and_metadata(model, val_loader)
        test_emb, test_labels, test_ids, test_reals, test_seg_ids = self._get_embeddings_and_metadata(model, test_loader)
        
        print(f"Train: {len(train_emb)} samples, Val: {len(val_emb)} samples, Test: {len(test_emb)} samples")
        
        # Detection evaluations
        print("\n=== Detection Evaluations ===")
        
        # 1. Mahalanobis distance
        if evaluation_config.get('mahalanobis', True):
            print("Evaluating Mahalanobis distance...")
            mahal_results = self._evaluate_mahalanobis(
                train_emb, train_labels,
                val_emb, val_labels,
                test_emb, test_labels
            )
            results.update(mahal_results)
        
        # 2. Linear probe
        if evaluation_config.get('linear_probe', True):
            print("Evaluating Linear Probe...")
            linear_results = self._evaluate_linear_probe(
                train_emb, train_labels,
                val_emb, val_labels,
                test_emb, test_labels
            )
            results.update(linear_results)
        
        # 3. MLP classifier
        if evaluation_config.get('mlp_classifier', True):
            print("Evaluating MLP Classifier...")
            mlp_config = evaluation_config.get('mlp_config', {})
            mlp_results = self._evaluate_mlp_classifier(
                train_emb, train_labels,
                val_emb, val_labels,
                test_emb, test_labels,
                mlp_config
            )
            results.update(mlp_results)
        
        # Identity evaluations
        print("\n=== Identity Evaluations ===")
        
        # KNN evaluation
        if evaluation_config.get('knn', True):
            print("Evaluating KNN Identity...")
            knn_results = self._evaluate_knn(
                train_emb, train_ids,
                test_emb, test_ids
            )
            results.update(knn_results)
        
        # Note: Few-shot episodic evaluation is handled separately in trainer
        # to use the special episodic test loader
        
        return results
    
    def _get_embeddings_and_metadata(self, model, data_loader):
        """Extract embeddings, labels, and metadata from model"""
        model.eval()
        embeddings = []
        labels = []
        identities = []
        is_reals = []
        segment_ids = []
        
        with torch.no_grad():
            for batch in data_loader:
                e = batch["e"].to(self.device)
                
                if hasattr(model, 'get_embeddings'):
                    emb = model.get_embeddings(e)
                    if isinstance(emb, tuple):
                        emb = emb[0]  # Use first embedding for evaluation
                else:
                    emb = e  # Raw embeddings
                
                embeddings.append(emb.cpu().numpy())
                labels.append(batch["y"].numpy())
                identities.append(batch["id_idx"].numpy())
                is_reals.append(batch["is_real"].numpy())
                segment_ids.extend(batch["segment_id"])
        
        return (
            np.vstack(embeddings),
            np.concatenate(labels),
            np.concatenate(identities),
            np.concatenate(is_reals),
            segment_ids
        )
    
    def _evaluate_mahalanobis(self, train_emb, train_labels, val_emb, val_labels, test_emb, test_labels):
        """Mahalanobis distance evaluation"""
        # Fit Gaussian on real training samples
        real_mask = train_labels == 1
        real_embeddings = train_emb[real_mask]
        
        mean = np.mean(real_embeddings, axis=0)
        cov = np.cov(real_embeddings.T)
        cov_inv = np.linalg.pinv(cov)
        
        # Compute Mahalanobis distances
        val_distances = self._mahalanobis_distance(val_emb, mean, cov_inv)
        test_distances = self._mahalanobis_distance(test_emb, mean, cov_inv)
        
        # Compute AUC (lower distance = more real)
        val_auc = roc_auc_score(val_labels, -val_distances)
        test_auc = roc_auc_score(test_labels, -test_distances)
        
        return {
            'mahalanobis_val_auc': val_auc,
            'mahalanobis_test_auc': test_auc
        }
    
    def _evaluate_linear_probe(self, train_emb, train_labels, val_emb, val_labels, test_emb, test_labels):
        """Linear probe evaluation"""
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(train_emb, train_labels)
        
        val_pred = clf.predict_proba(val_emb)[:, 1]
        test_pred = clf.predict_proba(test_emb)[:, 1]
        
        val_auc = roc_auc_score(val_labels, val_pred)
        test_auc = roc_auc_score(test_labels, test_pred)
        
        return {
            'linear_probe_val_auc': val_auc,
            'linear_probe_test_auc': test_auc
        }
    
    def _evaluate_mlp_classifier(self, train_emb, train_labels, val_emb, val_labels, test_emb, test_labels, config):
        """MLP classifier evaluation"""
        # Create MLP model
        mlp_model = self._create_mlp_model(train_emb.shape[1], config)
        
        # Train MLP
        self._train_mlp(mlp_model, train_emb, train_labels, val_emb, val_labels)
        
        # Evaluate
        val_pred = self._predict_mlp(mlp_model, val_emb)
        test_pred = self._predict_mlp(mlp_model, test_emb)
        
        val_auc = roc_auc_score(val_labels, val_pred)
        test_auc = roc_auc_score(test_labels, test_pred)
        
        return {
            'mlp_val_auc': val_auc,
            'mlp_test_auc': test_auc
        }
    
    def _evaluate_knn(self, train_emb, train_ids, test_emb, test_ids):
        """KNN evaluation for identity"""
        knn1 = KNeighborsClassifier(n_neighbors=1)
        knn5 = KNeighborsClassifier(n_neighbors=5)
        
        knn1.fit(train_emb, train_ids)
        knn5.fit(train_emb, train_ids)
        
        test_acc1 = knn1.score(test_emb, test_ids)
        test_acc5 = knn5.score(test_emb, test_ids)
        
        print(f"KNN @1: {test_acc1:.4f}, KNN @5: {test_acc5:.4f}")
        
        return {
            'knn_test_acc_1': test_acc1,
            'knn_test_acc_5': test_acc5
        }
    
    def _evaluate_few_shot_episodic(self, train_loader, test_loader, model):
        """Evaluate using 5-way 5-shot with 15 queries"""
        model.eval()
        correct = 0
        total = 0
        
        # Only evaluate on episodic data (test_loader should be episodic)
        pbar = tqdm(test_loader, desc="Few-shot episodic evaluation")
        for batch in pbar:
            e = batch["e"].to(self.device)
            id_idx = batch["id_idx"].numpy()
            
            # Get embeddings
            if hasattr(model, 'get_embeddings'):
                emb = model.get_embeddings(e)
                if isinstance(emb, tuple):
                    emb = emb[0]  # Use first embedding
            else:
                emb = e
            
            # Move to numpy
            emb_np = emb.cpu().numpy()
            
            # Episodic data structure: 5 classes × 20 samples per class = 100 total
            # Need to parse which samples belong to which class based on id_idx
            if len(emb_np) == 100:
                unique_classes = np.unique(id_idx)
                n_way = len(unique_classes)
                
                if n_way == 5:  # Expected 5-way
                    # Group by class
                    class_samples = {}
                    for idx, class_id in enumerate(id_idx):
                        if class_id not in class_samples:
                            class_samples[class_id] = []
                        class_samples[class_id].append(emb_np[idx])
                    
                    # Compute prototypes (mean of all samples for each class)
                    prototypes = []
                    class_ids_list = []
                    for class_id, samples in class_samples.items():
                        prototype = np.array(samples).mean(axis=0)
                        prototypes.append(prototype)
                        class_ids_list.append(class_id)
                    
                    # For each class, classify its samples
                    for class_id, samples in class_samples.items():
                        for sample in samples:
                            # Compute cosine similarity to all prototypes
                            similarities = []
                            for proto in prototypes:
                                sim = np.dot(sample, proto) / (np.linalg.norm(sample) * np.linalg.norm(proto))
                                similarities.append(sim)
                            
                            predicted_class_idx = np.argmax(similarities)
                            predicted_class = class_ids_list[predicted_class_idx]
                            
                            if predicted_class == class_id:
                                correct += 1
                            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"Few-Shot (5-way 5-shot + 15 queries) Accuracy: {accuracy:.4f}")
        
        return {
            'few_shot_acc': accuracy
        }
    
    def _mahalanobis_distance(self, x, mean, cov_inv):
        """Compute Mahalanobis distance"""
        diff = x - mean
        return np.sqrt(np.diag(diff @ cov_inv @ diff.T))
    
    def _create_mlp_model(self, input_dim, config):
        """Create MLP model"""
        hidden_dims = config.get('hidden_dims', [256, 128])
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers).to(self.device)
    
    def _train_mlp(self, model, train_emb, train_labels, val_emb, val_labels):
        """Train MLP model"""
        model.train()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        train_emb_t = torch.FloatTensor(train_emb).to(self.device)
        train_labels_t = torch.FloatTensor(train_labels).to(self.device)
        
        # Simple training loop
        for epoch in range(10):
            optimizer.zero_grad()
            logits = model(train_emb_t)
            loss = criterion(logits.squeeze(), train_labels_t)
            loss.backward()
            optimizer.step()
    
    def _predict_mlp(self, model, embeddings):
        """Predict with MLP model"""
        model.eval()
        with torch.no_grad():
            emb_t = torch.FloatTensor(embeddings).to(self.device)
            logits = model(emb_t)
            probs = torch.sigmoid(logits.squeeze())
        return probs.cpu().numpy()