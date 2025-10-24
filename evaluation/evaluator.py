# evaluation/evaluator.py
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class EmbeddingEvaluator:
    """Unified evaluation pipeline for any embedding model"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.mahalanobis_stats = None
    
    def evaluate_model(self, model: BaseEmbeddingModel, 
                      train_loader, val_loader, test_loader,
                      evaluation_config: Dict[str, Any]) -> Dict[str, float]:
        """Main evaluation function"""
        
        results = {}
        
        # Get embeddings for all splits
        train_embeddings, train_labels = self._get_embeddings(model, train_loader)
        val_embeddings, val_labels = self._get_embeddings(model, val_loader)
        test_embeddings, test_labels = self._get_embeddings(model, test_loader)
        
        # Mahalanobis evaluation
        if evaluation_config.get('mahalanobis', True):
            mahal_results = self._evaluate_mahalanobis(
                train_embeddings, train_labels,
                val_embeddings, val_labels,
                test_embeddings, test_labels
            )
            results.update(mahal_results)
        
        # Linear probe evaluation
        if evaluation_config.get('linear_probe', True):
            linear_results = self._evaluate_linear_probe(
                train_embeddings, train_labels,
                val_embeddings, val_labels,
                test_embeddings, test_labels
            )
            results.update(linear_results)
        
        # MLP classifier evaluation
        if evaluation_config.get('mlp_classifier', True):
            mlp_config = evaluation_config.get('mlp_config', {})
            mlp_results = self._evaluate_mlp_classifier(
                train_embeddings, train_labels,
                val_embeddings, val_labels,
                test_embeddings, test_labels,
                mlp_config
            )
            results.update(mlp_results)
        
        # KNN evaluation
        if evaluation_config.get('knn', True):
            knn_results = self._evaluate_knn(
                train_embeddings, train_labels,
                test_embeddings, test_labels
            )
            results.update(knn_results)
        
        return results
    
    def _get_embeddings(self, model, data_loader):
        """Extract embeddings from model"""
        model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                x, y = batch
                x = x.to(self.device)
                
                if hasattr(model, 'get_embeddings'):
                    emb = model.get_embeddings(x)
                    if isinstance(emb, tuple):
                        emb = emb[0]  # Use z_hom for evaluation
                else:
                    emb = x  # Raw embeddings
                
                embeddings.append(emb.cpu().numpy())
                labels.append(y.numpy())
        
        return np.vstack(embeddings), np.concatenate(labels)
    
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
    
    def _evaluate_knn(self, train_emb, train_labels, test_emb, test_labels):
        """KNN evaluation for identity"""
        # Extract identity labels (you'll need to modify this based on your data)
        train_identities = self._extract_identities(train_labels)  # Implement this
        test_identities = self._extract_identities(test_labels)
        
        knn1 = KNeighborsClassifier(n_neighbors=1)
        knn5 = KNeighborsClassifier(n_neighbors=5)
        
        knn1.fit(train_emb, train_identities)
        knn5.fit(train_emb, train_identities)
        
        test_acc1 = knn1.score(test_emb, test_identities)
        test_acc5 = knn5.score(test_emb, test_identities)
        
        return {
            'knn_test_acc_1': test_acc1,
            'knn_test_acc_5': test_acc5
        }