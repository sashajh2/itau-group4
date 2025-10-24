from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn

class BaseEmbeddingModel(ABC):
    """Base class for all embedding models"""
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def get_embeddings(self, x):
        """Return embeddings for evaluation"""
        pass

class OrthogonalModel(BaseEmbeddingModel):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapter = self._build_adapter()
        self.f_hom = self._build_homogeneous_head()
        self.f_id = self._build_identity_head()
        self.classifier = self._build_classifier()
    
    def _build_adapter(self):
        if self.config.get('use_adapter', True):
            return ResidualMLP(
                input_dim=self.config['input_dim'],
                hidden_dim=self.config['adapter_hidden_dim'],
                num_layers=self.config['adapter_layers']
            )
        return nn.Identity()
    
    def _build_homogeneous_head(self):
        return nn.Sequential(
            nn.Linear(self.config['adapter_hidden_dim'], self.config['head_dim']),
            nn.ReLU(),
            nn.Linear(self.config['head_dim'], self.config['head_dim'])
        )
    
    def _build_identity_head(self):
        return nn.Sequential(
            nn.Linear(self.config['adapter_hidden_dim'], self.config['head_dim']),
            nn.ReLU(),
            nn.Linear(self.config['head_dim'], self.config['head_dim'])
        )
    
    def _build_classifier(self):
        return nn.Sequential(
            nn.Linear(self.config['head_dim'], self.config['classifier_hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['classifier_hidden_dim'], 1)
        )
    
    def forward(self, x):
        z = self.adapter(x)
        z_hom = F.normalize(self.f_hom(z), dim=1)
        z_id = F.normalize(self.f_id(z), dim=1)
        logits = self.classifier(z_hom)
        return z_hom, z_id, logits
    
    def get_embeddings(self, x):
        z = self.adapter(x)
        z_hom = F.normalize(self.f_hom(z), dim=1)
        z_id = F.normalize(self.f_id(z), dim=1)
        return z_hom, z_id

class DirectClassifierModel(BaseEmbeddingModel):
    """Simple classifier directly on raw embeddings"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.classifier = self._build_classifier()
    
    def _build_classifier(self):
        layers = []
        dims = [self.config['input_dim']] + self.config['classifier_dims'] + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # Don't add activation after last layer
                layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.classifier(x)
        return logits
    
    def get_embeddings(self, x):
        return x  # Return raw embeddings

# Model factory function
def create_model(model_type: str, config: Dict[str, Any]) -> BaseEmbeddingModel:
    """Factory function to create models"""
    if model_type == "orthogonal":
        return OrthogonalModel(config)
    elif model_type == "direct_classifier":
        return DirectClassifierModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")