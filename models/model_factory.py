from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    """Tiny residual MLP to lightly reshape embeddings (adapter)"""
    
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        # For compatibility with existing code, use single-layer residual
        # but allow for depth via num_layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize near identity
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, e):
        return e + self.fc2(self.act(self.fc1(e)))


class MLPHead(nn.Module):
    """MLP head for embedding transformation"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)


class BaseEmbeddingModel(ABC):
    """Base class for all embedding models"""
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def get_embeddings(self, x):
        """Return embeddings for evaluation"""
        pass

class OrthogonalModel(nn.Module, BaseEmbeddingModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
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
        """Identity head takes adapter output and produces identity embeddings"""
        # Get the adapter output dim (should be adapter_hidden_dim)
        adapter_dim = self.config.get('adapter_hidden_dim', 512)
        
        return nn.Sequential(
            nn.Linear(adapter_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.functional.normalize  # L2 normalize the final embedding
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

class DirectClassifierModel(nn.Module, BaseEmbeddingModel):
    """Simple classifier directly on raw embeddings"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
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

class BaseEmbeddingsModel(nn.Module, BaseEmbeddingModel):
    """Simple model that just returns raw embeddings"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
    
    def forward(self, x):
        return x
    
    def get_embeddings(self, x):
        return x


def create_model(model_type: str, config: Dict[str, Any]) -> BaseEmbeddingModel:
    """Factory function to create models"""
    if model_type == "orthogonal":
        return OrthogonalModel(config)
    elif model_type == "direct_classifier":
        return DirectClassifierModel(config)
    elif model_type == "base_embeddings":
        return BaseEmbeddingsModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")