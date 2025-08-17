"""
Configuration classes for utils module
"""

# Import the main DTAConfig from core to avoid duplication
try:
    from ..core.config import DTAConfig, ProteinConfig, DrugConfig, FusionConfig, PredictorConfig, TrainingConfig, DataConfig
except ImportError:
    # Fallback definitions if core config is not available
    from dataclasses import dataclass
    from typing import List, Optional
    
    @dataclass
    class ProteinConfig:
        output_dim: int = 128
        max_length: int = 200
        model_name: str = "facebook/esm2_t6_8M_UR50D"
        freeze_initial: bool = True
        vocab_size: int = 25
        embed_dim: int = 128
        num_filters: int = 32
        kernel_size: int = 8

    @dataclass
    class DrugConfig:
        output_dim: int = 128
        node_features: int = 78
        hidden_dim: int = 128
        num_layers: int = 5
        dropout: float = 0.2
        use_batch_norm: bool = True

    @dataclass
    class FusionConfig:
        hidden_dim: int = 256
        num_heads: int = 8

    @dataclass
    class PredictorConfig:
        hidden_dims: List[int] = None
        dropout: float = 0.3
        activation: str = 'relu'
        
        def __post_init__(self):
            if self.hidden_dims is None:
                self.hidden_dims = [512, 256]

    @dataclass
    class TrainingConfig:
        batch_size: int = 4
        learning_rate_phase1: float = 1e-3
        learning_rate_phase2: float = 1e-4
        num_epochs_phase1: int = 50
        num_epochs_phase2: int = 30
        weight_decay: float = 1e-5
        early_stopping_patience: int = 10
        checkpoint_interval: int = 5
        gradient_clip_norm: float = 1.0
        max_memory_mb: float = 4000
        enable_gradient_checkpointing: bool = True
        memory_monitoring_interval: int = 10
        aggressive_memory_cleanup: bool = False

    @dataclass
    class DataConfig:
        datasets: List[str] = None
        data_dir: str = "data"
        max_protein_length: int = 200
        validation_split: float = 0.1
        test_split: float = 0.1
        num_workers: int = 4
        pin_memory: bool = True
        
        def __post_init__(self):
            if self.datasets is None:
                self.datasets = ["kiba", "davis", "bindingdb"]

    @dataclass
    class DTAConfig:
        protein_encoder_type: str = 'esm'
        drug_encoder_type: str = 'gin'
        use_fusion: bool = True
        protein_config: ProteinConfig = None
        drug_config: DrugConfig = None
        fusion_config: FusionConfig = None
        predictor_config: PredictorConfig = None
        training_config: TrainingConfig = None
        data_config: DataConfig = None
        device: str = 'auto'
        seed: int = 42
        verbose: bool = True
        log_level: str = 'INFO'
        
        def __post_init__(self):
            if self.protein_config is None:
                self.protein_config = ProteinConfig()
            if self.drug_config is None:
                self.drug_config = DrugConfig()
            if self.fusion_config is None:
                self.fusion_config = FusionConfig()
            if self.predictor_config is None:
                self.predictor_config = PredictorConfig()
            if self.training_config is None:
                self.training_config = TrainingConfig()
            if self.data_config is None:
                self.data_config = DataConfig()