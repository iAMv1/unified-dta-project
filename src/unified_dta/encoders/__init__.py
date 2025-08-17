"""
Encoder implementations for proteins and drugs
"""

from .protein_encoders import ESMProteinEncoder, EnhancedCNNProteinEncoder
from .drug_encoders import EnhancedGINDrugEncoder as GINDrugEncoder, EnhancedGINDrugEncoder, MultiScaleGINEncoder
from .fusion import MultiModalFusion, CrossAttentionFusion

__all__ = [
    # Protein encoders
    "ESMProteinEncoder",
    "EnhancedCNNProteinEncoder",
    
    # Drug encoders  
    "GINDrugEncoder",
    "EnhancedGINDrugEncoder",
    "MultiScaleGINEncoder",
    
    # Fusion mechanisms
    "MultiModalFusion",
    "CrossAttentionFusion"
]