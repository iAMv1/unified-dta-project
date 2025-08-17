"""
Encoder implementations for proteins and drugs.

This module provides various encoding strategies for protein sequences
and drug molecular graphs, along with fusion mechanisms.
"""

from .protein_encoders import ESMProteinEncoder, CNNProteinEncoder
from .drug_encoders import GINDrugEncoder
from .fusion import MultiModalFusion

__all__ = [
    'ESMProteinEncoder',
    'CNNProteinEncoder',
    'GINDrugEncoder', 
    'MultiModalFusion',
]