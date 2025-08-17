"""
Setup script for the Unified Drug-Target Affinity (DTA) Prediction System
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read version from __init__.py
def get_version():
    init_file = Path("unified_dta/__init__.py")
    if init_file.exists():
        content = init_file.read_text()
        match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
    return "0.1.0"

# Read long description from README
def get_long_description():
    readme_file = Path("README.md")
    if readme_file.exists():
        return readme_file.read_text(encoding="utf-8")
    return ""

# Read requirements
def get_requirements():
    req_file = Path("requirements.txt")
    if req_file.exists():
        return [line.strip() for line in req_file.read_text().splitlines() 
                if line.strip() and not line.startswith("#")]
    return []

setup(
    name="unified-dta",
    version=get_version(),
    author="Unified DTA Team",
    author_email="contact@unified-dta.org",
    description="A comprehensive platform for drug-target affinity prediction and drug generation",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/unified-dta/unified-dta-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "visualization": [
            "matplotlib>=3.3",
            "seaborn>=0.11",
            "plotly>=5.0",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "unified-dta=unified_dta.cli:main",
            "dta-train=unified_dta.cli:train_command",
            "dta-predict=unified_dta.cli:predict_command",
            "dta-evaluate=unified_dta.cli:evaluate_command",
        ],
    },
    include_package_data=True,
    package_data={
        "unified_dta": [
            "configs/*.yaml",
            "configs/*.json",
            "data/samples/*.csv",
        ],
    },
    zip_safe=False,
    keywords=[
        "drug-target affinity",
        "molecular property prediction",
        "graph neural networks",
        "protein language models",
        "ESM-2",
        "GIN",
        "deep learning",
        "bioinformatics",
        "cheminformatics",
        "drug discovery",
    ],
    project_urls={
        "Bug Reports": "https://github.com/unified-dta/unified-dta-system/issues",
        "Source": "https://github.com/unified-dta/unified-dta-system",
        "Documentation": "https://unified-dta.readthedocs.io/",
    },
)