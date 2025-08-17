"""
Training script for drug generation model
Implements training pipeline for protein-conditioned SMILES generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import warnings

# Import core modules
from core.models import ESMProteinEncoder
from core.drug_generation import (
    ProteinConditionedGenerator,
    SMILESTokenizer,
    ChemicalValidator
)
from core.generation_evaluation import GenerationEvaluationPipeline
from core.data_processing import create_data_loaders


class DrugGenerationDataset(Dataset):
    """Dataset for drug generation training"""
    
    def __init__(self, 
                 csv_file: str,
                 tokenizer: SMILESTokenizer,
                 max_protein_length: int = 200,
                 max_smiles_length: int = 128):
        
        self.tokenizer = tokenizer
        self.max_protein_length = max_protein_length
        self.max_smiles_length = max_smiles_length
        
        # Load data
        self.data = pd.read_csv(csv_file)
        
        # Filter valid SMILES
        validator = ChemicalValidator()
        valid_mask = self.data['compound_iso_smiles'].apply(validator.is_valid_smiles)
        self.data = self.data[valid_mask].reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} valid drug-protein pairs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get protein sequence and SMILES
        protein_seq = str(row['target_sequence'])[:self.max_protein_length]
        smiles = str(row['compound_iso_smiles'])
        
        # Tokenize SMILES
        smiles_tokens = self.tokenizer.encode(smiles, max_length=self.max_smiles_length)
        
        return {
            'protein_sequence': protein_seq,
            'smiles': smiles,
            'smiles_tokens': torch.tensor(smiles_tokens, dtype=torch.long),
            'affinity': float(row.get('affinity', 0.0))
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    protein_sequences = [item['protein_sequence'] for item in batch]
    smiles_list = [item['smiles'] for item in batch]
    smiles_tokens = torch.stack([item['smiles_tokens'] for item in batch])
    affinities = torch.tensor([item['affinity'] for item in batch], dtype=torch.float)
    
    return {
        'protein_sequences': protein_sequences,
        'smiles_list': smiles_list,
        'smiles_tokens': smiles_tokens,
        'affinities': affinities
    }


class GenerationTrainer:
    """Trainer for drug generation model"""
    
    def __init__(self,
                 model: ProteinConditionedGenerator,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Evaluation pipeline
        self.evaluator = GenerationEvaluationPipeline()
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            # Move to device
            smiles_tokens = batch['smiles_tokens'].to(self.device)
            protein_sequences = batch['protein_sequences']
            smiles_list = batch['smiles_list']
            
            # Forward pass
            try:
                logits, target_labels = self.model(protein_sequences, smiles_list)
                
                # Calculate loss
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_labels.reshape(-1),
                    ignore_index=self.model.tokenizer.pad_token_id
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                smiles_tokens = batch['smiles_tokens'].to(self.device)
                protein_sequences = batch['protein_sequences']
                smiles_list = batch['smiles_list']
                
                try:
                    logits, target_labels = self.model(protein_sequences, smiles_list)
                    
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        target_labels.reshape(-1),
                        ignore_index=self.model.tokenizer.pad_token_id
                    )
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate_generation(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate generation quality"""
        self.model.eval()
        
        # Sample some proteins from validation set
        sample_proteins = []
        for i, batch in enumerate(self.val_loader):
            sample_proteins.extend(batch['protein_sequences'])
            if len(sample_proteins) >= num_samples:
                break
        
        sample_proteins = sample_proteins[:num_samples]
        
        # Generate molecules
        with torch.no_grad():
            generated_smiles = self.model.generate(
                protein_sequences=sample_proteins,
                max_length=64,
                temperature=1.0,
                deterministic=False,
                num_return_sequences=1
            )
        
        # Flatten results
        flat_generated = []
        for smiles_list in generated_smiles:
            if isinstance(smiles_list, list):
                flat_generated.extend(smiles_list)
            else:
                flat_generated.append(smiles_list)
        
        # Evaluate
        results = self.evaluator.evaluate_single_model(
            generated_smiles=flat_generated,
            model_name="current_model",
            save_results=False
        )
        
        # Extract key metrics
        metrics = {}
        if 'validity' in results:
            metrics['validity_rate'] = results['validity']['validity_rate']
            metrics['uniqueness_rate'] = results['validity']['uniqueness_rate']
        
        if 'diversity' in results:
            metrics['tanimoto_diversity'] = results['diversity']['tanimoto_diversity']
        
        if 'drug_likeness' in results:
            metrics['avg_drug_likeness'] = results['drug_likeness']['avg_drug_likeness']
        
        return metrics
    
    def train(self, 
              num_epochs: int,
              save_dir: str,
              eval_frequency: int = 5,
              save_frequency: int = 10):
        """Main training loop"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path / 'best_model.pth')
                print("Saved best model!")
            
            # Evaluate generation quality
            if (epoch + 1) % eval_frequency == 0:
                print("Evaluating generation quality...")
                try:
                    gen_metrics = self.evaluate_generation()
                    print("Generation Metrics:")
                    for metric, value in gen_metrics.items():
                        print(f"  {metric}: {value:.4f}")
                except Exception as e:
                    print(f"Error in generation evaluation: {e}")
            
            # Save checkpoint
            if (epoch + 1) % save_frequency == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                }, save_path / f'checkpoint_epoch_{epoch + 1}.pth')
        
        # Save final model
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, save_path / 'final_model.pth')
        
        print("Training completed!")


def create_model(config: Dict) -> ProteinConditionedGenerator:
    """Create generation model from config"""
    
    # Create protein encoder
    protein_encoder = ESMProteinEncoder(
        output_dim=config.get('protein_output_dim', 128),
        max_length=config.get('max_protein_length', 200)
    )
    
    # Create tokenizer
    tokenizer = SMILESTokenizer()
    
    # Create generator
    generator = ProteinConditionedGenerator(
        protein_encoder=protein_encoder,
        vocab_size=len(tokenizer),
        d_model=config.get('d_model', 512),
        nhead=config.get('nhead', 8),
        num_layers=config.get('num_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        max_length=config.get('max_smiles_length', 128)
    )
    
    return generator


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Drug Generation Model")
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation CSV file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file')
    parser.add_argument('--save_dir', type=str, default='generation_models',
                       help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            'protein_output_dim': 128,
            'max_protein_length': 200,
            'd_model': 256,  # Smaller for demo
            'nhead': 8,
            'num_layers': 4,  # Fewer layers for demo
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_smiles_length': 64
        }
    
    print("Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create tokenizer
    tokenizer = SMILESTokenizer()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = DrugGenerationDataset(
        args.train_data, 
        tokenizer,
        max_protein_length=config['max_protein_length'],
        max_smiles_length=config['max_smiles_length']
    )
    
    val_dataset = DrugGenerationDataset(
        args.val_data,
        tokenizer,
        max_protein_length=config['max_protein_length'],
        max_smiles_length=config['max_smiles_length']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    
    # Create trainer
    trainer = GenerationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate
    )
    
    # Train model
    trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        eval_frequency=5,
        save_frequency=10
    )


if __name__ == "__main__":
    main()