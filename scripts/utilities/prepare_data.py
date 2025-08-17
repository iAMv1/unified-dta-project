"""
Data Preparation Script for Combined DTA Model
==============================================

This script prepares datasets from both DeepDTAGen and DoubleSG-DTA repositories.
"""

import os
import pandas as pd
import shutil
from pathlib import Path


def check_repositories():
    """Check if both repositories are available"""
    repos = {
        'DeepDTAGen': 'DeepDTAGen',
        'DoubleSG-DTA': 'DoubleSG-DTA'
    }
    
    missing = []
    for name, path in repos.items():
        if not os.path.exists(path):
            missing.append(name)
    
    if missing:
        print(f"❌ Missing repositories: {', '.join(missing)}")
        print("Please clone them first:")
        if 'DeepDTAGen' in missing:
            print("  git clone https://github.com/CSUBioGroup/DeepDTAGen.git")
        if 'DoubleSG-DTA' in missing:
            print("  git clone https://github.com/YongtaoQian/DoubleSG-DTA.git")
        return False
    
    print("✅ Both repositories found")
    return True


def prepare_datasets():
    """Prepare datasets from DoubleSG-DTA repository"""
    
    # Check if DoubleSG-DTA data exists
    data_dir = Path("DoubleSG-DTA/data")
    if not data_dir.exists():
        print("❌ DoubleSG-DTA data directory not found")
        return False
    
    # Expected dataset files
    datasets = {
        'kiba': ['kiba_train.csv', 'kiba_test.csv'],
        'davis': ['davis_train.csv', 'davis_test.csv'],
        'bindingdb': ['bindingdb_train.csv', 'bindingdb_test.csv']
    }
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    print("📊 Preparing datasets...")
    
    for dataset_name, files in datasets.items():
        print(f"\n--- {dataset_name.upper()} Dataset ---")
        
        for file_name in files:
            source_path = data_dir / file_name
            dest_path = output_dir / file_name
            
            if source_path.exists():
                # Copy file
                shutil.copy2(source_path, dest_path)
                
                # Check file content
                try:
                    df = pd.read_csv(dest_path)
                    print(f"✅ {file_name}: {len(df)} samples")
                    
                    # Show required columns
                    required_cols = ['compound_iso_smiles', 'target_sequence', 'affinity']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        print(f"⚠️  Missing columns: {missing_cols}")
                        print(f"   Available columns: {list(df.columns)}")
                    else:
                        print(f"✅ All required columns present")
                        
                except Exception as e:
                    print(f"❌ Error reading {file_name}: {e}")
            else:
                print(f"❌ {file_name} not found")
    
    return True


def create_sample_data():
    """Create small sample datasets for testing"""
    
    print("\n📝 Creating sample datasets for testing...")
    
    # Check if we have any real data
    data_dir = Path("data")
    sample_dir = Path("data/samples")
    sample_dir.mkdir(exist_ok=True)
    
    datasets = ['kiba_train.csv', 'davis_train.csv']
    
    for dataset_file in datasets:
        source_path = data_dir / dataset_file
        
        if source_path.exists():
            try:
                df = pd.read_csv(source_path)
                
                # Create small sample (100 rows)
                sample_df = df.head(100)
                
                # Save sample
                sample_path = sample_dir / f"sample_{dataset_file}"
                sample_df.to_csv(sample_path, index=False)
                
                print(f"✅ Created {sample_path} with {len(sample_df)} samples")
                
            except Exception as e:
                print(f"❌ Error creating sample for {dataset_file}: {e}")
        else:
            print(f"⚠️  {dataset_file} not found, skipping sample creation")


def verify_data_format():
    """Verify that data has the correct format for our model"""
    
    print("\n🔍 Verifying data format...")
    
    data_dir = Path("data")
    test_files = ['kiba_train.csv', 'davis_train.csv']
    
    for file_name in test_files:
        file_path = data_dir / file_name
        
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                
                print(f"\n--- {file_name} ---")
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                
                # Check required columns
                required_cols = ['compound_iso_smiles', 'target_sequence', 'affinity']
                
                for col in required_cols:
                    if col in df.columns:
                        print(f"✅ {col}: {df[col].dtype}")
                        
                        # Show sample values
                        if col == 'compound_iso_smiles':
                            print(f"   Sample SMILES: {df[col].iloc[0][:50]}...")
                        elif col == 'target_sequence':
                            print(f"   Sample protein: {df[col].iloc[0][:50]}...")
                        elif col == 'affinity':
                            print(f"   Affinity range: {df[col].min():.2f} to {df[col].max():.2f}")
                    else:
                        print(f"❌ Missing column: {col}")
                        
            except Exception as e:
                print(f"❌ Error verifying {file_name}: {e}")


def main():
    """Main data preparation function"""
    
    print("🚀 Combined DTA Data Preparation")
    print("=" * 40)
    
    # Step 1: Check repositories
    if not check_repositories():
        return
    
    # Step 2: Prepare datasets
    if not prepare_datasets():
        print("❌ Failed to prepare datasets")
        return
    
    # Step 3: Create sample data
    create_sample_data()
    
    # Step 4: Verify data format
    verify_data_format()
    
    print("\n✅ Data preparation completed!")
    print("\nNext steps:")
    print("1. Run: python simple_demo.py  (works immediately)")
    print("2. Run: python train_combined.py  (uses prepared data)")


if __name__ == "__main__":
    main()
