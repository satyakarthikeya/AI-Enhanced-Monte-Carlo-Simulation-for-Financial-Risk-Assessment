#!/usr/bin/env python3
"""
Data Loading Module for Credit Risk Assessment
===============================================

This module handles all data loading, preprocessing, and validation operations
for the XGBoost credit risk assessment system.

Functions:
- load_preprocessed_data(): Load training and test data
- clean_and_validate_data(): Handle missing values and data validation
- create_train_val_split(): Create training/validation splits
- get_data_info(): Get detailed information about the dataset
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataLoader:
    """Data loading and preprocessing class for credit risk assessment."""
    
    def __init__(self, data_dir: str = "processed_data", random_state: int = 42):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing preprocessed data files
            random_state: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Data containers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Metadata
        self.feature_names = None
        self.target_name = 'default.payment.next.month'
        self.data_info = {}
    
    def load_preprocessed_data(self, use_parquet: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed training and test data.
        
        Args:
            use_parquet: Whether to use parquet files for faster loading
            
        Returns:
            Tuple of (train_df, test_df)
        """
        self.logger.info("Loading preprocessed data...")
        
        try:
            if use_parquet and self._parquet_files_exist():
                # Load from parquet files (faster)
                train_df = pd.read_parquet(os.path.join(self.data_dir, "train", "train.parquet"))
                test_df = pd.read_parquet(os.path.join(self.data_dir, "test", "test.parquet"))
                self.logger.info("✓ Data loaded from parquet files")
            else:
                # Load from CSV files
                train_df = pd.read_csv(os.path.join(self.data_dir, "train", "train.csv"))
                test_df = pd.read_csv(os.path.join(self.data_dir, "test", "test.csv"))
                self.logger.info("✓ Data loaded from CSV files")
            
            # Store basic info
            self.data_info['train_shape'] = train_df.shape
            self.data_info['test_shape'] = test_df.shape
            self.data_info['total_features'] = len(train_df.columns) - 1  # Exclude target
            
            self.logger.info(f"✓ Training data: {train_df.shape}")
            self.logger.info(f"✓ Test data: {test_df.shape}")
            
            return train_df, test_df
            
        except FileNotFoundError as e:
            self.logger.error(f"Data files not found: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _parquet_files_exist(self) -> bool:
        """Check if parquet files exist."""
        train_parquet = os.path.join(self.data_dir, "train", "train.parquet")
        test_parquet = os.path.join(self.data_dir, "test", "test.parquet")
        return os.path.exists(train_parquet) and os.path.exists(test_parquet)
    
    def clean_and_validate_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean and validate the loaded data.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            
        Returns:
            Tuple of cleaned (train_df, test_df)
        """
        self.logger.info("Cleaning and validating data...")
        
        # Combine data for consistent cleaning
        X_train = train_df.drop(columns=[self.target_name])
        y_train = train_df[self.target_name]
        X_test = test_df.drop(columns=[self.target_name])
        y_test = test_df[self.target_name]
        
        # Combine features for consistent preprocessing
        X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        
        # Check for missing values
        missing_before = X_combined.isnull().sum().sum()
        self.logger.info(f"Missing values before cleaning: {missing_before:,}")
        
        if missing_before > 0:
            # Use median imputation for numerical features
            self.logger.info("Applying median imputation for missing values...")
            imputer = SimpleImputer(strategy='median')
            X_combined_clean = pd.DataFrame(
                imputer.fit_transform(X_combined),
                columns=X_combined.columns,
                index=X_combined.index
            )
        else:
            X_combined_clean = X_combined.copy()
        
        # Handle infinite values
        inf_count_before = np.isinf(X_combined_clean.values).sum()
        if inf_count_before > 0:
            self.logger.info(f"Handling {inf_count_before:,} infinite values...")
            X_combined_clean = X_combined_clean.replace([np.inf, -np.inf], np.nan)
            X_combined_clean = X_combined_clean.fillna(X_combined_clean.median())
        
        # Final validation
        missing_after = X_combined_clean.isnull().sum().sum()
        inf_after = np.isinf(X_combined_clean.values).sum()
        
        self.logger.info(f"✓ Missing values after cleaning: {missing_after}")
        self.logger.info(f"✓ Infinite values after cleaning: {inf_after}")
        
        # Split back into train and test
        n_train = len(X_train)
        X_train_clean = X_combined_clean.iloc[:n_train].copy()
        X_test_clean = X_combined_clean.iloc[n_train:].copy()
        
        # Recreate dataframes with target
        train_df_clean = X_train_clean.copy()
        train_df_clean[self.target_name] = y_train.values
        
        test_df_clean = X_test_clean.copy()
        test_df_clean[self.target_name] = y_test.values
        
        # Store feature names
        self.feature_names = list(X_train_clean.columns)
        
        # Update data info
        self.data_info['features_after_cleaning'] = len(self.feature_names)
        self.data_info['missing_values_handled'] = missing_before
        self.data_info['infinite_values_handled'] = inf_count_before
        
        return train_df_clean, test_df_clean
    
    def create_train_val_split(self, train_df: pd.DataFrame, val_size: float = 0.15) -> None:
        """
        Create training and validation splits.
        
        Args:
            train_df: Training dataframe
            val_size: Validation set size (as fraction of training data)
        """
        self.logger.info(f"Creating train/validation split (val_size={val_size})...")
        
        # Separate features and target
        X = train_df.drop(columns=[self.target_name])
        y = train_df[self.target_name]
        
        # Create stratified split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Log split information
        self.logger.info(f"✓ Training set: {self.X_train.shape[0]:,} samples")
        self.logger.info(f"✓ Validation set: {self.X_val.shape[0]:,} samples")
        
        # Log target distribution
        train_dist = self.y_train.value_counts().to_dict()
        val_dist = self.y_val.value_counts().to_dict()
        
        self.logger.info(f"✓ Training target distribution: {train_dist}")
        self.logger.info(f"✓ Validation target distribution: {val_dist}")
        
        # Store in data_info
        self.data_info['train_samples'] = len(self.X_train)
        self.data_info['val_samples'] = len(self.X_val)
        self.data_info['train_target_dist'] = train_dist
        self.data_info['val_target_dist'] = val_dist
    
    def set_test_data(self, test_df: pd.DataFrame) -> None:
        """
        Set the test data.
        
        Args:
            test_df: Test dataframe
        """
        self.X_test = test_df.drop(columns=[self.target_name])
        self.y_test = test_df[self.target_name]
        
        test_dist = self.y_test.value_counts().to_dict()
        self.logger.info(f"✓ Test set: {self.X_test.shape[0]:,} samples")
        self.logger.info(f"✓ Test target distribution: {test_dist}")
        
        self.data_info['test_samples'] = len(self.X_test)
        self.data_info['test_target_dist'] = test_dist
    
    def get_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Get all data splits.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if any(data is None for data in [self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test]):
            raise ValueError("Data splits not created. Call load_and_prepare_data() first.")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def get_data_info(self) -> Dict:
        """
        Get comprehensive information about the loaded data.
        
        Returns:
            Dictionary containing data information
        """
        if self.feature_names is not None:
            self.data_info['feature_names'] = self.feature_names
            self.data_info['n_features'] = len(self.feature_names)
        
        if self.X_train is not None:
            # Feature statistics
            numeric_features = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = self.X_train.select_dtypes(exclude=[np.number]).columns.tolist()
            
            self.data_info['numeric_features'] = len(numeric_features)
            self.data_info['categorical_features'] = len(categorical_features)
            
            # Data quality metrics
            self.data_info['data_quality'] = {
                'missing_percentage': (self.X_train.isnull().sum().sum() / (self.X_train.shape[0] * self.X_train.shape[1])) * 100,
                'duplicate_rows': self.X_train.duplicated().sum(),
                'constant_features': (self.X_train.nunique() == 1).sum()
            }
        
        return self.data_info.copy()
    
    def load_and_prepare_data(self, val_size: float = 0.15, use_parquet: bool = True) -> None:
        """
        Complete data loading and preparation pipeline.
        
        Args:
            val_size: Validation set size
            use_parquet: Whether to use parquet files
        """
        self.logger.info("Starting complete data loading and preparation pipeline...")
        
        # 1. Load raw data
        train_df, test_df = self.load_preprocessed_data(use_parquet=use_parquet)
        
        # 2. Clean and validate
        train_df_clean, test_df_clean = self.clean_and_validate_data(train_df, test_df)
        
        # 3. Create train/val split
        self.create_train_val_split(train_df_clean, val_size=val_size)
        
        # 4. Set test data
        self.set_test_data(test_df_clean)
        
        self.logger.info("✅ Data loading and preparation completed successfully!")
        
        # Log final summary
        info = self.get_data_info()
        self.logger.info(f"📊 Final dataset summary:")
        self.logger.info(f"   • Features: {info.get('n_features', 'N/A')}")
        self.logger.info(f"   • Training samples: {info.get('train_samples', 'N/A'):,}")
        self.logger.info(f"   • Validation samples: {info.get('val_samples', 'N/A'):,}")
        self.logger.info(f"   • Test samples: {info.get('test_samples', 'N/A'):,}")


# Utility functions for standalone usage
def load_credit_risk_data(data_dir: str = "processed_data", 
                         val_size: float = 0.15, 
                         random_state: int = 42,
                         use_parquet: bool = True) -> Tuple[DataLoader, Tuple]:
    """
    Convenience function to load and prepare credit risk data.
    
    Args:
        data_dir: Data directory path
        val_size: Validation set size
        random_state: Random seed
        use_parquet: Whether to use parquet files
    
    Returns:
        Tuple of (DataLoader instance, (X_train, X_val, X_test, y_train, y_val, y_test))
    """
    loader = DataLoader(data_dir=data_dir, random_state=random_state)
    loader.load_and_prepare_data(val_size=val_size, use_parquet=use_parquet)
    
    data_splits = loader.get_data_splits()
    return loader, data_splits


if __name__ == "__main__":
    # Example usage
    import logging
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # Load data
    loader, (X_train, X_val, X_test, y_train, y_val, y_test) = load_credit_risk_data()
    
    # Display information
    info = loader.get_data_info()
    print("\n📋 Data Information:")
    print("=" * 50)
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")