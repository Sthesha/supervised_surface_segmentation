# data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

class DataLoader:
    """Handles loading and preprocessing of morphological segmentation data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config['model_name']
    
    def load_data(self):
        """Load and split data from CSV file"""
        try:
            # Read the CSV and print its shape first
            df = pd.read_csv(self.config['data_path'])
            print(f"Original dataset shape: {df.shape}")
            
            # Get actual column names
            actual_columns = df.columns.tolist()
            print(f"Available columns: {actual_columns}")
            
            # Verify that we have the required columns
            if 'tokens' not in actual_columns or self.model_name not in actual_columns:
                raise ValueError(f"Required columns not found. Need 'tokens' and '{self.model_name}'")
            
            # Select only the columns we need for this experiment
            df = df[['tokens', self.model_name]]
            print(f"\nUsing columns for {self.model_name}:")
            print(df.head(2))
            print()
            
            # Sample data if specified
            if self.config['data_length'] is not None:
                if self.config['data_length'] > len(df):
                    print(f"Requested sample size ({self.config['data_length']}) exceeds available data ({len(df)})")
                    print("Using full dataset instead.")
                else:
                    df = df.sample(n=self.config['data_length'], random_state=self.config['random_seed'])
                    print(f"Cut-off dataset shape: {df.shape}")
            
            # Preprocess data
            df = self._preprocess_dataframe(df)
            
            if len(df) == 0:
                raise ValueError("No valid data remains after preprocessing")
            
            # Split data using config ratios
            train_ratio = self.config['train_ratio']
            val_ratio = self.config['validation_ratio']
            test_ratio = self.config['test_ratio']
            
            # Verify ratios sum to 1
            total_ratio = train_ratio + val_ratio + test_ratio
            if not 0.99 <= total_ratio <= 1.01:  # Allow for small floating point differences
                raise ValueError(f"Split ratios must sum to 1, got {total_ratio}")
            
            # First split off the test set
            train_val_data, test_data = train_test_split(
                df, 
                test_size=test_ratio,
                random_state=self.config['random_seed']
            )
            
            # Then split the remaining data into train and validation
            remaining_ratio = val_ratio / (train_ratio + val_ratio)
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=remaining_ratio,
                random_state=self.config['random_seed']
            )
            
            # Create data dictionaries
            processed_data = {
                'train': {
                    'tokens_labels': dict(zip(train_data['tokens'], train_data['bmes_labels'])),
                    'segments': dict(zip(train_data['tokens'], train_data[self.model_name]))
                },
                'dev': {
                    'tokens_labels': dict(zip(val_data['tokens'], val_data['bmes_labels'])),
                    'segments': dict(zip(val_data['tokens'], val_data[self.model_name]))
                },
                'test': {
                    'tokens_labels': dict(zip(test_data['tokens'], test_data['bmes_labels'])),
                    'segments': dict(zip(test_data['tokens'], test_data[self.model_name]))
                }
            }
            
            print("\nData split statistics:")
            print(f"  Training samples: {len(processed_data['train']['tokens_labels'])}")
            print(f"  Development samples: {len(processed_data['dev']['tokens_labels'])}")
            print(f"  Test samples: {len(processed_data['test']['tokens_labels'])}")
            
            return processed_data
            
        except Exception as e:
            print(f"Error details: {str(e)}")
            raise ValueError(f"Error reading CSV file: {str(e)}")

    def _get_unsegmented_token(self, segmented):
        """Convert segmented form to unsegmented token"""
        return ''.join(segmented.split('-'))
    
    def _generate_bmes_labels(self, row):
        """Generate BMES labels for a token"""
        segmented = row[self.model_name]
        token = self._get_unsegmented_token(segmented)
        segments = segmented.split('-')
        
        labels = ''
        for segment in segments:
            if len(segment) == 0:
                continue
            elif len(segment) == 1:
                labels += 'S'
            else:
                labels += 'B' + 'M' * (len(segment) - 2) + 'E'
        
        if len(labels) != len(token):
            warnings.warn(f"Length mismatch: {segmented}")
            return None
            
        return labels

    def _preprocess_dataframe(self, df):
        """Clean and preprocess the input dataframe"""
        df = df.copy()
        df[self.model_name] = df[self.model_name].str.strip()
        df['tokens'] = df[self.model_name].apply(self._get_unsegmented_token)
        df['bmes_labels'] = df.apply(self._generate_bmes_labels, axis=1)
        return df.dropna(subset=['bmes_labels'])