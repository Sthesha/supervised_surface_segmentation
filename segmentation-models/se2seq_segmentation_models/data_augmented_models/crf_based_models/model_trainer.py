# model_trainer.py
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os


class ModelTrainer:
    """Handles training of the morphological segmentation model"""
    
    def __init__(self, feature_extractor, crf_params: dict, file_path):
        """
        Initialize the model trainer
        
        Args:
            feature_extractor: Feature extractor instance
            crf_params: Dictionary containing CRF parameters
        """
        self.feature_extractor = feature_extractor
        self.crf_params = crf_params
        self.file_path = file_path
        self.model = None
        self.f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted')

    def train(self, training_data, dev_data=None, use_grid_search=False, grid_params=None):
        """
        Train the CRF model
        
        Args:
            training_data: Training data dictionary
            dev_data: Development data dictionary (optional)
            use_grid_search: Whether to use grid search for hyperparameter tuning
            grid_params: Grid search parameters dictionary
        """
        X_train, Y_train, _ = self.feature_extractor.extract_features(training_data)
        
        if use_grid_search and grid_params:
            print("Generating validation curves...")
            # self.plot_validation_curves(X_train, Y_train, self.file_path, grid_params)
            self.model = self._train_with_grid_search(X_train, Y_train, grid_params)
        else:
            self.model = self._train_basic(X_train, Y_train)
        
        return self.model

    
    def custom_validation_curve(self, X, y, param_name, param_range):
        """
        Custom validation curve implementation for CRF
        """
        train_scores = []
        test_scores = []
        
        # Create KFold cross validator
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # For each parameter value
        for param_value in param_range:
            # Store scores for each fold
            fold_train_scores = []
            fold_test_scores = []
            
            # For each fold
            for train_idx, test_idx in kf.split(X):
                # Create a new CRF with current parameter
                params = {
                    'algorithm': self.crf_params['algorithm'],
                    'all_possible_transitions': self.crf_params['all_possible_transitions'],
                    param_name: param_value
                }
                crf = sklearn_crfsuite.CRF(**params)
                
                # Get fold data
                X_train_fold = [X[i] for i in train_idx]
                y_train_fold = [y[i] for i in train_idx]
                X_test_fold = [X[i] for i in test_idx]
                y_test_fold = [y[i] for i in test_idx]
                
                # Train model
                crf.fit(X_train_fold, y_train_fold)
                
                # Get scores
                train_score = metrics.flat_f1_score(y_train_fold, crf.predict(X_train_fold), average='weighted')
                test_score = metrics.flat_f1_score(y_test_fold, crf.predict(X_test_fold), average='weighted')
                
                fold_train_scores.append(train_score)
                fold_test_scores.append(test_score)
            
            # Store mean scores for this parameter value
            train_scores.append(fold_train_scores)
            test_scores.append(fold_test_scores)
        
        return np.array(train_scores), np.array(test_scores)

    def plot_validation_curves(self, X_train, Y_train, save_path, grid_params):
        """Generate validation curves for model parameters"""
        # Create a figure with subplots for each parameter
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Parameter Validation Curves')
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        for ax, (param_name, param_range) in zip(axes_flat, grid_params.items()):
            print(f"Analyzing {param_name}...")
            try:
                train_scores, test_scores = self.custom_validation_curve(
                    X_train,
                    Y_train,
                    param_name,
                    param_range
                )
                
                # Calculate mean and std
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)
                
                # Plot
                ax.plot(param_range, train_mean, label='Training score')
                ax.fill_between(param_range, train_mean - train_std,
                              train_mean + train_std, alpha=0.1)
                ax.plot(param_range, test_mean, label='Cross-validation score')
                ax.fill_between(param_range, test_mean - test_std,
                              test_mean + test_std, alpha=0.1)
                
                ax.set_xlabel(param_name)
                ax.set_ylabel('Score')
                ax.legend(loc='best')
                ax.grid(True)
                
            except Exception as e:
                print(f"Error analyzing {param_name}: {str(e)}")
                continue
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save the plot
        plot_path = os.path.join(save_path, 'validation_curves.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Validation curves saved to {plot_path}")

    def _train_basic(self, X_train, Y_train):
        """Train model with parameters from config"""
        print("training _train_basic")
        crf = sklearn_crfsuite.CRF(**self.crf_params)
        crf.fit(X_train, Y_train)
        return crf
    
    def _train_with_grid_search(self, X_train, Y_train, grid_params):
        """
        Train model with grid search for hyperparameter tuning
        
        Args:
            X_train: Training features
            Y_train: Training labels
            grid_params: Dictionary of parameters to search over
        """
        base_crf = sklearn_crfsuite.CRF(
            algorithm=self.crf_params['algorithm'],
            all_possible_transitions=self.crf_params['all_possible_transitions']
        )
        
        f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted')
        
        grid_search = GridSearchCV(
            estimator=base_crf,
            param_grid=grid_params,
            scoring=f1_scorer,
            cv=5,
            verbose=1,
            n_jobs=-1,
            return_train_score=True,
            refit=True,
            error_score='raise'
        )
        
        print("ATTENTION!!!! Running grid search...")
        grid_search.fit(X_train, Y_train)
        print("Best parameters:", grid_search.best_params_)
        
        return grid_search.best_estimator_
    
    def save_model(self, path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model has been trained yet")
        
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load_model(self, path):
        """Load a trained model"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        return self.model