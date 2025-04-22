import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import joblib
from src.evaluate import calculate_metrics
from src.config import config


class EnsembleModel:
    def __init__(self):
        self.model = None
        self.feature_selector = None
        self.scaler = None
        self.selected_features = None
        self.feature_importances = None
    
    def _create_ensemble(self, class_weights=None):
        """Create a voting ensemble of classifiers"""
        # SGD Classifier
        sgd = SGDClassifier(**{
            **config.SGD_PARAMS,
            'class_weight': class_weights
        })
        
        # Random Forest
        rf = RandomForestClassifier(**{
            **config.RF_PARAMS,
            'class_weight': class_weights
        })
        
        # Linear SVM (calibrated for probability outputs)
        svm = CalibratedClassifierCV(
            LinearSVC(**{
                **config.SVM_PARAMS, 
                'class_weight': class_weights
            }),
            cv=3
        )
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('sgd', sgd),
                ('rf', rf),
                ('svm', svm)
            ],
            voting='soft',  # Use probability estimates
            n_jobs=-1
        )
        
        return ensemble
    
    def select_features(self, X, y):
        """Perform feature selection"""
        print("Performing feature selection...")
        
        # Initialize and fit a RandomForest for feature importance
        rf_selector = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
        rf_selector.fit(X, y)
        
        # Get feature importances
        importances = rf_selector.feature_importances_
        self.feature_importances = importances
        
        # Select top features
        selector = SelectFromModel(
            rf_selector,
            threshold=-np.inf,  # Keep all features initially
            prefit=True,
            max_features=config.MAX_FEATURES
        )
        
        X_selected = selector.transform(X)
        self.feature_selector = selector
        
        # Record selected feature indices
        self.selected_features = selector.get_support(indices=True)
        
        print(f"Selected {len(self.selected_features)} features out of {X.shape[1]}")
        return X_selected
    
    def fit(self, train_gen_func, val_gen_func=None, class_weights=None):
        """Train the model with early stopping"""
        print("Starting model training...")
        
        # Get initial batch of data for feature selection and scaling
        for X_train, y_train in train_gen_func():
            break
        
        # Initial preprocessing
        print(f"Initial data shape: {X_train.shape}")
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        
        # Feature selection
        if config.FEATURE_SELECTION:
            X_train = self.select_features(X_train, y_train)
        
        # Create model
        print("Creating model...")
        self.model = self._create_ensemble(class_weights)
        
        # Training loop
        best_f1 = 0
        no_improvement_count = 0
        val_metrics_history = []
        
        for epoch in range(config.N_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{config.N_EPOCHS}")
            
            # Training phase
            batch_count = 0
            for X_batch, y_batch in tqdm(train_gen_func(), desc="Training"):
                # Apply preprocessing
                X_batch = self.scaler.transform(X_batch)
                if config.FEATURE_SELECTION:
                    X_batch = self.feature_selector.transform(X_batch)
                
                # Partial fit for incremental learning
                if hasattr(self.model, 'partial_fit'):
                    self.model.partial_fit(X_batch, y_batch, classes=[0, 1])
                else:  # For non-incremental models, accumulate data and fit once
                    if batch_count == 0:
                        X_accumulated = X_batch
                        y_accumulated = y_batch
                    else:
                        X_accumulated = np.vstack([X_accumulated, X_batch])
                        y_accumulated = np.hstack([y_accumulated, y_batch])
                
                batch_count += 1
            
            # For non-incremental models, fit on accumulated data
            if not hasattr(self.model, 'partial_fit') and batch_count > 0:
                print(f"Fitting model on {len(y_accumulated)} samples...")
                self.model.fit(X_accumulated, y_accumulated)
            
            print(f"Trained on {batch_count} batches")
            
            # Validation phase
            if val_gen_func:
                val_preds, val_true, val_proba = [], [], []
                
                for X_val, y_val in val_gen_func():
                    # Apply preprocessing
                    X_val = self.scaler.transform(X_val)
                    if config.FEATURE_SELECTION:
                        X_val = self.feature_selector.transform(X_val)
                    
                    # Make predictions
                    val_preds.extend(self.model.predict(X_val))
                    val_true.extend(y_val)
                    
                    # Get probabilities if available
                    if hasattr(self.model, 'predict_proba'):
                        val_proba.extend(self.model.predict_proba(X_val)[:, 1])
                
                if val_true:
                    # Calculate metrics
                    metrics = calculate_metrics(val_true, val_preds, val_proba if val_proba else None)
                    val_metrics_history.append(metrics)
                    
                    current_f1 = metrics['f1']
                    print(f"Val F1: {current_f1:.4f} | Best: {best_f1:.4f}")
                    
                    # Early stopping check
                    if current_f1 > best_f1 + config.EARLY_STOP_DELTA:
                        best_f1 = current_f1
                        no_improvement_count = 0
                        # Save best model
                        self.save(config.MODEL_PATH / "best_model.joblib")
                        print("↑ New best model saved ↑")
                    else:
                        no_improvement_count += 1
                        print(f"No improvement ({no_improvement_count}/{config.EARLY_STOP_PATIENCE})")
                        
                        if no_improvement_count >= config.EARLY_STOP_PATIENCE:
                            print(f"Early stopping triggered at epoch {epoch+1}!")
                            break
        
        # Save final model
        self.save(config.MODEL_PATH / "final_model.joblib")
        return val_metrics_history
    
    def predict(self, X):
        """Make binary predictions"""
        # Preprocess
        X = self.scaler.transform(X)
        if config.FEATURE_SELECTION and self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # Predict
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        # Preprocess
        X = self.scaler.transform(X)
        if config.FEATURE_SELECTION and self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # Predict probabilities
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            # Fall back to binary predictions
            return self.predict(X).astype(float)
    
    def save(self, path):
        """Save model and preprocessing components"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'feature_importances': self.feature_importances
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load model from file"""
        model_data = joblib.load(path)
        
        model = cls()
        model.model = model_data['model']
        model.scaler = model_data['scaler']
        model.feature_selector = model_data['feature_selector']
        model.selected_features = model_data['selected_features']
        model.feature_importances = model_data['feature_importances']
        return model