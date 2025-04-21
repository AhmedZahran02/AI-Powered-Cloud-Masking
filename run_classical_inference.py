import os
import joblib
import numpy as np
from src.data_loader import load_processed_data  # You might need a dedicated function here
from src.models.model_utils import postprocess_predictions
from src.evaluate import compute_metrics
from src.visualization import visualize_prediction

# Load model
model_path = 'outputs/models/rf_model.pkl'
model = joblib.load(model_path)

# Load data
X_test, y_test = load_processed_data(split='test', classical=True)

# Predict
preds = model.predict(X_test)

# Post-process if needed
preds_post = postprocess_predictions(preds)

# Evaluate
metrics = compute_metrics(y_test, preds_post)
print(metrics)

# Save predictions
np.save('outputs/predictions/classical_preds.npy', preds_post)

# Visualize (optional)
visualize_prediction(X_test[0], preds_post[0], y_test[0])
