# MLOPs-Assessment
# The Artefact
# Repository Structure for ML Pipeline Project
# Created by Solomon Ejasę-Tobrisę Udele
# Student number: L00194499

# Create the main directory structure
mkdir -p ml-housing-pipeline/{src,tests,models,data,docker,scripts,.github/workflows,docs}

# Main application files
cat > ml-housing-pipeline/src/app.py << 'EOF'
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model at startup
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/housing_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for load balancer"""
    if model is None:
        return jsonify({'status': 'unhealthy', 'message': 'Model not loaded'}), 503
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not available'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Expected features for California housing dataset
        required_features = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
        
        # Validate input features
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        # Create feature array
        features = np.array([[data[feature] for feature in required_features]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'timestamp': datetime.now().isoformat(),
            'model_version': os.environ.get('MODEL_VERSION', '1.0.0')
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the current model"""
    if model is None:
        return jsonify({'error': 'Model not available'}), 503
    
    return jsonify({
        'model_type': type(model).__name__,
        'model_version': os.environ.get('MODEL_VERSION', '1.0.0'),
        'features': [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ],
        'target': 'Median House Value'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
EOF

# Model training script
cat > ml-housing-pipeline/src/train_model.py << 'EOF'
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load California housing dataset"""
    logger.info("Loading California housing dataset...")
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='MedHouseVal')
    return X, y

def train_model(X, y, model_params=None):
    """Train Random Forest model"""
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
    
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training model with parameters: {model_params}")
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    metrics = {
        'train_mse': float(mean_squared_error(y_train, train_pred)),
        'test_mse': float(mean_squared_error(y_test, test_pred)),
        'train_r2': float(r2_score(y_train, train_pred)),
        'test_r2': float(r2_score(y_test, test_pred)),
        'training_date': datetime.now().isoformat(),
        'n_samples': len(X),
        'n_features': len(X.columns)
    }
    
    logger.info(f"Model performance: R2 = {metrics['test_r2']:.4f}, MSE = {metrics['test_mse']:.4f}")
    
    return model, metrics

def save_model(model, metrics, model_dir='models'):
    """Save model and metrics"""
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'housing_model.pkl')
    metrics_path = os.path.join(model_dir, 'model_metrics.json')
    
    joblib.dump(model, model_path)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    return model_path

def main():
    """Main training pipeline"""
    try:
        X, y = load_data()
        model, metrics = train_model(X, y)
        model_path = save_model(model, metrics)
        
        # Check if model meets performance threshold
        min_r2 = float(os.environ.get('MIN_R2_SCORE', '0.7'))
        if metrics['test_r2'] < min_r2:
            raise ValueError(f"Model R2 score {metrics['test_r2']:.4f} below threshold {min_r2}")
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
EOF

# Data preprocessing utilities
cat > ml-housing-pipeline/src/data_utils.py << 'EOF'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

def validate_data(df, required_columns=None):
    """Validate input data"""
    if required_columns is None:
        required_columns = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for excessive missing values
    missing_pct = df[required_columns].isnull().sum() / len(df)
    high_missing = missing_pct[missing_pct > 0.5]
    if not high_missing.empty:
        logger.warning(f"Columns with >50% missing values: {high_missing.to_dict()}")
    
    return True

def preprocess_features(X, fit_preprocessor=True, preprocessor=None):
    """Preprocess features for training or inference"""
    if preprocessor is None and not fit_preprocessor:
        raise ValueError("Must provide preprocessor for inference")
    
    # Handle missing values
    if fit_preprocessor:
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_imputed),
            columns=X.columns,
            index=X.index
        )
        
        preprocessor = {'imputer': imputer, 'scaler': scaler}
        return X_scaled, preprocessor
    
    else:
        # Apply existing preprocessor
        X_imputed = pd.DataFrame(
            preprocessor['imputer'].transform(X),
            columns=X.columns,
            index=X.index
        )
        
        X_scaled = pd.DataFrame(
            preprocessor['scaler'].transform(X_imputed),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled

def detect_outliers(X, threshold=3):
    """Detect outliers using z-score method"""
    z_scores = np.abs(stats.zscore(X))
    outliers = (z_scores > threshold).any(axis=1)
    return outliers
EOF

# Unit tests
cat > ml-housing-pipeline/tests/test_app.py << 'EOF'
import unittest
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app
import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class TestFlaskApp(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create a dummy model for testing
        self.dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(100, 8)
        y_dummy = np.random.rand(100)
        self.dummy_model.fit(X_dummy, y_dummy)
        
        # Save dummy model
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.dummy_model, 'models/housing_model.pkl')
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists('models/housing_model.pkl'):
            os.remove('models/housing_model.pkl')
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_predict_valid_input(self):
        """Test prediction with valid input"""
        test_data = {
            'MedInc': 5.0,
            'HouseAge': 10.0,
            'AveRooms': 6.0,
            'AveBedrms': 1.2,
            'Population': 3000.0,
            'AveOccup': 3.0,
            'Latitude': 34.0,
            'Longitude': -118.0
        }
        
        response = self.app.post('/predict',
                                data=json.dumps(test_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIsInstance(data['prediction'], float)
    
    def test_predict_missing_features(self):
        """Test prediction with missing features"""
        test_data = {
            'MedInc': 5.0,
            'HouseAge': 10.0
            # Missing other required features
        }
        
        response = self.app.post('/predict',
                                data=json.dumps(test_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_predict_no_data(self):
        """Test prediction with no data"""
        response = self.app.post('/predict',
                                data='',
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = self.app.get('/model-info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('model_type', data)
        self.assertIn('features', data)

if __name__ == '__main__':
    unittest.main()
EOF

cat > ml-housing-pipeline/tests/test_training.py << 'EOF'
import unittest
import os
import sys
import tempfile
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_model import load_data, train_model, save_model
import joblib

class TestModelTraining(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.test_dir)
    
    def test_load_data(self):
        """Test data loading"""
        X, y = load_data()
        self.assertGreater(len(X), 0)
        self.assertGreater(len(y), 0)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X.columns), 8)  # 8 features in California housing
    
    def test_train_model(self):
        """Test model training"""
        X, y = load_data()
        # Use small sample for fast testing
        X_small = X.head(100)
        y_small = y.head(100)
        
        model, metrics = train_model(X_small, y_small, 
                                   {'n_estimators': 10, 'random_state': 42})
        
        self.assertIsNotNone(model)
        self.assertIn('test_r2', metrics)
        self.assertIn('test_mse', metrics)
        self.assertGreater(metrics['test_r2'], 0)
    
    def test_save_model(self):
        """Test model saving"""
        X, y = load_data()
        X_small = X.head(100)
        y_small = y.head(100)
        
        model, metrics = train_model(X_small, y_small, 
                                   {'n_estimators': 10, 'random_state': 42})
        
        model_path = save_model(model, metrics, self.test_dir)
        
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'model_metrics.json')))
        
        # Test loading saved model
        loaded_model = joblib.load(model_path)
        self.assertIsNotNone(loaded_model)

if __name__ == '__main__':
    unittest.main()
EOF

# Dockerfile
cat > ml-housing-pipeline/docker/Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mluser
RUN chown -R mluser:mluser /app
USER mluser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["python", "src/app.py"]
EOF

# Requirements file
cat > ml-housing-pipeline/requirements.txt << 'EOF'
Flask==2.3.3
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2
gunicorn==21.2.0
pytest==7.4.0
flake8==6.0.0
black==23.7.0
EOF

# GitHub Actions workflow for CI/CD
cat > ml-housing-pipeline/.github/workflows/ci-cd.yml << 'EOF'
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  SERVICE_NAME: ml-housing-api
  REGION: us-central1

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Lint with flake8
      run: |
        flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Format check with black
      run: |
        black --check src tests
    
    - name: Train model
      run: |
        cd src && python train_model.py
    
    - name: Run unit tests
      run: |
        python -m pytest tests/ -v
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: models/

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Download model artifacts
      uses: actions/download-artifact@v3
      with:
        name: model-artifacts
        path: models/
    
    - name: Set up Google Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}
    
    - name: Configure Docker
      run: gcloud auth configure-docker
    
    - name: Build Docker image
      run: |
        docker build -f docker/Dockerfile -t gcr.io/$PROJECT_ID/$SERVICE_NAME:$GITHUB_SHA .
    
    - name: Push Docker image
      run: |
        docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:$GITHUB_SHA
    
    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy $SERVICE_NAME \
          --image gcr.io/$PROJECT_ID/$SERVICE_NAME:$GITHUB_SHA \
          --platform managed \
          --region $REGION \
          --allow-unauthenticated \
          --set-env-vars MODEL_VERSION=$GITHUB_SHA
EOF

# Continuous Training workflow
cat > ml-housing-pipeline/.github/workflows/retrain.yml << 'EOF'
name: Model Retraining

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM
  workflow_dispatch:
    inputs:
      force_retrain:
        description: 'Force retrain even if performance is good'
        required: false
        default: 'false'

jobs:
  evaluate-model:
    runs-on: ubuntu-latest
    outputs:
      needs_retraining: ${{ steps.check.outputs.needs_retraining }}
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Evaluate current model
      id: check
      run: |
        # This would typically check against new data
        # For demo, we'll use a simple check
        python -c "
        import json
        import os
        
        # Load current metrics
        if os.path.exists('models/model_metrics.json'):
            with open('models/model_metrics.json') as f:
                metrics = json.load(f)
            
            # Simple check - retrain if R2 < 0.8 or forced
            needs_retrain = metrics.get('test_r2', 0) < 0.8 or '${{ github.event.inputs.force_retrain }}' == 'true'
        else:
            needs_retrain = True
        
        print(f'::set-output name=needs_retraining::{str(needs_retrain).lower()}')
        "

  retrain:
    needs: evaluate-model
    if: needs.evaluate-model.outputs.needs_retraining == 'true'
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Retrain model
      run: |
        cd src && python train_model.py