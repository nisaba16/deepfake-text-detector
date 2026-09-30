import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import copy

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.sparse import issparse

import xgboost as xgb

from scipy.stats import entropy

import matplotlib.pyplot as plt

import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class NeuralClassifier(nn.Module):
    def __init__(self, input_dim=4096, hidden_dims=[1024, 512], dropout_rate=0.2, normalization="layernorm"):
        """Simple MLP for binary classification.

        Replaces BatchNorm with LayerNorm by default to avoid crashes when the
        final mini-batch has size 1 (BatchNorm requires >=2 samples to compute
        variance during training). If you explicitly want BatchNorm you can set
        normalization="batchnorm", but LayerNorm is typically safer for smaller
        or uneven batch sizes.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.normalization = normalization.lower()

        def norm_layer(dim):
            if self.normalization == "batchnorm":
                return nn.BatchNorm1d(dim)
            elif self.normalization == "layernorm":
                return nn.LayerNorm(dim)
            else:
                return nn.Identity()

        # Input layer block
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(norm_layer(hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        # Hidden layer blocks
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(norm_layer(hidden_dims[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))

        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            # If using BatchNorm and batch size == 1, temporarily switch to eval to avoid error
            if isinstance(layer, nn.BatchNorm1d) and x.size(0) == 1 and self.training:
                layer.eval()
                x = layer(x)
                layer.train()  # restore
            else:
                x = layer(x)
        return x

class DeepBinaryDetector:
    def __init__(self, input_dim=4096, hidden_dims=[1024, 512], device='cuda', normalization="layernorm"):
        self.device = device
        self.model = NeuralClassifier(input_dim, hidden_dims, normalization=normalization).to(device)
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, embeddings, labels, validation_split=0.2, epochs=10, batch_size=32, lr=0.001, X_val=None, y_val=None):
        """Train neural network.

        If external validation tensors X_val/y_val are provided they are used; otherwise
        a split is created when validation_split>0.
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        embeddings, labels = np.asarray(embeddings), np.asarray(labels)

        # Perform internal split only if external not provided
        if X_val is None and validation_split > 0:
            embeddings, X_val, labels, y_val = train_test_split(
                embeddings, labels,
                test_size=validation_split,
                stratify=labels
            )

        # Fit the scaler on the training rows only; validation rows (internal or external) are only
        # transformed, otherwise validation loss and best-epoch selection run on differently scaled inputs
        X_train = torch.FloatTensor(self.scaler.fit_transform(embeddings)).to(self.device)
        y_train = torch.FloatTensor(labels).to(self.device)
        if X_val is not None:
            X_val = X_val.cpu().numpy() if isinstance(X_val, torch.Tensor) else np.asarray(X_val)
            X_val = torch.FloatTensor(self.scaler.transform(X_val)).to(self.device)
            y_val = y_val.cpu().numpy() if isinstance(y_val, torch.Tensor) else np.asarray(y_val)
            y_val = torch.FloatTensor(y_val).to(self.device)

        # DataLoaders
        train_dataset = TensorDataset(X_train, y_train.view(-1, 1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        if X_val is not None:
            val_dataset = TensorDataset(X_val, y_val.view(-1, 1))
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val_loss = float('inf')
        best_model = None
        val_acc = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            if X_val is not None:
                self.model.eval()
                val_loss = 0.0
                val_preds = []
                val_true = []
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        val_preds.extend((outputs > 0.5).cpu().numpy())
                        val_true.extend(batch_y.cpu().numpy())
                val_loss /= len(val_loader)
                val_acc = accuracy_score(val_true, val_preds)
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(self.model.state_dict())

        if best_model is not None:
            self.model.load_state_dict(best_model)

        self._is_fitted = True
        return {'val_accuracy': val_acc}
    
    def predict(self, embeddings, return_probabilities=True):
        """
        Predict using the trained neural network
        Args:
            embeddings: numpy array of embeddings
            return_probabilities: whether to return probabilities along with predictions
        Returns:
            predictions: binary predictions (0=real, 1=fake)
            probabilities: probability of being fake (if return_probabilities=True)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted yet!")
            
        # Preprocess embeddings
        embeddings = self.scaler.transform(embeddings)
        X = torch.FloatTensor(embeddings).to(self.device)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            probs = outputs.cpu().numpy().reshape(-1)  # 1-D even for a single text
            preds = (probs > 0.5).astype(int)
        
        # Return predictions and optionally probabilities 
        if return_probabilities:
            return preds, probs
        return preds
    
class BinaryDetector:
    """Handles binary classification with optional PCA reduction"""
    
    def __init__(self, n_components=None, contamination=0.1, random_state=42, input_dim=None):
        """
        Args:
            n_components: Optional PCA components (None for no PCA)
            contamination: Expected fraction of outliers
            random_state: Random state for reproducibility
            input_dim: Input dimension for neural classifier
        """
        self.scaler = StandardScaler()
        self.scaler_sparse = MaxAbsScaler()
        self.pca = None  # PCA for dense
        self.svd = None  # TruncatedSVD for sparse
        self.n_components = n_components
        self.random_state = random_state
        self.input_dim = input_dim  # Store raw input dimension

        # Components
        self.classifier = None
        self.real_centroid = None  # computed in preprocessed feature space
        self._is_fitted = False
        self._centroid_space = "preprocessed"  # informational
    
    def _init_classifier(self, classifier_type="svm", sparse_input: bool = False, n_features: int = None):
        """Initialize classifier (SVM or logistic regression)

        n_features: dimension of the preprocessed features (after PCA/SVD), used to size the neural net
        """
        if classifier_type == "svm":
            # RBF SVM doesn't support sparse matrices. If features are sparse, prefer LR with saga.
            if sparse_input:
                return LogisticRegression(
                    random_state=self.random_state,
                    class_weight='balanced',
                    solver='saga',
                    max_iter=2000
                )
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif classifier_type == "lr":
            # Use saga to support both dense and sparse
            return LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=2000,
                solver='saga'
            )
        elif classifier_type== "xgb":
            return xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                max_depth=7,
                learning_rate=0.01,
                n_estimators=500,
                use_label_encoder=False,
                random_state=self.random_state
            )
        elif classifier_type == "neural":
            if sparse_input:
                raise ValueError("The neural classifier needs dense features: with TF-IDF pass --svd_components "
                                 "or --tfidf_dense")
            # Size the network on the preprocessed features: PCA or SVD output, else the raw input
            if n_features is not None:
                dim = n_features
            else:
                dim = self.pca.n_components_ if self.pca is not None else self.input_dim
            return DeepBinaryDetector(
                input_dim=dim,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")
        
    def _preprocess_features(self, features, fit=False):
        """Preprocess features with scaling and dimensionality reduction.

        Dense path: StandardScaler (+ optional PCA)
        Sparse path: If n_components is set -> TruncatedSVD to dense then StandardScaler.
                     Else -> MaxAbsScaler (keeps sparse) and no PCA.
        """
        sparse = issparse(features)

        if not sparse:
            # Dense pipeline
            if fit:
                features = self.scaler.fit_transform(features)
            else:
                features = self.scaler.transform(features)

            if self.n_components is not None:
                if self.pca is None:
                    self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
                    features = self.pca.fit_transform(features)
                    print(f"PCA reduced shape: {features.shape}")
                    print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
                else:
                    features = self.pca.transform(features)
            return features
        else:
            # Sparse pipeline
            if self.n_components is not None:
                # Dimensionality reduction to dense with SVD, then scale dense
                if fit or self.svd is None:
                    self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
                    features = self.svd.fit_transform(features)
                    print(f"SVD reduced shape: {features.shape}")
                    explained = getattr(self.svd, 'explained_variance_ratio_', None)
                    if explained is not None:
                        print(f"Explained variance ratio (SVD): {explained.sum():.3f}")
                    features = self.scaler.fit_transform(features)
                else:
                    features = self.svd.transform(features)
                    features = self.scaler.transform(features)
                return features
            else:
                # Keep sparse structure; scale with MaxAbsScaler (no centering)
                if fit:
                    features = self.scaler_sparse.fit_transform(features)
                else:
                    features = self.scaler_sparse.transform(features)
                return features
    
    def fit(self, embeddings, labels, validation_split=0.2, classifier_type="svm", pca=True):
        """
        Train the classifier on embeddings
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            labels: numpy array of shape (n_samples,) with binary labels (0=real, 1=fake)
            validation_split: Fraction of data for validation (0 for no validation)
            classifier_type: Type of classifier ("svm", "lr", "xgb", "neural")
            pca: Whether to apply PCA dimensionality reduction
            
        Returns:
            training_results: Dictionary with training metrics
        """

        # Store raw input dimension
        self.input_dim = embeddings.shape[1]
        
        # Preprocess features
        features = self._preprocess_features(embeddings, fit=True)

        # Calculate centroid in PREPROCESSED space for stable distance calculations (handles sparse)
        real_mask = (labels == 0)
        feats_for_centroid = features[real_mask] if np.sum(real_mask) > 0 else features
        if issparse(feats_for_centroid):
            # Convert 1xN sparse mean to dense 1D array
            self.real_centroid = np.asarray(feats_for_centroid.mean(axis=0)).ravel()
        else:
            self.real_centroid = np.mean(feats_for_centroid, axis=0)
        
        # Train/validation split if requested
        n_samples = embeddings.shape[0]
        if validation_split > 0 and n_samples > 50:
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels,
                test_size=validation_split,
                random_state=self.random_state,
                stratify=labels
            )
        else:
            X_train, y_train = features, labels
            X_val, y_val = None, None
        
        # Train classifier
        print(f"Fitting {classifier_type.upper()} classifier...")
        sparse_input = issparse(features)
        self.classifier = self._init_classifier(classifier_type, sparse_input=sparse_input,
                                                n_features=features.shape[1])
        
        # Handle neural network training separately
        if classifier_type == "neural":
            # If we already created a validation split (X_val not None), pass it directly
            if X_val is not None:
                training_results = self.classifier.fit(
                    X_train, y_train,
                    validation_split=0.0,  # external provided
                    epochs=10,
                    batch_size=32,
                    lr=0.001,
                    X_val=X_val,
                    y_val=y_val
                )
            else:
                training_results = self.classifier.fit(
                    X_train, y_train,
                    validation_split=validation_split,
                    epochs=10,
                    batch_size=32,
                    lr=0.001
                )
            val_acc = training_results.get('val_accuracy')
            train_pred = self.classifier.predict(X_train, return_probabilities=False)
        else:
            # Standard sklearn-style classifiers
            self.classifier.fit(X_train, y_train)
            train_pred = self.classifier.predict(X_train)
            val_acc = None
        
        # Training metrics
        train_acc = accuracy_score(y_train, train_pred)
        
        # Report dimensionality info depending on reduction path
        if self.pca is not None:
            n_comps = self.pca.n_components_
        elif self.svd is not None:
            n_comps = self.svd.n_components
        else:
            n_comps = features.shape[1] if not issparse(features) else getattr(features, 'shape')[1]

        results = {
            'train_accuracy': train_acc,
            'n_samples': embeddings.shape[0],
            'n_features': embeddings.shape[1],
            'n_components': n_comps
        }
        
        print(f"Training accuracy: {train_acc:.3f}")
        print("Training Classification Report:")
        print(classification_report(y_train, train_pred, target_names=['Real', 'Fake']))
        
        # Validation metrics if available
        if X_val is not None and classifier_type != "neural":
            val_pred = self.classifier.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            results['val_accuracy'] = val_acc
            
            print(f"Validation accuracy: {val_acc:.3f}")
            print("Validation Classification Report:")
            print(classification_report(y_val, val_pred, target_names=['Real', 'Fake']))
        elif val_acc is not None:
            results['val_accuracy'] = val_acc
            print(f"Validation accuracy: {val_acc:.3f}")
        
        self._is_fitted = True
        print(f"{classifier_type.upper()} training completed!")
        
        return results
    
    def predict(self, embeddings, return_probabilities=True, return_distances=True, pca=True):
        """
        Predict labels for embeddings
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            return_probabilities: Whether to return prediction probabilities
            return_distances: Whether to return distances to real centroid
            pca: Whether to apply PCA transformation
            
        Returns:
            predictions: numpy array of predictions (0=real, 1=fake)
            probabilities: numpy array of probabilities (if requested)
            distances: numpy array of distances to real centroid (if requested)
        """
        if not self._is_fitted:
            raise ValueError("Detector not fitted! Call fit() first.")

        features = self._preprocess_features(embeddings, fit=False)
        
        # Check if classifier is neural network
        is_neural = isinstance(self.classifier, DeepBinaryDetector)
        
        # Get predictions
        if is_neural:
            # Neural network was trained on preprocessed features; reuse them here
            if return_probabilities:
                predictions, probabilities = self.classifier.predict(features, return_probabilities=True)
            else:
                predictions = self.classifier.predict(features, return_probabilities=False)
        else:
            # Standard sklearn classifiers
            predictions = self.classifier.predict(features)
            if return_probabilities:
                probabilities = self.classifier.predict_proba(features)[:, 1]  # P(fake)
        
        # Build results list
        results = [predictions]
        
        if return_probabilities:
            results.append(probabilities)
        
        if return_distances:
            # Distances in the same preprocessed space as centroid
            if issparse(features):
                feats_dense = features.toarray()
            else:
                feats_dense = features
            distances = np.linalg.norm(feats_dense - self.real_centroid, axis=1)
            results.append(distances)
        
        return results if len(results) > 1 else results[0]

    def predict_pairs(self, embeddings_1, embeddings_2, pca=True):
        """
        Predict which text in each pair is real
        
        Args:
            embeddings_1: embeddings for first texts in pairs
            embeddings_2: embeddings for second texts in pairs
            pca: Whether to apply PCA transformation
            
        Returns:
            pair_predictions: 1 if first text is real, 2 if second text is real
            details: detailed predictions for each text
        """
        # Get predictions for both sets
        pred_1, prob_1, dist_1 = self.predict(embeddings_1, pca=pca)
        pred_2, prob_2, dist_2 = self.predict(embeddings_2, pca=pca)
        
        # For each pair, choose text with higher probability of being real
        prob_real_1 = 1 - prob_1  # Convert P(fake) to P(real)
        prob_real_2 = 1 - prob_2
        
        pair_predictions = np.where(prob_real_1 > prob_real_2, 1, 2)
        
        details = []
        for i in range(len(embeddings_1)):
            details.append({
                'text1_pred': int(pred_1[i]),
                'text2_pred': int(pred_2[i]),
                'text1_prob_fake': float(prob_1[i]),
                'text2_prob_fake': float(prob_2[i]),
                'text1_prob_real': float(prob_real_1[i]),
                'text2_prob_real': float(prob_real_2[i]),
                'text1_distance': float(dist_1[i]),
                'text2_distance': float(dist_2[i]),
                'pair_prediction': int(pair_predictions[i])
            })
        
        return pair_predictions, details


    

class OutlierDetections:
    def __init__(self, detector_type="elliptic", contamination=0.1, random_state=42,n_components=0.95, use_trajectory=False):
        """
        Generalized outlier detection class

        Args:
            detector_type: one of ["elliptic", "ocsvm", "iforest"]
            contamination: expected fraction of outliers
        """
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.detector_type = detector_type
        self.contamination = contamination
        self.random_state = random_state

        if use_trajectory:
            # The trajectory detector used to be fit on broken features and was never read by predict_*
            raise NotImplementedError("use_trajectory is not supported: trajectory scores are not used in predictions")
        self.use_trajectory = use_trajectory
        self.traj_detector = None  

        self.outlier_detector = None

        # Data storage
        self.real_embeddings = None
        self.real_embeddings_scaled = None
        self.real_embeddings_pca = None
        self.real_centroid = None

        # For SVM training
        self.fake_embeddings = None
        self.fake_embeddings_scaled = None
        self.fake_embeddings_pca = None


    def _init_detector(self):
        """Factory to initialize detector based on type"""
        if self.detector_type == "elliptic":
            return EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state,
                support_fraction=0.9
            )
        elif self.detector_type == "ocsvm":
            return OneClassSVM(
                kernel="rbf",
                nu=self.contamination,
                gamma="scale"
            )
        elif self.detector_type == "iforest":
            return IsolationForest(
                contamination=self.contamination,
                n_estimators=200,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown detector_type: {self.detector_type}. Supported: 'elliptic', 'ocsvm', 'iforest'")

    def fit(self, embeddings, labels, validation_split=0.2):
        """Train the outlier detector with optional validation split.

        Notes:
        - Scaler/PCA are fit on the training split only to avoid leakage.
        - Detector is trained on the training split and evaluated on both train and val (if provided).
        """
        n_samples = embeddings.shape[0]
        print(f"Training {self.detector_type} outlier detector on {n_samples} samples...")

        # Prepare split
        do_split = validation_split and validation_split > 0 and n_samples > 50 and labels is not None
        if do_split:
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    embeddings, labels,
                    test_size=validation_split,
                    random_state=self.random_state,
                    stratify=labels if len(np.unique(labels)) > 1 else None
                )
            except Exception:
                # Fallback without stratify if labels are degenerate
                X_train, X_val, y_train, y_val = train_test_split(
                    embeddings, labels,
                    test_size=validation_split,
                    random_state=self.random_state
                )
        else:
            X_train, y_train = embeddings, labels
            X_val, y_val = None, None

        # One-class: learn the human (label 0) distribution only; AI rows are only used for evaluation.
        X_real = X_train if y_train is None else X_train[np.asarray(y_train) == 0]
        if X_real.shape[0] == 0:
            raise ValueError("No human (label 0) rows in the training split to fit the one-class detector on")
        print(f"Fitting on {X_real.shape[0]} human texts (AI texts are only used for evaluation)")

        # Fit scaler/PCA on the human training rows only
        self.real_embeddings = X_real
        self.real_embeddings_scaled = self.scaler.fit_transform(X_real)
        self.real_embeddings_pca = self.pca.fit_transform(self.real_embeddings_scaled)

        print(f"PCA reduced shape: {self.real_embeddings_pca.shape}")
        try:
            ratio_sum = float(np.nansum(self.pca.explained_variance_ratio_))
            print(f"Explained variance ratio: {ratio_sum:.3f}")
        except Exception:
            print("Explained variance ratio: N/A")

        # Centroid on train
        self.real_centroid = np.mean(self.real_embeddings_pca, axis=0)

        # Initialize and fit detector on train
        self.outlier_detector = self._init_detector()
        self.outlier_detector.fit(self.real_embeddings_pca)

        # Train predictions/accuracy on every training row (human and AI)
        X_train_pca = self.pca.transform(self.scaler.transform(X_train))
        train_raw_preds = self.outlier_detector.predict(X_train_pca)
        train_predictions = np.array([0 if pred == 1 else 1 for pred in train_raw_preds])
        train_acc = accuracy_score(y_train, train_predictions) if y_train is not None else None

        results = {
            'train_accuracy': train_acc,
            'n_samples': n_samples,
            'n_features': self.real_embeddings_pca.shape[1],
            'contamination': self.contamination
        }

        # Validation evaluation (transform with train scaler/PCA)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_pca = self.pca.transform(X_val_scaled)
            val_raw_preds = self.outlier_detector.predict(X_val_pca)
            val_predictions = np.array([0 if pred == 1 else 1 for pred in val_raw_preds])
            val_acc = accuracy_score(y_val, val_predictions)
            results['val_accuracy'] = val_acc
            # Accuracy is misleading when classes are imbalanced; AUROC of the anomaly score is not.
            if len(np.unique(y_val)) > 1:
                results['val_auroc'] = roc_auc_score(y_val, -self.outlier_detector.decision_function(X_val_pca))
                print(f"Validation AUROC: {results['val_auroc']:.3f}")
            train_acc_str = f"{train_acc:.3f}" if isinstance(train_acc, (int, float, np.floating)) else "N/A"
            val_acc_str = f"{val_acc:.3f}" if isinstance(val_acc, (int, float, np.floating)) else "N/A"
            print(f"Training accuracy: {train_acc_str} | Validation accuracy: {val_acc_str}")
        else:
            train_acc_str = f"{train_acc:.3f}" if isinstance(train_acc, (int, float, np.floating)) else "N/A"
            print(f"Training accuracy: {train_acc_str}")

        print(f"{self.detector_type.upper()} outlier detector training completed!")

        return results

    def predict(self, embeddings, return_probabilities=True, return_distances=True):
        """Predict with support for both one-class and binary approaches"""
        if self.outlier_detector is None:
            raise ValueError("Detector not fitted! Call fit() first.")
        
        # Preprocess embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        embeddings_pca = self.pca.transform(embeddings_scaled)
        
        # Get predictions based on detector type
        if self.detector_type.endswith('_synthetic'):
            # Binary classifier trained with synthetic negatives
            if self.detector_type == "neural_synthetic":
                if return_probabilities:
                    raw_preds, probabilities = self.outlier_detector.predict(embeddings_pca, return_probabilities=True)
                else:
                    raw_preds = self.outlier_detector.predict(embeddings_pca, return_probabilities=False)
            else:
                raw_preds = self.outlier_detector.predict(embeddings_pca)
                if return_probabilities and hasattr(self.outlier_detector, 'predict_proba'):
                    proba = self.outlier_detector.predict_proba(embeddings_pca)
                    probabilities = proba[:, 0]  # P(outlier) = P(class 0)
            
            predictions = np.array([0 if pred == 1 else 1 for pred in raw_preds])  # 1->0 (real), 0->1 (fake)
        else:
            # Traditional one-class methods
            raw_preds = self.outlier_detector.predict(embeddings_pca)
            predictions = np.array([0 if pred == 1 else 1 for pred in raw_preds])
        
        results = [predictions]
        
        if return_probabilities:
            if self.detector_type.endswith('_synthetic'):
                results.append(probabilities)
            else:
                # Traditional one-class: use decision function or distances -> convert to P(fake)
                if hasattr(self.outlier_detector, "decision_function"):
                    scores = self.outlier_detector.decision_function(embeddings_pca)
                    # Stable sigmoid: higher score (more normal) -> lower P(fake)
                    probabilities = 1.0 / (1.0 + np.exp(np.clip(scores, -50, 50)))
                else:
                    distances = np.linalg.norm(embeddings_pca - self.real_centroid, axis=1)
                    probabilities = 1.0 / (1.0 + np.exp(-np.clip(distances, -50, 50)))
                results.append(probabilities)
        
        if return_distances:
            distances = np.linalg.norm(embeddings_pca - self.real_centroid, axis=1)
            results.append(distances)
        
        return results if len(results) > 1 else results[0]

    def predict_pairs_from_embeddings(self, embeddings_1, embeddings_2):
        """
        Predict which embedding in each pair is real using outlier detection (embeddings API).
        
        Args:
            embeddings_1: embeddings for first items in pairs
            embeddings_2: embeddings for second items in pairs
            
        Returns:
            pair_predictions: 1 if first embedding is real, 2 if second embedding is real
            details: detailed predictions for each embedding
        """
        # Get predictions for both sets
        pred_1, prob_1, dist_1 = self.predict(embeddings_1)
        pred_2, prob_2, dist_2 = self.predict(embeddings_2)
        
        # For each pair, choose embedding with lower probability of being fake (higher probability of being real)
        prob_real_1 = 1 - prob_1  # Convert P(fake) to P(real)
        prob_real_2 = 1 - prob_2
        
        pair_predictions = np.where(prob_real_1 > prob_real_2, 1, 2)
        
        details = []
        for i in range(len(embeddings_1)):
            details.append({
                'text1_pred': int(pred_1[i]),
                'text2_pred': int(pred_2[i]),
                'text1_prob_fake': float(prob_1[i]),
                'text2_prob_fake': float(prob_2[i]),
                'text1_prob_real': float(prob_real_1[i]),
                'text2_prob_real': float(prob_real_2[i]),
                'text1_distance': float(dist_1[i]),
                'text2_distance': float(dist_2[i]),
                'pair_prediction': int(pair_predictions[i])
            })
        
        return pair_predictions, details

    def learn_real_manifold(self, real_texts, extractor_model, target_layer=-2, pooling="mean", batch_size=32,
                            max_length=512):
        """
        Train outlier detector on real texts only - texts that deviate from this manifold are considered fake.
        
        Args:
            real_texts: list of real text strings to learn the manifold from
            extractor_model: EmbeddingExtractor
            target_layer: which hidden state to use (negative counts from the last)
            pooling: pooling method of EmbeddingExtractor.get_pooled_layer_embeddings ('mean', 'max', 'last', ...)
            batch_size: batch size for processing
            max_length: maximum tokens per text
        """
        
        print(f"Learning real text manifold from {len(real_texts)} samples...")
        print(f"Using layer {target_layer} with {self.detector_type} detector")

        # One pooled vector per text from the target layer
        self.real_embeddings = extractor_model.get_pooled_layer_embeddings(
            real_texts, layer_idx=target_layer, pooling=pooling, batch_size=batch_size, max_length=max_length
        )  # shape: (num_texts, hidden_dim)

        # Check for NaN/Inf values
        n_nan = np.isnan(self.real_embeddings).sum()
        n_inf = np.isinf(self.real_embeddings).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"Warning: Found {n_nan} NaN and {n_inf} Inf values in embeddings")

        print("Embedding stats:",
            "Shape:", self.real_embeddings.shape,
            "NaNs:", n_nan,
            "Infs:", n_inf,
            "Max abs:", np.max(np.abs(self.real_embeddings)))

        # Standardize + PCA
        self.real_embeddings_scaled = self.scaler.fit_transform(self.real_embeddings)
        self.real_embeddings_pca = self.pca.fit_transform(self.real_embeddings_scaled)

        print(f"PCA reduced shape: {self.real_embeddings_pca.shape}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")

        # Centroid for distance measure
        self.real_centroid = np.mean(self.real_embeddings_pca, axis=0)

        # Initialize and fit outlier detector on real text manifold
        self.outlier_detector = self._init_detector()
        self.outlier_detector.fit(self.real_embeddings_pca)

        print(f"✓ {self.detector_type.upper()} outlier detector trained on real text manifold!")
        
                    
    def predict_texts_batch(self, texts, extractor_model, target_layer=-2, pooling="mean", batch_size=32,
                            return_probabilities=True, max_length=512):
        """
        Predict multiple texts in batch using trained outlier detector.

        Args:
            texts: list of text strings to predict
            extractor_model: EmbeddingExtractor
            target_layer: which hidden state to use (same as in learn_real_manifold)
            pooling: pooling method (same as in learn_real_manifold)
            batch_size: batch size for processing
            return_probabilities: whether to return probability estimates
            max_length: maximum tokens per text
            
        Returns:
            predictions: list[int] (0=real/inlier, 1=fake/outlier)
            probabilities: list[float] probability of being fake (if requested)
            distances: list[float] distance to real centroid
            scores: list[float] outlier scores
        """
        if self.outlier_detector is None:
            raise ValueError("Detector not trained! Call learn_real_manifold() first.")
            
        embeddings = extractor_model.get_pooled_layer_embeddings(
            texts, layer_idx=target_layer, pooling=pooling, batch_size=batch_size, max_length=max_length
        )  # shape: (num_texts, hidden_dim)

        # Transform embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        embeddings_pca = self.pca.transform(embeddings_scaled)

        # Distance to centroid
        distances = np.linalg.norm(embeddings_pca - self.real_centroid, axis=1)

        # Predictions: 1 = inlier (real), -1 = outlier (fake)
        raw_preds = self.outlier_detector.predict(embeddings_pca)
        predictions = [0 if pred == 1 else 1 for pred in raw_preds]  # Convert to 0=real, 1=fake

        # Outlier scores (higher = more normal/real)
        if hasattr(self.outlier_detector, "decision_function"):
            scores = self.outlier_detector.decision_function(embeddings_pca)
        elif hasattr(self.outlier_detector, "score_samples"):
            scores = self.outlier_detector.score_samples(embeddings_pca)
        else:
            scores = -distances  # fallback: negative distance (closer to centroid = higher score)

        results = [predictions]
        
        if return_probabilities:
            # Convert scores to probabilities using sigmoid-like transformation
            # Higher scores (more normal) -> lower probability of being fake
            prob_fake = 1 / (1 + np.exp(scores))  # Sigmoid transformation
            results.append(prob_fake.tolist())
        
        results.extend([distances.tolist(), scores.tolist()])
        
        return results if len(results) > 1 else results[0]


    def predict_text(self, text, extractor_model, target_layer=-2, pooling="mean", batch_size=32, return_probabilities=True):
        """
        Predict single text using trained outlier detector.
        
        Args:
            text: text string to predict
            extractor_model: embedding extractor
            target_layer: which transformer layer to use
            pooling: pooling method
            batch_size: batch size for processing
            return_probabilities: whether to return probability estimates
            
        Returns:
            prediction: int (0=real, 1=fake)
            probability: float probability of being fake (if requested)
            distance: float distance to real centroid
            score: float outlier score
        """
        # Use batch prediction for consistency
        results = self.predict_texts_batch(
            [text], extractor_model, target_layer, pooling, batch_size, return_probabilities
        )
        
        if return_probabilities:
            predictions, probabilities, distances, scores = results
            return predictions[0], probabilities[0], distances[0], scores[0]
        else:
            predictions, distances, scores = results
            return predictions[0], distances[0], scores[0]

    def predict_pairs(self, texts_1, texts_2, extractor_model, target_layer=-2, pooling="mean", batch_size=32):
        """
        Predict which text in each pair is real using outlier detection.
        
        Args:
            texts_1: list of first texts in pairs
            texts_2: list of second texts in pairs
            extractor_model: embedding extractor
            target_layer: which transformer layer to use
            pooling: pooling method
            batch_size: batch size for processing
            
        Returns:
            pair_predictions: list[int] (1 if first text is real, 2 if second text is real)
            details: list[dict] detailed predictions for each text
        """
        # Get predictions for both sets
        pred_1, prob_1, dist_1, score_1 = self.predict_texts_batch(
            texts_1, extractor_model, target_layer, pooling, batch_size
        )
        pred_2, prob_2, dist_2, score_2 = self.predict_texts_batch(
            texts_2, extractor_model, target_layer, pooling, batch_size
        )
        
        # For each pair, choose text with lower probability of being fake (higher probability of being real)
        pair_predictions = []
        details = []
        
        for i in range(len(texts_1)):
            prob_real_1 = 1 - prob_1[i]  # Convert P(fake) to P(real)
            prob_real_2 = 1 - prob_2[i]
            
            prediction = 1 if prob_real_1 > prob_real_2 else 2
            pair_predictions.append(prediction)
            
            details.append({
                'text1_pred': pred_1[i],
                'text2_pred': pred_2[i],
                'text1_prob_fake': prob_1[i],
                'text2_prob_fake': prob_2[i],
                'text1_prob_real': prob_real_1,
                'text2_prob_real': prob_real_2,
                'text1_distance': dist_1[i],
                'text2_distance': dist_2[i],
                'text1_score': score_1[i],
                'text2_score': score_2[i],
                'pair_prediction': prediction
            })
        
        return pair_predictions, details

    
    
class TrajectoryClassifier:
    """
    A standalone classifier that uses trajectory metrics from token embeddings
    """
    def __init__(self, classifier_type="svm", random_state=42):
        """
        Args:
            classifier_type: "svm", "rf" (random forest), or "lr" (logistic regression)
            random_state: random seed
        """
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = None
        
    def _init_classifier(self):
        """Initialize the underlying classifier"""
        if self.classifier_type == "svm":
            from sklearn.svm import SVC
            return SVC(kernel="rbf", C=1.0, probability=True, random_state=self.random_state)
        elif self.classifier_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif self.classifier_type == "lr":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")
    
    def extract_trajectory_features(self, texts, extractor_model, target_layer=-2, batch_size=32, max_length=512):
        """
        Extract trajectory features from a list of texts
        
        Args:
            texts: list of text strings
            extractor_model: EmbeddingExtractor
            target_layer: which hidden state to use (negative counts from the last)
            batch_size: batch size for processing
            max_length: maximum tokens per text
            
        Returns:
            features: np.array of shape (n_texts, n_features)
        """
        print(f"Extracting trajectory features from {len(texts)} texts...")
        
        features = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            # Token-level embeddings of every layer, one dict per text: {layer_idx: (seq, hidden)}
            batch_embeds = extractor_model.get_all_layer_embeddings(
                batch_texts, batch_size=batch_size, max_length=max_length, show_progress=False
            )
            for layer_embeds in batch_embeds:
                token_embeds = layer_embeds[sorted(layer_embeds)[target_layer]].astype(np.float32)
            
                # Compute trajectory metrics (texts under 3 tokens get zeros)
                traj_metrics = self.trajectory_metrics(token_embeds)
                    
                # Convert to feature vector
                features.append([
                    traj_metrics['mean_angle'],
                    traj_metrics['std_angle'],
                    traj_metrics['angle_entropy'],
                    traj_metrics['trajectory_smoothness'],
                    traj_metrics['max_curvature']
                ])
        
        return np.array(features)
    
    def trajectory_metrics(self, token_embeddings):
        """
        Compute turning angles between successive embedding steps.
        Same as OutlierDetections.trajectory_metrics but standalone
        """
        if len(token_embeddings) < 3:
            return {
                'mean_angle': 0.0,
                'std_angle': 0.0,
                'angle_entropy': 0.0,
                'trajectory_smoothness': 0.0,
                'max_curvature': 0.0
            }
            
        angles = []
        for i in range(len(token_embeddings) - 2):
            p1 = token_embeddings[i]
            p2 = token_embeddings[i + 1]
            p3 = token_embeddings[i + 2]

            v1 = p2 - p1
            v2 = p3 - p2

            cos_traj = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
            angle = np.arccos(np.clip(cos_traj, -1.0, 1.0))  # radians in [0, π]

            angles.append(angle)
            
        if len(angles) == 0:
            return {
                'mean_angle': 0.0,
                'std_angle': 0.0,
                'angle_entropy': 0.0,
                'trajectory_smoothness': 0.0,
                'max_curvature': 0.0
            }
            
        hist, _ = np.histogram(angles, bins=10, density=True)
        hist = hist + 1e-10

        return {
            'mean_angle': np.mean(angles),
            'std_angle': np.std(angles),
            'angle_entropy': entropy(hist),
            'trajectory_smoothness': np.mean(np.diff(angles)) if len(angles) > 1 else 0.0,
            'max_curvature': np.max(angles)
        }
    
    def train(self, real_texts, fake_texts, extractor_model, target_layer=-2, batch_size=32):
        """
        Train the trajectory classifier
        
        Args:
            real_texts: list of real text strings
            fake_texts: list of fake text strings
            extractor_model: embedding extractor
            target_layer: which transformer layer to use
            batch_size: batch size for processing
        """
        print(f"Training trajectory classifier with {len(real_texts)} real and {len(fake_texts)} fake texts...")
        
        # Extract features
        real_features = self.extract_trajectory_features(real_texts, extractor_model, target_layer, batch_size)
        fake_features = self.extract_trajectory_features(fake_texts, extractor_model, target_layer, batch_size)
        
        # Combine features and labels
        X = np.vstack([real_features, fake_features])
        y = np.hstack([np.ones(len(real_features)), np.zeros(len(fake_features))])  # 1=real, 0=fake
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Feature statistics: mean={X.mean():.3f}, std={X.std():.3f}")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier = self._init_classifier()
        self.classifier.fit(X_scaled, y)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(self.classifier, X_scaled, y, cv=5)
        print(f"Trajectory classifier CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Training predictions for analysis
        train_predictions = self.classifier.predict(X_scaled)
        print("\nTraining Classification Report:")
        print(classification_report(y, train_predictions, target_names=['Fake', 'Real']))
        
        return cv_scores.mean()
    
    def predict_batch(self, texts, extractor_model, target_layer=-2, batch_size=32):
        """
        Predict multiple texts
        
        Returns:
            predictions: list[int] (1=real, 0=fake)
            probabilities: list[float] probability of being real
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained! Call train() first.")
        
        # Extract features
        features = self.extract_trajectory_features(texts, extractor_model, target_layer, batch_size)
        
        # Standardize and predict
        features_scaled = self.scaler.transform(features)
        predictions = self.classifier.predict(features_scaled)
        
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features_scaled)[:, 1]  # Prob of class 1 (real)
        else:
            # For classifiers without probability estimates
            probabilities = predictions.astype(float)
        
        return predictions.tolist(), probabilities.tolist()
    
    def predict_single(self, text, extractor_model, target_layer=-2):
        """Predict single text"""
        predictions, probabilities = self.predict_batch([text], extractor_model, target_layer, batch_size=1)
        return predictions[0], probabilities[0]
    
    def evaluate_pairs(self, df, labels_df, extractor_model, target_layer=-2, batch_size=32):
        """
        Evaluate text pairs using trajectory classifier
        """
        print("Evaluating text pairs with trajectory classifier...")
        
        # Prepare texts
        texts1 = df['file_1'].tolist()
        texts2 = df['file_2'].tolist()
        
        # Predict in batch
        pred1_list, prob1_list = self.predict_batch(texts1, extractor_model, target_layer, batch_size)
        pred2_list, prob2_list = self.predict_batch(texts2, extractor_model, target_layer, batch_size)
        
        predictions = []
        details = []
        
        for idx, (prob1, prob2, pred1, pred2) in enumerate(
            zip(prob1_list, prob2_list, pred1_list, pred2_list)
        ):
            # Choose text with higher probability of being real
            prediction = 1 if prob1 > prob2 else 2
            predictions.append(prediction)
            
            true_label = labels_df.loc[df.index[idx]]['real_text_id'] if df.index[idx] in labels_df.index else None
            
            details.append({
                'text1_prediction': pred1,
                'text2_prediction': pred2,
                'text1_probability': prob1,
                'text2_probability': prob2,
                'prediction': prediction,
                'true_label': true_label
            })
        
        return np.array(predictions), details