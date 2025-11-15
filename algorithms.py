import numpy as np
import pandas as pd



class MyStandardScaler:
    """
    A custom implementation of StandardScaler.
    
    Scales features by removing the mean and scaling to unit variance.
    The standard score of a sample x is calculated as:
    z = (x - u) / s
    where u is the mean of the training samples, and s is the standard deviation.
    """
    
    def __init__(self):
        """
        Initializes the scaler.
        'mean_' and 'scale_' (std dev) will be stored here after fitting.
        """
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Computes the mean and standard deviation for each feature in X.
        
        Parameters:
        X (pd.DataFrame or np.ndarray): The data used to compute the mean 
                                         and standard deviation.
        """
        # Ensure X is a numpy array for calculations
        X_values = self._check_input(X)
        
        # Calculate mean and std deviation along columns (axis=0)
        self.mean_ = np.mean(X_values, axis=0)
        self.scale_ = np.std(X_values, axis=0)
        
        # --- Handle Edge Case: Zero Standard Deviation ---
        # If a feature is constant, its std dev is 0, causing division by zero.
        # sklearn's behavior is to set scale to 1, so (x - u) / 1 = 0.
        # This results in the entire column becoming 0.
        self.scale_[self.scale_ == 0] = 1.0
        
        return self

    def transform(self, X):
        """
        Performs standardization by centering and scaling.
        
        Parameters:
        X (pd.DataFrame or np.ndarray): The data to scale.
        
        Returns:
        (pd.DataFrame or np.ndarray): The transformed data.
        """
        # Check if 'fit' has been called
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("This MyStandardScaler instance is not fitted yet. "
                             "Call 'fit' with appropriate data before using 'transform'.")
        
        is_dataframe = isinstance(X, pd.DataFrame)
        X_values = self._check_input(X)
        
        # Apply the transformation: (X - mean) / std_dev
        # NumPy broadcasting handles this element-wise for all rows.
        X_scaled = (X_values - self.mean_) / self.scale_
        
        # Return in the original format (DataFrame or numpy array)
        if is_dataframe:
            return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            return X_scaled

    def fit_transform(self, X):
        """
        Fits to data, then transforms it.
        
        Parameters:
        X (pd.DataFrame or np.ndarray): The data to fit and transform.
        
        Returns:
        (pd.DataFrame or np.ndarray): The transformed data.
        """
        # Fit on the data
        self.fit(X)
        # Then transform the same data
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """
        Scales the data back to the original representation.
        
        Parameters:
        X_scaled (pd.DataFrame or np.ndarray): The scaled data.
        
        Returns:
        (pd.DataFrame or np.ndarray): The original (un-scaled) data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("This MyStandardScaler instance is not fitted yet.")
            
        is_dataframe = isinstance(X_scaled, pd.DataFrame)
        X_values = self._check_input(X_scaled)

        # Apply the inverse formula: (X_scaled * std_dev) + mean
        X_original = (X_values * self.scale_) + self.mean_
        
        if is_dataframe:
            return pd.DataFrame(X_original, columns=X_scaled.columns, index=X_scaled.index)
        else:
            return X_original

    def _check_input(self, X):
        """Helper function to get numpy values from DataFrame or array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")
        
class MyPCA:
    """
    A from-scratch implementation of Principal Component Analysis (PCA).
    
    This class handles component selection by a fixed integer count 
    or by the minimum explained variance ratio.
    
    It also stores original DataFrame column names for inverse_transform.
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None            # To store the mean of the training data
        self.scale_ = None           # To store the std dev of the training data
        self.components_ = None      # The principal components (eigenvectors)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None    # The *actual* number of components selected
        
        # --- NEW ---
        self.feature_names_in_ = None # To store original DataFrame column names

    def fit(self, X):
        """
        Fits the PCA model to the data X.
        """
        
        # --- MODIFIED ---
        # Store feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_list()
        else:
            # Ensure it's reset if fitting on an array after a DataFrame
            self.feature_names_in_ = None 
        # --- END MODIFICATION ---
            
        # --- 1. Standardize the Data ---
        X_values = self._check_input(X)
        
        # Calculate and store mean and std dev for later use (in transform)
        self.mean_ = np.mean(X_values, axis=0)
        self.scale_ = np.std(X_values, axis=0)
        
        # Handle features with zero variance (constants)
        self.scale_[self.scale_ == 0] = 1.0
        
        X_std = (X_values - self.mean_) / self.scale_
        
        # --- 2. Compute Covariance Matrix ---
        cov_matrix = np.cov(X_std, rowvar=False)
        
        # --- 3. Eigen-decomposition ---
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        
        # --- 4. Sort Components ---
        idx = eigen_values.argsort()[::-1]
        all_eigen_values = eigen_values[idx]
        all_components = eigen_vectors[:, idx]
        
        # --- 5. Select 'k' Components ---
        total_variance = np.sum(all_eigen_values)
        self.explained_variance_ = all_eigen_values
        self.explained_variance_ratio_ = all_eigen_values / total_variance
        
        k = 0  # This will be our number of components
        
        if self.n_components is None:
            k = X_values.shape[1]
        elif isinstance(self.n_components, int):
            if 0 < self.n_components <= X_values.shape[1]:
                k = self.n_components
            else:
                raise ValueError("n_components (int) must be > 0 and <= n_features")
        elif isinstance(self.n_components, float) and 0.0 < self.n_components <= 1.0:
            cumulative_variance = np.cumsum(self.explained_variance_ratio_)
            k = np.argmax(cumulative_variance >= self.n_components) + 1
        else:
            raise ValueError("n_components must be None, an int > 0, or a float (0, 1]")

        # --- Store Final Results ---
        self.n_components_ = k
        self.components_ = all_components[:, :k]
        
        return self

    def transform(self, X):
        """
        Transforms the data X into the principal component space.
        """
        if self.components_ is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("PCA instance is not fitted yet. Call 'fit' first.")
            
        is_dataframe = isinstance(X, pd.DataFrame)
        X_values = self._check_input(X)
        
        # 1. Standardize X
        X_std = (X_values - self.mean_) / self.scale_
        
        # 2. Project
        X_pca = np.dot(X_std, self.components_)
        
        if is_dataframe:
            # Create a new DataFrame with PC names
            pc_columns = [f'PC{i+1}' for i in range(self.n_components_)] # type: ignore
            return pd.DataFrame(X_pca, columns=pc_columns, index=X.index)
        else:
            return X_pca

    def fit_transform(self, X):
        """
        Fits the model to X and then transforms X.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_pca):
        """
        Transforms data from PC space back to the original feature space.
        Returns a DataFrame if the model was originally fit on one.
        """
        if self.components_ is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("PCA instance is not fitted yet. Call 'fit' first.")
            
        is_dataframe = isinstance(X_pca, pd.DataFrame)
        X_pca_values = self._check_input(X_pca)
        
        # Store original index if it exists
        original_index = X_pca.index if is_dataframe else None
            
        # 1. Project back from PC space to standardized space
        X_std_reconstructed = np.dot(X_pca_values, self.components_.T)
        
        # 2. "Un-standardize" the data (reverse the scaling)
        X_reconstructed = (X_std_reconstructed * self.scale_) + self.mean_
        
        # --- MODIFIED ---
        # Check if we stored column names during fit
        if self.feature_names_in_ is not None:
            # Return a DataFrame with original columns and index
            return pd.DataFrame(
                X_reconstructed, 
                columns=self.feature_names_in_, 
                index=original_index
            )
        else:
            # Fallback to returning a numpy array
            return X_reconstructed
        # --- END MODIFICATION ---

    def _check_input(self, X):
        """Helper to get numpy values from DataFrame or array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")
        
class MySoftmaxRegression:
    """
    Implements Multinomial Logistic Regression (Softmax Regression) from scratch
    using NumPy. This version is designed to work with CSV data.
    """
    
    def __init__(self, n_classes,n_features,learning_rate=0.1):
        """
        Initializes the model.
        
        Args:
            n_features (int): Number of input features (e.g., 784 for MNIST).
            n_classes (int): Number of output classes (e.g., 10 for MNIST).
            learning_rate (float): The step size for gradient descent.
        """
        # Initialize weights with small random values
        # Shape: (n_features, n_classes) -> (784, 10)
        self.W = np.random.randn(n_features, n_classes) * 0.01
        
        # Initialize biases with zeros
        # Shape: (1, n_classes) -> (1, 10)
        self.b = np.zeros((1, n_classes))
        
        self.lr = learning_rate
        print(f"Initialized model: {n_classes} classes, {n_features} features, lr={learning_rate}")

    def _one_hot(self, y, n_classes):
        """
        Helper function to one-hot encode integer labels.
        """
        y_one_hot = np.zeros((y.shape[0], n_classes))
        y_one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
        return y_one_hot

    def _softmax(self, z):
        """
        Computes the stable softmax function.
        """
        # Subtract max(z) for numerical stability (prevents overflow)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _cross_entropy_loss(self, A, y_one_hot):
        """
        Computes the categorical cross-entropy loss.
        """
        m = y_one_hot.shape[0]
        # Clip probabilities to avoid log(0)
        A_clipped = np.clip(A, 1e-9, 1.0 - 1e-9)
        
        # Loss formula: L = - (1/m) * sum(y * log(A))
        loss = - (1 / m) * np.sum(y_one_hot * np.log(A_clipped))
        return loss

    def fit(self, X, y, epochs=20, batch_size=128, X_val=None, y_val=None):
        """
        Trains the model using mini-batch gradient descent.
        
        Args:
            X (np.array): Training data (n_samples, n_features).
            y (np.array): Training labels (1D array of integers).
            epochs (int): Number of passes over the entire dataset.
            batch_size (int): Number of samples per training batch.
            X_val (np.array, optional): Validation data.
            y_val (np.array, optional): Validation labels.
        """
        m, n_features = X.shape
        n_classes = self.W.shape[1]
        
        # One-hot encode labels
        y_one_hot = self._one_hot(y, n_classes)
        y_val_one_hot = None
        if X_val is not None and y_val is not None:
            y_val_one_hot = self._one_hot(y_val, n_classes)
        
        print(f"\nStarting training for {epochs} epochs with batch size {batch_size}...")
        
        for epoch in range(epochs):
            # Shuffle the data at the start of each epoch
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y_one_hot[permutation]
            
            for i in range(0, m, batch_size):
                # Get the mini-batch
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                
                batch_m = X_batch.shape[0] 
                
                # --- 1. Forward Pass ---
                Z = X_batch @ self.W + self.b
                A = self._softmax(Z)
                
                # --- 2. Backward Pass (Gradient Calculation) ---
                dZ = A - y_batch
                dW = (1 / batch_m) * (X_batch.T @ dZ)
                db = (1 / batch_m) * np.sum(dZ, axis=0, keepdims=True)
                
                # --- 3. Update Parameters ---
                self.W -= self.lr * dW
                self.b -= self.lr * db
            
            # --- End of Epoch ---
            # Calculate and print loss and accuracy on the *full* training set
            Z_full = X @ self.W + self.b
            A_full = self._softmax(Z_full)
            loss = self._cross_entropy_loss(A_full, y_one_hot)
            train_preds = np.argmax(A_full, axis=1)
            train_acc = np.mean(train_preds == y)
            
            log_message = f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Train Acc: {train_acc:.4f}"

            # Calculate validation metrics if provided
            if X_val is not None and y_val is not None:
                Z_val = X_val @ self.W + self.b
                A_val = self._softmax(Z_val)
                val_loss = self._cross_entropy_loss(A_val, y_val_one_hot)
                val_preds = np.argmax(A_val, axis=1)
                val_acc = np.mean(val_preds == y_val)
                log_message += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
            
            print(log_message)

    def predict(self, X):
        """
        Makes predictions for new data.
        """
        Z = X @ self.W + self.b
        A = self._softmax(Z)
        return np.argmax(A, axis=1)

    def evaluate(self, X, y):
        """
        Evaluates the model's accuracy on a given dataset.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

