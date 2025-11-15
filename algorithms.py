import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial import KDTree
from collections import Counter

np.random.seed(0)

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

    def fit(self, X, y, epochs=100, batch_size=128, X_val=None, y_val=None):
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
            
            # log_message = f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Train Acc: {train_acc:.4f}"
            
            # Calculate validation metrics if provided
            if X_val is not None and y_val is not None:
                Z_val = X_val @ self.W + self.b
                A_val = self._softmax(Z_val)
                val_loss = self._cross_entropy_loss(A_val, y_val_one_hot)
                val_preds = np.argmax(A_val, axis=1)
                val_acc = np.mean(val_preds == y_val)
                # log_message += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
            
            # print(log_message)

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

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.tree = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Build the KDTree from the training data.
        """
        self.tree = KDTree(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        """
        Predict labels for new data points.
        """
        X_test = np.atleast_2d(X_test)
        predictions = []
        
        # Query the tree for the k nearest neighbors for all test points at once
        # tree.query returns distances and indices
        distances, indices = self.tree.query(X_test, k=self.k) # type: ignore
        
        # If k=1, indices is a 1D array, needs to be 2D
        if self.k == 1:
            indices = indices.reshape(-1, 1)

        # Go through each set of neighbor indices
        for neighbor_indices in indices: # type: ignore
            # Get the labels of the neighbors
            neighbor_labels = self.y_train[neighbor_indices] # type: ignore
            
            # Find the most common label (majority vote)
            most_common = Counter(neighbor_labels).most_common(1)
            predictions.append(most_common[0][0])
            
        return np.array(predictions)

class Node_XG:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, default_left=True):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.default_left = default_left  # direction for missing values


class XGBoostClassifier:
    """ A simplified implementation of XGBoost for binary classification.
    Use for split method 'approx' for weighted quantile thresholds
    or 'exact' for exact greedy."""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 reg_lambda=1.0, gamma=0.0, min_child_weight=1.0, n_bins=100, split_method='approx'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.trees = []
        self.base_score = 0.0
        self.n_bins = n_bins  # for weighted quantile thresholds
        self.split_method = split_method

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _leaf_weight(self, G, H):
        return -G / (H + self.reg_lambda)


    def _gain(self, GL, HL, GR, HR):
        G = GL + GR
        H = HL + HR
        gain = 0.5 * ((GL**2 / (HL + self.reg_lambda)) +
                      (GR**2 / (HR + self.reg_lambda)) -
                      (G**2 / (H + self.reg_lambda))) - self.gamma
        return gain


    def build_tree_exact(self, X, g, h, depth):
        G, H = g.sum(), h.sum()
        if depth >= self.max_depth or H < self.min_child_weight or X.shape[0] <= 1:
            return Node_XG(value=self._leaf_weight(G, H))

        best_gain, best_feat, best_thr = -np.inf, None, None
        best_mask = None

        for j in range(X.shape[1]):
            xj = X[:, j]
            order = np.argsort(xj)
            x_sorted = xj[order]
            g_sorted = g[order]
            h_sorted = h[order]

            G_prefix = np.cumsum(g_sorted)[:-1]
            H_prefix = np.cumsum(h_sorted)[:-1]
            distinct = np.where(x_sorted[:-1] != x_sorted[1:])[0]

            for pos in distinct:
                GL, HL = G_prefix[pos], H_prefix[pos]
                GR, HR = G - GL, H - HL
                if HL < self.min_child_weight or HR < self.min_child_weight:
                    continue
                gain = self._gain(GL, HL, GR, HR)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thr = (x_sorted[pos] + x_sorted[pos + 1]) / 2
                    best_mask = xj <= best_thr

        if best_gain <= 0 or best_mask is None:
            return Node_XG(value=self._leaf_weight(G, H))

        left = self.build_tree_exact(X[best_mask], g[best_mask], h[best_mask], depth + 1)
        right = self.build_tree_exact(X[~best_mask], g[~best_mask], h[~best_mask], depth + 1)
        return Node_XG(feature_index=best_feat, threshold=best_thr, left=left, right=right)


    def _weighted_quantile_thresholds(self, feature, hessian):
        """Approximate split thresholds using Hessian-weighted quantiles."""
        # Sort by feature value
        order = np.argsort(feature)
        x_sorted = feature[order]
        h_sorted = hessian[order]
        cum_h = np.cumsum(h_sorted)
        cum_h /= cum_h[-1]  # normalize to 0–1 range

        quantile_levels = np.linspace(0, 1, self.n_bins + 1)
        thresholds = np.interp(quantile_levels, cum_h, x_sorted)
        thresholds = np.unique(thresholds)
        return thresholds[1:-1]  # skip min/max edges


    def build_tree_approx(self, X, g, h, depth):
        G, H = g.sum(), h.sum()
        if depth >= self.max_depth or H < self.min_child_weight or X.shape[0] <= 1:
            return Node_XG(value=self._leaf_weight(G, H))

        best_gain, best_feat, best_thr = -np.inf, None, None
        best_mask, best_default_left = None, True

        for j in range(X.shape[1]):
            xj = X[:, j]
            thresholds = self._weighted_quantile_thresholds(xj, h)
            if thresholds.size == 0:
                continue

            mask_valid = ~np.isnan(xj)
            x_valid = xj[mask_valid]
            g_valid = g[mask_valid]
            h_valid = h[mask_valid]

            for thr in thresholds:
                mask_left = x_valid <= thr
                GL, HL = g_valid[mask_left].sum(), h_valid[mask_left].sum()
                GR, HR = g_valid[~mask_left].sum(), h_valid[~mask_left].sum()

                # compute total G/H of missing samples
                g_miss = g[~mask_valid].sum()
                h_miss = h[~mask_valid].sum()

                # missing -> left
                gain_left = self._gain(GL + g_miss, HL + h_miss, GR, HR)
                # missing -> right
                gain_right = self._gain(GL, HL, GR + g_miss, HR + h_miss)

                if gain_left > best_gain:
                    best_gain = gain_left
                    best_feat = j
                    best_thr = thr
                    best_mask = (xj <= thr) | np.isnan(xj)
                    best_default_left = True
                if gain_right > best_gain:
                    best_gain = gain_right
                    best_feat = j
                    best_thr = thr
                    best_mask = (xj <= thr) & (~np.isnan(xj))
                    best_default_left = False

        if best_gain <= 0 or best_mask is None:
            return Node_XG(value=self._leaf_weight(G, H))

        left = self.build_tree_approx(X[best_mask], g[best_mask], h[best_mask], depth + 1)
        right = self.build_tree_approx(X[~best_mask], g[~best_mask], h[~best_mask], depth + 1)
        return Node_XG(feature_index=best_feat, threshold=best_thr,
                    left=left, right=right, default_left=best_default_left)


    def predict_tree(self, node, X):
        if node.value is not None:
            return np.full(X.shape[0], node.value)
        mask = X[:, node.feature_index] <= node.threshold
        preds = np.empty(X.shape[0])
        preds[mask] = self.predict_tree(node.left, X[mask])
        preds[~mask] = self.predict_tree(node.right, X[~mask])
        return preds


    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)


        pos_ratio = np.clip(np.mean(y), 1e-6, 1 - 1e-6) 

        self.base_score = np.log(pos_ratio / (1 - pos_ratio))
        y_pred = np.full_like(y, self.base_score)

        for t in range(self.n_estimators):
            p = self.sigmoid(y_pred)
            g = p - y
            h = p * (1 - p)

            if self.split_method == "exact":
                tree = self.build_tree_exact(X, g, h, depth=0)
            elif self.split_method == "approx":
                tree = self.build_tree_approx(X, g, h, depth=0)
            else:
                raise ValueError("split_method must be 'exact' or 'approx'")

            self.trees.append(tree)

            y_pred += self.learning_rate * self.predict_tree(tree, X)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        y_pred = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            y_pred += self.learning_rate * self.predict_tree(tree, X)
        p = self.sigmoid(y_pred)
        return np.vstack([1 - p, p]).T  # shape (n, 2)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)
    
class DecisionTreeNode:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        self.feature_index = feature_index  # index of feature to split on
        self.threshold = threshold  # threshold value to split
        self.left = left  # left subtree
        self.right = right  # right subtree
        self.value = value  # class label for leaf nodes

    def is_leaf_node(self):
        # returns true if this node hold a value

        return self.value is not None


# ----------- Decision Tree Classifier -----------s
class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2, feature_indices=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This is an extra line from earlier
        # the indices of features to be used are passed as argument
        self.feature_indices = feature_indices
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def fit(self, X, y):
        # ensure numpy arrays so X[i] indexes rows, not DataFrame columns
        X = np.asarray(X)
        y = np.asarray(y)
        # print ('in fit function, type of X is', type(X))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This is an different from earlier
        # if subset of features is not defined
        if self.feature_indices is None:
            self.feature_indices = list(np.arange(len(X[0])))
            assert (
                len(self.feature_indices) > 1
            ), "less than 2 features, tree building may fail"
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.root = self._build_tree(X, y)

    # additional argument
    def _build_tree(self, X, y, depth=0):
        num_classes = len(set(y))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # base case 1
        # This is an different from earlier
        num_samples = len(y)
        if num_samples == 0:
            return None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # base case 2
        # stopping conditions
        if (
            depth >= self.max_depth
            or num_classes == 1
            or num_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        # greedy search for best split
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # feat_idxs instead of num_features in this line
        best_feat, best_thresh = self._best_split(X, y)
        # print ('best feat and best thresh are', best_feat, best_thresh)
        # base case 3 - best split not found
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            # print ('best feat is none, returning common label', leaf_value)
            return DecisionTreeNode(value=leaf_value)

        # recursive case
        # split is found
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        # print ('left_idx and right idx are',sum(left_idx), sum(right_idx))

        # if one of the parts is empty
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            # If either side is empty, return a leaf node with the most common label
            leaf_value = self.most_common_label( # type: ignore
                y
            )  # pyright: ignore[reportAttributeAccessIssue]
            return DecisionTreeNode(value=leaf_value)

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return DecisionTreeNode(
            feature_index=best_feat, threshold=best_thresh, left=left, right=right
        )

    def _best_split(self, X, y):
        best_gain = 0
        split_idx, split_thresh = None, None

        # ensure we have an iterable of feature indices
        n_features = X.shape[1]
        feat_idxs = (
            self.feature_indices
            if self.feature_indices is not None
            else np.arange(n_features)
        )

        for feat_idx in feat_idxs:
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                gain = self._gini_gain(y, X[:, feat_idx], thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh
        return split_idx, split_thresh

    def _gini_gain(self, y, feature_column, threshold):
        # parent gini
        parent_gini = self._gini(y)

        # generate splits
        left_idx = feature_column <= threshold
        right_idx = feature_column > threshold

        if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
            return 0

        # weighted avg. gini of children
        n = len(y)
        n_left, n_right = len(y[left_idx]), len(y[right_idx])
        gini_left = self._gini(y[left_idx])
        gini_right = self._gini(y[right_idx])
        child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right

        # gini gain
        return parent_gini - child_gini

    def _gini(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1.0 - sum(p**2 for p in probabilities if p > 0)

    def _most_common_label(self, y):
        if set(y) is None:
            return None
        # print ('unique y values are', set(y))
        counter = Counter(y)
        # print ('in most common label', counter)
        most_common = counter.most_common(1)[0][0]
        # print ('in most common label', counter, most_common)
        return most_common

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _predict(self, inputs, node):
        # base case
        if node.is_leaf_node():
            return node.value
        # recursive calling of left and right branches
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)


# ----------- Random Forest Classifier -----------
class RandomForest:
    def __init__(
        self, n_trees=10, max_depth=10, min_samples_split=2, max_features=None
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        # ensure numpy arrays so bootstrap sampling uses row indices
        X = np.asarray(X)
        y = np.asarray(y)
        self.max_features = self.max_features or len(X[0])

        for tci in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # selection feature indices
            feature_indices = np.random.choice(
                range(len(X[0])), self.max_features, replace=False
            )

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_indices=feature_indices,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        # print ('in bootstrap, type of X is', type(X))
        n_samples = len(X)
        indices = [np.random.randint(0, n_samples - 1) for _ in range(n_samples)]
        # print ('length of samples and indices is', n_samples, len(indices))
        return np.array([X[i] for i in indices]), np.array([y[i] for i in indices])

    def predict(self, X):
        X = np.asarray(X)
        tree_preds = [tree.predict(X) for tree in self.trees]
        tree_preds = list(zip(*tree_preds))
        return [self._most_common_label(preds) for preds in tree_preds]

    def _most_common_label(self, labels):
        return Counter(labels).most_common(1)[0][0]
