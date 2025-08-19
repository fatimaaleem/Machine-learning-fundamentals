import numpy as np

# 3. Create a simple sample dataset suitable for Linear Regression.
# Let's create a dataset with a clear linear relationship: y = 2*x + 5 + noise
np.random.seed(42) # for reproducibility
X_sample = np.random.rand(50, 1) * 10 # 50 data points, one feature between 0 and 10
y_sample = 2 * X_sample.squeeze() + 5 + np.random.randn(50) * 2 # Target variable with some noise

print("--- Sample Data (first 5 rows) ---")
print("X:")
print(X_sample[:5])
print("\ny:")
print(y_sample[:5])

# 4. Implement the Linear Regression model from scratch using NumPy (Ordinary Least Squares).
class SimpleLinearRegressionOLS:
    def __init__(self):
        self.m = None # slope
        self.b = None # intercept

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term (b)
        # This allows us to treat the intercept as a coefficient of a feature that is always 1
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))

        # Calculate coefficients using the Ordinary Least Squares formula:
        # beta = (X_transpose * X)^-1 * X_transpose * y
        # Where beta is the vector of coefficients [b, m]
        try:
            X_b_transpose = X_b.T
            beta = np.linalg.inv(X_b_transpose @ X_b) @ X_b_transpose @ y
            self.b = beta[0]
            self.m = beta[1]
        except np.linalg.LinAlgError:
            print("Error: Could not compute the inverse. Check for multicollinearity or singularity in features.")
            self.b = None
            self.m = None


    def predict(self, X):
        if self.m is None or self.b is None:
            print("Error: Model has not been trained yet. Call fit() first.")
            return None
        # Make predictions using the learned coefficients: y_hat = m * x + b
        # Add a column of ones to X for the intercept term (b)
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))
        return X_b @ np.array([self.b, self.m])


# Instantiate and train the model
model = SimpleLinearRegressionOLS()
model.fit(X_sample, y_sample)

# 5. Print the learned coefficients.
print("\n--- Learned Coefficients (Scratch Implementation) ---")
print(f"Intercept (b): {model.b:.4f}")
print(f"Slope (m): {model.m:.4f}")

# 6. Make predictions on the sample data using the scratch implementation.
print("\n--- Example Predictions (Scratch Implementation) ---")
sample_indices = [0, 10, 25, 40, 49]
for i in sample_indices:
    actual_y = y_sample[i]
    predicted_y = model.predict(X_sample[i].reshape(1, -1))[0] # Predict expects 2D array, output is 1D array
    print(f"For X = {X_sample[i][0]:.2f}, Actual y = {actual_y:.2f}, Predicted y = {predicted_y:.2f}")
# 1. Import the LinearRegression class and evaluation metrics from scikit-learn.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Instantiate a LinearRegression object from scikit-learn.
model_sklearn = LinearRegression()

# 3. Train the scikit-learn Linear Regression model on the sample data.
# scikit-learn's fit method expects X to be a 2D array, which X_sample already is.
model_sklearn.fit(X_sample, y_sample)

# 4. Print the learned coefficients from the scikit-learn model.
print("--- Learned Coefficients (scikit-learn Implementation) ---")
print(f"Intercept: {model_sklearn.intercept_:.4f}")
print(f"Coefficient (Slope): {model_sklearn.coef_[0]:.4f}")
# 5. Make predictions on the sample data using the scikit-learn model.
y_pred_sklearn = model_sklearn.predict(X_sample)

# Make predictions using the scratch implementation for evaluation
y_pred_scratch = model.predict(X_sample)

# 6. Evaluate the performance of the scratch implementation.
print("\n--- Evaluation Metrics (Scratch Implementation) ---")
mse_scratch = mean_squared_error(y_sample, y_pred_scratch)
r2_scratch = r2_score(y_sample, y_pred_scratch)
print(f"Mean Squared Error (MSE): {mse_scratch:.4f}")
print(f"R-squared (R2): {r2_scratch:.4f}")

# 7. Evaluate the performance of the scikit-learn implementation.
print("\n--- Evaluation Metrics (scikit-learn Implementation) ---")
mse_sklearn = mean_squared_error(y_sample, y_pred_sklearn)
r2_sklearn = r2_score(y_sample, y_pred_sklearn)
print(f"Mean Squared Error (MSE): {mse_sklearn:.4f}")
print(f"R-squared (R2): {r2_sklearn:.4f}")

# 8. Compare the learned coefficients and the evaluation metrics.
print("\n--- Comparison of Implementations ---")
print(f"Scratch Intercept: {model.b:.4f}, scikit-learn Intercept: {model_sklearn.intercept_:.4f}")
print(f"Scratch Slope: {model.m:.4f}, scikit-learn Slope: {model_sklearn.coef_[0]:.4f}")
print(f"Scratch MSE: {mse_scratch:.4f}, scikit-learn MSE: {mse_sklearn:.4f}")
print(f"Scratch R2: {r2_scratch:.4f}, scikit-learn R2: {r2_sklearn:.4f}")
print("\nObservation: The learned coefficients and evaluation metrics are very similar between the scratch implementation (using OLS) and the scikit-learn implementation, which also uses an OLS-based approach for this type of problem by default.")
import numpy as np
import pandas as pd

# 3. Create a simple sample dataset suitable for binary classification.
np.random.seed(42) # for reproducibility

# Class 0 data (around bottom-left)
X0 = np.random.rand(25, 2) * 5

# Class 1 data (around top-right)
X1 = np.random.rand(25, 2) * 5 + 5

# Combine data and create labels
X_sample_lr = np.vstack((X0, X1))
y_sample_lr = np.vstack((np.zeros((25, 1)), np.ones((25, 1)))).squeeze()

print("--- Sample Binary Classification Data (first 5 rows) ---")
print("Features (X):")
print(pd.DataFrame(X_sample_lr).head())   # fixed: replaced display with print
print("\nTarget (y):")
print(pd.DataFrame(y_sample_lr, columns=['Target']).head())
print("\nTarget (y) tail (to show Class 1):")
print(pd.DataFrame(y_sample_lr, columns=['Target']).tail())


# 4. Implement the Logistic Regression model from scratch using NumPy (Gradient Descent).
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """The sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Trains the Logistic Regression model using Gradient Descent."""
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros((n_features, 1))  # column vector
        self.bias = 0

        # Reshape y to be a column vector
        y = y.reshape(-1, 1)

        # Gradient Descent loop
        for _ in range(self.n_iterations):
            # Linear model: z = Xw + b
            linear_model = X @ self.weights + self.bias
            # Sigmoid
            y_predicted = self._sigmoid(linear_model)

            # Gradients
            dw = (1/n_samples) * (X.T @ (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """Predicts the probability of belonging to class 1."""
        linear_model = X @ self.weights + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Predicts the class label (0 or 1)."""
        y_predicted_proba = self.predict_proba(X)
        return (y_predicted_proba >= threshold).astype(int)


print("\n--- Logistic Regression Scratch Implementation Ready ---")

#  Test the model
model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X_sample_lr, y_sample_lr)
y_pred = model.predict(X_sample_lr)

print("\nPredictions (first 10):", y_pred[:10].ravel())
print("Actual (first 10):     ", y_sample_lr[:10])
import numpy as np
import pandas as pd

# 3. Create a simple sample dataset suitable for classification.
# Let's create a dataset with two numerical features and a binary target.
# We'll make it somewhat separable to illustrate splitting.
np.random.seed(100) # for reproducibility

# Class 0 data (around bottom-left)
X0_dt = np.random.rand(30, 2) * 5

# Class 1 data (around top-right)
X1_dt = np.random.rand(30, 2) * 5 + 5

# Combine data and create labels
X_sample_dt = np.vstack((X0_dt, X1_dt))
y_sample_dt = np.vstack((np.zeros((30, 1)), np.ones((30, 1)))).squeeze()

# Add a categorical feature for demonstration (will need encoding later if used directly in scratch)
categorical_feature = np.array(['Red'] * 15 + ['Blue'] * 15 + ['Red'] * 15 + ['Blue'] * 15)
np.random.shuffle(categorical_feature) # Shuffle to mix categories

df_sample_dt = pd.DataFrame(X_sample_dt, columns=['Feature_A', 'Feature_B'])
df_sample_dt['Categorical_Feature'] = categorical_feature
df_sample_dt['Target'] = y_sample_dt

print("--- Sample Decision Tree Data (first 5 rows) ---")
print(df_sample_dt.head())
print("\n--- Sample Decision Tree Data (last 5 rows) ---")
print(df_sample_dt.tail())
# 4. Implement the Decision Tree model from scratch using NumPy (for classification).

class Node:
    """Represents a node in the Decision Tree."""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index # Index of the feature to split on
        self.threshold = threshold       # Threshold value for the split
        self.left = left                 # Left child node
        self.right = right               # Right child node
        self.value = value               # Value if this is a leaf node (class label)

class DecisionTreeScratch:
    """Decision Tree Classifier implemented from scratch."""
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split # Minimum number of samples required to split a node
        self.max_depth = max_depth               # Maximum depth of the tree
        self.root = None                         # The root node of the tree

    def _is_finished(self, depth):
        """Checks if the stopping criteria are met."""
        return depth >= self.max_depth or self.n_samples < self.min_samples_split or self.n_classes == 1

    def _gini_impurity(self, y):
        """Calculates the Gini impurity of a set of labels."""
        if len(y) == 0:
            return 0
        class_counts = np.bincount(y.astype(int))
        probabilities = class_counts / len(y)
        gini = 1.0 - np.sum(probabilities**2)
        return gini

    def _best_split(self, X, y):
        """Finds the best feature and threshold to split the data."""
        best_gini = float('inf')
        split_idx = None
        split_threshold = None

        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                # Split data based on the current feature and threshold
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue # Skip if split results in empty child node

                y_left, y_right = y[left_indices], y[right_indices]

                # Calculate weighted Gini impurity
                gini_left = self._gini_impurity(y_left)
                gini_right = self._gini_impurity(y_right)

                # Weighted average of Gini impurities of the child nodes
                weighted_gini = (len(y_left) / n_samples) * gini_left + (len(y_right) / n_samples) * gini_right

                # Check if this split is better than the current best
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    split_idx = feature_index
                    split_threshold = threshold

        return split_idx, split_threshold

    def _build_tree(self, X, y, depth):
        """Recursively builds the Decision Tree."""
        self.n_samples = X.shape[0]
        self.n_classes = len(np.unique(y))

        # Stopping criteria
        if self._is_finished(depth):
            # Create a leaf node with the majority class label
            most_common_class = np.argmax(np.bincount(y.astype(int)))
            return Node(value=most_common_class)

        # Find the best split
        split_idx, split_threshold = self._best_split(X, y)

        # If no good split was found (e.g., all samples are the same), create a leaf node
        if split_idx is None:
             most_common_class = np.argmax(np.bincount(y.astype(int)))
             return Node(value=most_common_class)


        # Split the data
        left_indices = np.where(X[:, split_idx] <= split_threshold)[0]
        right_indices = np.where(X[:, split_idx] > split_threshold)[0]
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # Recursively build left and right subtrees
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        # Return the current node with the best split
        return Node(feature_index=split_idx, threshold=split_threshold,
                    left=left_child, right=right_child)


    def fit(self, X, y):
        """Trains the Decision Tree model."""
        # Convert pandas DataFrame/Series to numpy arrays if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Handle potential categorical features by dropping them for this numerical-only scratch implementation
        # In a real implementation, you'd encode them first (e.g., One-Hot Encoding)
        # For this simple demonstration, we'll only use the numerical features.
        X_numerical = X[:, :2] # Assuming the first two columns are numerical features Feature_A and Feature_B

        self.root = self._build_tree(X_numerical, y, depth=0)

    def _traverse_tree(self, x, node):
        """Recursively traverses the tree to make a prediction for a single data point."""
        # If it's a leaf node, return its value
        if node.value is not None:
            return node.value

        # Traverse left or right based on the feature and threshold
        feature_value = x[node.feature_index]
        if feature_value <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        """Makes predictions for an array of data points."""
        # Convert pandas DataFrame/Series to numpy arrays if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Ensure we only use the numerical features used during training
        X_numerical = X[:, :2] # Assuming the first two columns are numerical features

        # Make a prediction for each data point by traversing the tree
        predictions = [self._traverse_tree(x, self.root) for x in X_numerical]
        return np.array(predictions)

print("--- Decision Tree Scratch Implementation Defined ---")
# 1. Import necessary libraries
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example dataset (replace with your X_sample_hc)
np.random.seed(42)
X_sample_hc = np.vstack([
    np.random.randn(10, 2) + [0, 0],
    np.random.randn(10, 2) + [5, 5],
    np.random.randn(10, 2) + [0, 5]
])

# 2. Instantiate scikit-learn AgglomerativeClustering model for full hierarchy
model_hc_sklearn_full = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0,
    linkage='single',          # linkage method
    metric='euclidean'       # fixed: metric → metric
)

# 3. Train model
model_hc_sklearn_full.fit(X_sample_hc)
print("--- Scikit-learn Agglomerative Clustering (Full Hierarchy) Trained ---")

# 4. Generate SciPy linkage matrix (pass raw data directly!)
linkage_matrix_scipy = linkage(X_sample_hc, method='single', metric='euclidean')

print("\n--- SciPy Linkage Matrix Generated ---")
print("Shape of Linkage Matrix:", linkage_matrix_scipy.shape)
print("First 5 rows:\n", linkage_matrix_scipy[:5])

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix_scipy)
plt.title('Hierarchical Clustering Dendrogram (SciPy, Single Linkage)')
plt.xlabel('Sample Index or Cluster ID')
plt.ylabel('Distance')
plt.show()

# 5. Train scikit-learn model with k=3 clusters
model_hc_sklearn_k3 = AgglomerativeClustering(
    n_clusters=3,
    linkage='single',
    affinity='euclidean'   # fixed
)
model_hc_sklearn_k3.fit(X_sample_hc)
print("\n--- Scikit-learn Agglomerative Clustering (K=3) Trained ---")

# 6. Get sklearn labels
cluster_labels_sklearn = model_hc_sklearn_k3.labels_
print("\n--- Scikit-learn Cluster Labels (K=3) ---")
print(cluster_labels_sklearn)

# 7. Reference labels from SciPy
reference_labels_scipy = fcluster(linkage_matrix_scipy, t=3, criterion='maxclust')
print("\n--- Reference Cluster Labels (SciPy fcluster, K=3) ---")
print(reference_labels_scipy)

# 8. Evaluate with Adjusted Rand Index
ari_sklearn_scipy = adjusted_rand_score(cluster_labels_sklearn, reference_labels_scipy)
print("\n--- Evaluation ---")
print(f"Adjusted Rand Index: {ari_sklearn_scipy:.4f}")
