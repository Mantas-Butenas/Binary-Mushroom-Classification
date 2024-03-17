import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the Mushroom Classification dataset
df = pd.read_csv("mushrooms.csv")

# Perform one-hot encoding on categorical features
df_encoded = pd.get_dummies(df)

# Display the first few rows of the encoded DataFrame
print(df_encoded.head())
# Print the columns of the encoded DataFrame
print(df_encoded.columns.tolist())

# Split the dataset into features (X) and target variable (y)
X = df_encoded.drop(['class_e', 'class_p'], axis=1)  # Features
y = df_encoded['class_p']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Define the SVM classifier
svm_clf = SVC()

# Define the hyperparameters grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=svm_clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df[['params', 'mean_test_score']])

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)
# Best Parameters: {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}
# Best Score: 1.0

# Use the best parameters to create the final SVM classifier
best_svm_clf = SVC(**grid_search.best_params_)

# Train the final SVM classifier on the training set
best_svm_clf.fit(X_train, y_train)

# Evaluate the model on the testing set
test_accuracy = best_svm_clf.score(X_test, y_test)

print("Accuracy on Testing Set:", test_accuracy)

# Make predictions on the testing set
y_pred = best_svm_clf.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Perform PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Get the loadings of the original features for each principal component
loadings = pca.components_

# Identify the original features with the highest absolute loadings for each principal component
top_features = [X.columns[np.argmax(abs(loadings[i]))] for i in range(pca.n_components_)]

# Plot the data points in the PCA-transformed space
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', s=30, edgecolors='k')

# Use top_features as labels for the principal components
plt.xlabel(f'Principal Component 1 (Top Feature: {top_features[0]})')
plt.ylabel(f'Principal Component 2 (Top Feature: {top_features[1]})')
plt.title('PCA Visualization of Mushroom Dataset')

# Add a color bar and label it with class information
plt.colorbar(scatter, label='Class')

# Show plot
plt.show()

