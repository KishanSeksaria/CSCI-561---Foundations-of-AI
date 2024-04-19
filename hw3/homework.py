# Imports
import numpy as np
import csv
import os
import re

####################################################################################################
# ↓ Global variables ↓
# Global variable to store unique values of categorical features in training data
unique_values_train = {}

####################################################################################################
# ↓ I/O functions ↓

# Read input features
def readFeatures(filename):
  filename = os.path.join(os.path.dirname(__file__), filename)
  with open(filename, 'r') as f:
    reader = csv.reader(f)
    features = next(reader)
    X = []
    for row in reader:
      X.append(row)
  return features, X

# Read input labels
def readLabels(filename):
  filename = os.path.join(os.path.dirname(__file__), filename)
  with open(filename, 'r') as f:
    reader = csv.reader(f)
    labels = next(reader)
    Y = []
    for row in reader:
      Y.append(row)

  # Convert the labels to float64 type
  Y = np.array(Y).astype(np.float64)

  return labels, Y

# Write the output to output.csv
def writeOutput(filename, outputs):
  filename = os.path.join(os.path.dirname(__file__), filename)
  with open(filename, 'w') as f:
    f.write("BEDS\n")
    for output in outputs:
      f.write(str(output) + "\n")

####################################################################################################
# ↓ Preprocess functions ↓

# Normalize features
def normalize_features(X, feature_indices):
  for i in feature_indices:
    feature_values = [float(row[i]) for row in X]
    min_val = min(feature_values)
    max_val = max(feature_values)
    if max_val != min_val:
      for j in range(len(X)):
        X[j][i] = (float(X[j][i]) - min_val) / (max_val - min_val)
    else:
      for j in range(len(X)):
        X[j][i] = 0  # or any other value you want to assign when max_val == min_val
  return X

# Extract zip codes from "STATE" feature
def extract_zip_codes(X, state_index):
  for i in range(len(X)):
    state = X[i][state_index]
    zip_code = re.search(r'\d{5}', state)
    if zip_code:
      X[i][state_index] = zip_code.group(0)
    else:
      X[i][state_index] = "00000"
  return X

# One-hot encode categorical features
def one_hot_encode(X, categorical_feature_indices):
  # One-hot encode categorical features
  encoded_categorical = []
  for col in categorical_feature_indices:
    if col not in unique_values_train:
      unique_values_train[col] = list(set([row[col] for row in X]))
    unique_values = unique_values_train[col]
    encoded_data = np.zeros((len(X), len(unique_values)))
    for i, row in enumerate(X):
      if row[col] in unique_values:
        encoded_data[i, unique_values.index(row[col])] = 1
    encoded_categorical.append(encoded_data)

  # Remove original categorical features
  X = np.delete(X, categorical_feature_indices, axis=1)

  # Concatenate one-hot encoded features
  for encoded_data in encoded_categorical:
    X = np.concatenate((X, encoded_data), axis=1)

  return X

# Remove irrelevant features and preprocess the data
# Note: This function is specific to the given dataset
def preprocess_features(features, X):
  irrelevant_features = ["BROKERTITLE", "ADDRESS", "MAIN_ADDRESS", "STREET_NAME", "LONG_NAME", "FORMATTED_ADDRESS", "LATITUDE", "LONGITUDE"]
  relevant_feature_indices = [i for i, f in enumerate(features) if f not in irrelevant_features]
  relevant_features = [features[i] for i in relevant_feature_indices]

  # Select relevant features from the data
  X_processed = np.array([[val for j, val in enumerate(row) if j in relevant_feature_indices] for row in X])

  # Extract zip codes from the "STATE" feature
  X_processed = extract_zip_codes(X_processed, relevant_features.index("STATE"))

  # Normalize the "PROPERTYSQFT", "PRICE", "BATH" and "STATE" features
  indices = [relevant_features.index(f) for f in ["PROPERTYSQFT", "PRICE", "BATH", "STATE"]]
  X_processed = normalize_features(X_processed, indices)

  # One-hot encode categorical features
  # Identify categorical features
  categorical_features = [relevant_features.index(f) for f in relevant_features if f in [
      "TYPE", "ADMINISTRATIVE_AREA_LEVEL_2", "LOCALITY", "SUBLOCALITY"]]
  X_processed = one_hot_encode(X_processed, categorical_features)

  # Convert the data to float64 type
  X_processed = X_processed.astype(np.float64)

  return relevant_features, X_processed


####################################################################################################
# ↓ Model functions ↓

# Multi-layer perceptron model
class MLP:
  def __init__(self, input_size, hidden_layers):
    self.input_size = input_size
    self.hidden_layers = hidden_layers
    self.weights = []
    self.biases = []

    # Initialize weights and biases
    for i in range(len(hidden_layers)):
      if i == 0:
        self.weights.append(np.random.randn(input_size, hidden_layers[i]))
      else:
        self.weights.append(np.random.randn(hidden_layers[i - 1], hidden_layers[i]))
      self.biases.append(np.zeros(hidden_layers[i]))

    # Initialize output layer weights and biases
    self.weights.append(np.random.randn(hidden_layers[-1], 1))
    self.biases.append(np.zeros(1))

  # ReLU activation function
  def relu(self, X):
    # Compute the ReLU activation
    activation = np.maximum(0, X)

    # Check for NaN or infinite values
    if np.isnan(activation).any() or np.isinf(activation).any():
      print("NaN or infinite value encountered in activations")

    # Clip activations
    activation = np.clip(activation, -1e10, 1e10)

    return activation

  # Softmax activation function
  def softmax(self, x):
    return np.exp(x) / np.sum(np.exp(x))

  # Forward pass
  def forward(self, X):
    self.activations = []
    self.activations.append(X)
    for i in range(len(self.weights)):
      X = np.dot(X, self.weights[i]) + self.biases[i]
      if i == len(self.weights) - 1:
        X = X  # No activation for output layer in regression
      else:
        X = self.relu(X)
      self.activations.append(X)
    return X

  # Cross-entropy loss function
  def cross_entropy_loss(self, y_pred, y_true):
    epsilon = 1e-12  # to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    N = y_pred.shape[0]
    ce_loss = -np.sum(y_true*np.log(y_pred+1e-9))/N
    return ce_loss

  # Mean squared error loss function
  def mse_loss(self, y_pred, y_true):
    N = y_pred.shape[0]
    mse_loss = np.sum((y_true - y_pred)**2)/N
    return mse_loss

  # Backward pass
  def backward(self, Y):
    # Compute the loss
    self.loss = self.mse_loss(self.activations[-1], Y)

    # Compute the gradients
    self.gradients = []
    for i in range(len(self.weights) - 1, -1, -1):
      if i == len(self.weights) - 1:
        gradient = -1 * (Y - self.activations[-1])  # Derivative of cross-entropy loss
      else:
        gradient = np.dot(self.gradients[-1], self.weights[i + 1].T) * (self.activations[i + 1] > 0)

      # Check for NaN or infinite values
      if np.isnan(gradient).any() or np.isinf(gradient).any():
        print("NaN or infinite value encountered in gradients")

      # Clip gradients
      gradient = np.clip(gradient, -1, 1)

      self.gradients.append(gradient)

    # Update the weights and biases
    for i in range(len(self.weights)):
      if i == 0:
        self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, self.gradients[-1])
      else:
        self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, self.gradients[-i - 1])
      self.biases[i] -= self.learning_rate * np.sum(self.gradients[-i - 1], axis=0)

  # Train the model
  def train(self, X, Y, epochs=1000, learning_rate=0.01):
    print("Training the model...")
    self.learning_rate = learning_rate
    for epoch in range(epochs):
      for i in range(len(X)):
        x = np.array(X[i], ndmin=2)
        y = np.array(Y[i], ndmin=2)
        self.forward(x)
        self.backward(y)
      if epoch % 10 == 0:
        print("Epoch:", epoch, "Loss:", self.loss)
    print("Training complete! Epoch:", epoch, "Loss:", self.loss)

  # Test the model
  def test(self, X):
    print("Testing the model...")
    outputs = []
    for i in range(len(X)):
      x = np.array(X[i], ndmin=2)
      output = self.forward(x)
      outputs.append(output[0][0])
    return outputs


####################################################################################################
# ↓ Main function ↓
def main():
  # Read input features from train_data.csv
  features, X_train = readFeatures("train_data1.csv")
  # Read input labels from train_label.csv
  labels, Y_train = readLabels("train_label1.csv")

  # Preprocess the features
  features, X_train = preprocess_features(features, X_train)

  # Create the model
  model = MLP(input_size=X_train.shape[1], hidden_layers=[128, 64])

  # Train the model
  model.train(X_train, Y_train, epochs=1000, learning_rate=0.01)

  # Read input features from test_data.csv
  features, X_test = readFeatures("test_data1.csv")

  # Preprocess the features
  features, X_test = preprocess_features(features, X_test)

  # Test the model
  outputs = model.test(X_test)

  # Write the output to output.csv
  writeOutput("output.csv", outputs)


# Run the main function
if __name__ == "__main__":
  main()
