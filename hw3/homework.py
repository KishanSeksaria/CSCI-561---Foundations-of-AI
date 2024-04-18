# Imports
import numpy as np
import csv
import os
import re

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
    for j in range(len(X)):
      X[j][i] = (float(X[j][i]) - min_val) / (max_val - min_val)
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
    unique_values = list(set([row[col] for row in X]))
    encoded_data = np.zeros((len(X), len(unique_values)))
    for i, row in enumerate(X):
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

  return relevant_features, X_processed


####################################################################################################
# ↓ Model functions ↓

# I want to create a MLP neural network model to predict the number of beds in a property
# The model will have the following architecture:
# Input layer: Number of features
# Hidden layer 1: 128 units, ReLU activation
# Hidden layer 2: 64 units, ReLU activation
# Output layer: 1 unit, Softmax activation
# Loss function: Mean Squared Error
# Optimizer: Adam
# Create the model

def create_model(input_dim, hidden_dims=[128, 64], output_dim=1):
  np.random.seed(0)
  model = {}
  model['num_layers'] = len(hidden_dims) + 1
  model['W1'] = np.random.randn(input_dim, hidden_dims[0]) / np.sqrt(input_dim)
  model['b1'] = np.zeros((1, hidden_dims[0]))
  for i in range(1, len(hidden_dims)):
    model[f'W{i+1}'] = np.random.randn(hidden_dims[i-1], hidden_dims[i]) / np.sqrt(hidden_dims[i-1])
    model[f'b{i+1}'] = np.zeros((1, hidden_dims[i]))
  model[f'W{len(hidden_dims)+1}'] = np.random.randn(hidden_dims[-1], output_dim) / np.sqrt(hidden_dims[-1])
  model[f'b{len(hidden_dims)+1}'] = np.zeros((1, output_dim))
  return model

# Forward pass
def forward(model, X):
  num_layers = model['num_layers']
  W = [model[f'W{i}'] for i in range(1, num_layers+1)]
  b = [model[f'b{i}'] for i in range(1, num_layers+1)]

  # Forward pass
  a = X
  for i in range(num_layers):
    z = np.dot(a, W[i]) + b[i]
    a = np.maximum(0, z)

  # Output layer
  z = np.dot(a, W[num_layers]) + b[num_layers]
  a = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

  return a

# Loss function
def compute_loss(model, X, y):
  N = X.shape[0]
  predictions = forward(model, X)
  return np.sum((predictions - y) ** 2) / N

# Backward pass
def backward(model, X, y):
  num_layers = model['num_layers']
  W = [model[f'W{i}'] for i in range(1, num_layers+1)]
  b = [model[f'b{i}'] for i in range(1, num_layers+1)]
  m = X.shape[0]
  gradients = {}
  # Compute gradients for output layer
  dz = forward(model, X) - y
  gradients[f'dW{num_layers}'] = np.dot(forward(model, X).T, dz) / m
  gradients[f'db{num_layers}'] = np.sum(dz, axis=0, keepdims=True) / m
  # Compute gradients for hidden layers
  da = dz
  for i in range(num_layers-1, 0, -1):
    dz = np.dot(da, W[i].T) * (forward(model, X) > 0)
    gradients[f'dW{i}'] = np.dot(forward(model, X).T, dz) / m
    gradients[f'db{i}'] = np.sum(dz, axis=0, keepdims=True) / m
    da = dz
  return gradients, compute_loss(model, X, y)

# Update the model parameters
def update_parameters(model, gradients, learning_rate):
  for key in model:
    model[key] -= learning_rate * gradients['d' + key]

  return model

# Train the model
def train(model, features, labels, epochs=1000, learning_rate=1e-3):
  for epoch in range(epochs):
    gradients, loss = backward(model, features, labels)
    model = update_parameters(model, gradients, learning_rate)
    if epoch % 100 == 0:
      print(f'Epoch: {epoch}, Loss: {loss}')

  return model

# Test the model
def test(model, features):
  predictions = forward(model, features)
  return predictions.flatten()


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
  model = create_model(X_train.shape[1])

  # Train the model
  model = train(model, X_train, Y_train)

  # Read input features from test_data.csv
  features, X_test = readFeatures("test_data1.csv")

  # Preprocess the features
  features, X_test = preprocess_features(features, X_test)

  # Test the model
  outputs = test(model, X_test)

  # Write the output to output.csv
  writeOutput("output.csv", outputs)


# Run the main function
if __name__ == "__main__":
  main()
