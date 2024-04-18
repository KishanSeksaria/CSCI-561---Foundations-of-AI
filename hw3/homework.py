# Imports
import numpy as np
import csv
import os
import re

####################################################################################################
# I/O functions

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
# Preprocess functions
# Normalize features
def normalize_features(X, feature_indices):
  for i in feature_indices:
    feature_values = [float(row[i]) for row in X]
    min_val = min(feature_values)
    max_val = max(feature_values)
    for j in range(len(X)):
      X[j][i] = (float(X[j][i]) - min_val) / (max_val - min_val)
  return X

# Label encode the "STATE" feature
def label_encode_feature(X, feature_index):
  unique_values = sorted(set(X[:, feature_index]))
  value_to_int = {v: i for i, v in enumerate(unique_values)}

  # Encode each data point's category value
  for i in range(len(X)):
    X[i, feature_index] = value_to_int[X[i, feature_index]]

  return X, value_to_int

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

  # Label encode the "STATE" feature
  X_processed, state_mapping = label_encode_feature(X_processed, relevant_features.index("STATE"))

  # Normalize the "PROPERTYSQFT", "PRICE", "BATH" and "STATE" features
  indices = [relevant_features.index(f) for f in ["PROPERTYSQFT", "PRICE", "BATH", "STATE"]]
  X_processed = normalize_features(X_processed, indices)

  # One-hot encode categorical features
  # Identify categorical features
  categorical_features = [relevant_features.index(f) for f in relevant_features if f in [
      "TYPE", "ADMINISTRATIVE_AREA_LEVEL_2", "LOCALITY", "SUBLOCALITY"]]
  X_processed = one_hot_encode(X_processed, categorical_features)

  return relevant_features, X_processed, state_mapping


####################################################################################################
# Model functions

# TODO: Implement the following function
# Train the model
def train(features, labels):
  # Train the model using the features and labels
  return 0


####################################################################################################
# Test functions

# TODO: Implement the following function
# Test the model
def test(features, model):
  # Test the model using the features
  return [0] * len(features)


####################################################################################################
# Main function
def main():
  # Read input features from train_data.csv
  features, X_train = readFeatures("train_data1.csv")
  # Read input labels from train_label.csv
  labels, Y_train = readLabels("train_label1.csv")

  # Remove irrelevant features
  features, X_train, state_mapping = preprocess_features(features, X_train)

  # TODO: Implement the following function
  # Train the model
  model = train(X_train, Y_train)

  # TODO: Implement the following function
  # Test the model
  outputs = test(X_train, model)

  # Write the output to output.csv
  writeOutput("output.csv", outputs)


# Run the main function
if __name__ == "__main__":
  main()
