# Imports
import numpy as np
import csv
import os

# I/O functions
# Read input features and labels from train_data.csv and train_label.csv
def readInput():
  # Read the features from train_data.csv
  filename = os.path.join(os.path.dirname(__file__), "train_data1.csv")
  with open(filename, 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    features = []
    for row in reader:
      features.append(row)

  # Read the labels from train_label.csv
  filename = os.path.join(os.path.dirname(__file__), "train_label1.csv")
  with open(filename, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    labels = []
    for row in reader:
      labels.append(row)

  # Return the features and labels
  return features, labels

# Write the output to output.csv
def writeOutput(filename, outputs: list[int]):
  # Write the output to the file
  filename = os.path.join(os.path.dirname(__file__), filename)
  with open(filename, 'w') as f:
    f.write("BEDS\n")
    for output in outputs:
      f.write(str(output) + ",\n")

# Model functions
# TODO: Implement the following function
# Train the model
def train(features, labels):
  # Train the model using the features and labels
  return 0


# main function
def main():
  # Read input features from train_data.csv
  features, labels = readInput()

  # TODO: Implement the following functions
  # # Train the model
  # model = train(features, labels)

  # Write the output to output.csv
  writeOutput("output.csv", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# Run the main function
if __name__ == "__main__":
  main()
