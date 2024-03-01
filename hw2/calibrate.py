# In this file, you will write a script that calibrates the evaluation function by running the evaluation function on a list of states and measuring the average execution time. The script should write the average execution time to a file called calibration.txt. The file should be located in the same directory as the script. The file should contain a single number, the average execution time in seconds.

# Imports
import os
import random
import time
from homework import evaluate

states = []

# Generate a list of states
def generate_states(numberOfStates):
  for i in range(numberOfStates):
    states.append({
        "player": random.choice([1, -1]),
        "timeRemaining": random.uniform(1, 300),
        "opponentTimeRemaining": random.uniform(1, 300),
        "board": [[random.choice([1, -1, 0]) for _ in range(12)] for _ in range(12)]
    })

def calibrate():
  generate_states(1000)
  start_time = time.time()
  for state in states:
    evaluate(state)
  end_time = time.time()
  execution_time = end_time - start_time
  current_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(current_dir, 'calibration.txt')
  with open(file_path, 'w') as file:
    file.write(str(execution_time))
  print('Calibration complete. Average execution time:', execution_time, 'seconds')


if __name__ == "__main__":
  calibrate()
