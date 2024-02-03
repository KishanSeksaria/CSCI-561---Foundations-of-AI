# Reading the input from input.txt file and populating the variables
import os
import math
from queue import PriorityQueue
inputFile = os.path.join(os.path.dirname(__file__), "input.txt")
outputFile = os.path.join(os.path.dirname(__file__), "output.txt")

with open(inputFile, "r") as file:
  inputLines = file.readlines()

# Reading the variables from the input file
variant = inputLines[0].strip()
energyLimit = int(inputLines[1].strip())
numberOfSafeLocations = int(inputLines[2].strip())
numberOfSafePaths = int(inputLines[3+numberOfSafeLocations])

# safeLocations is a dictionary with the key as the location name and the value as a tuple of the x, y, and z coordinates
safeLocations = {}
for location in inputLines[3:3+numberOfSafeLocations]:
  location = location.strip().split()
  safeLocations[location[0]] = (int(location[1]), int(location[2]), int(location[3]))

# safePaths is a bidirectional dictionary with the key as the location name and the value as a list of the safe paths from that location
# It stores the paths in both the forward and reverse directions since the paths are bidirectional
safePaths = {}
for path in inputLines[4+numberOfSafeLocations:]:
  path = path.strip().split()
  if path[0] not in safePaths:
    safePaths[path[0]] = [path[1]]
  else:
    safePaths[path[0]].append(path[1])
  # Also adding the reverse path to the dictionary
  if path[1] not in safePaths:
    safePaths[path[1]] = [path[0]]
  else:
    safePaths[path[1]].append(path[0])

# ########## Implementing the methods here ##########

# Implementing the BFS algorithm
def bfs(safeLocations, safePaths, energyLimit):
  queue = [(0, "start", ["start"], 0)]
  visited = set()

  while queue:
    cost, current, path, momentum = queue.pop(0)

    if current == "goal":
      return path
    if current in safePaths:
      neighbors = safePaths[current]
    else:
      neighbors = []

    for neighbor in neighbors:
      required_energy = safeLocations[neighbor][2] - safeLocations[current][2]
      if energyLimit + momentum >= required_energy:
        new_cost = cost + 1
        new_momentum = max(-required_energy, 0)
        if (current, neighbor) not in visited:
          visited.add((current, neighbor))
          queue.append((new_cost, neighbor, path + [neighbor], new_momentum))

  # if no path is found, return "FAIL"
  return ["FAIL"]

# Implementing the UCS algorithm
def ucs(safeLocations, safePaths, energyLimit):
  queue = PriorityQueue()
  queue.put((0, "start", ["start"], 0))
  visited = set()
  steps = 0

  while not queue.empty():
    steps += 1
    cost, current, path, momentum = queue.get()

    if current == "goal":
      print(cost)
      return path
    if current in safePaths:
      neighbors = safePaths[current]
    else:
      neighbors = []

    for neighbor in neighbors:
      required_energy = safeLocations[neighbor][2] - safeLocations[current][2]
      new_cost = cost + math.sqrt((safeLocations[neighbor][0] - safeLocations[current][0])**2 +
                                  (safeLocations[neighbor][1] - safeLocations[current][1])**2)
      new_momentum = max(-required_energy, 0)
      if (current, neighbor) not in visited:
        if energyLimit + momentum >= required_energy:
          visited.add((current, neighbor))
          queue.put((new_cost, neighbor, path + [neighbor], new_momentum))

  # if no path is found, return "FAIL"
  return ["FAIL"]

# Implementing the A* algorithm
def aStar(safeLocations, safePaths, energyLimit):
  queue = PriorityQueue()
  queue.put((0 + heuristic("start"), 0, "start", ["start"], 0))
  visited = {}
  steps = 0

  while not queue.empty():
    steps += 1
    hcost, cost, current, path, momentum = queue.get()

    if current == "goal":
      return path
    if current in safePaths:
      neighbors = safePaths[current]
    else:
      neighbors = []

    for neighbor in neighbors:
      required_energy = safeLocations[neighbor][2] - safeLocations[current][2]
      new_cost = cost + math.sqrt((safeLocations[neighbor][0] - safeLocations[current][0])**2 +
                                  (safeLocations[neighbor][1] - safeLocations[current][1])**2 +
                                  (safeLocations[neighbor][2] - safeLocations[current][2])**2)
      new_momentum = max(-required_energy, 0)
      if (current, neighbor) not in visited:
        if energyLimit + momentum >= required_energy:
          visited[(current, neighbor)] = (new_cost, new_momentum)
          queue.put((heuristic(neighbor) + new_cost, new_cost, neighbor, path + [neighbor], new_momentum))

  # if no path is found, return "FAIL"
  return ["FAIL"]

# Implementing the heuristic function for the A* algorithm
# The heuristic function is the Euclidean distance between the current location and the goal
def heuristic(a):
  return math.sqrt((safeLocations["goal"][0] - safeLocations[a][0])**2 +
                   (safeLocations["goal"][1] - safeLocations[a][1])**2 +
                   (safeLocations["goal"][2] - safeLocations[a][2])**2)

# ########## Implemented methods end here ##########


# Using the variant to determine the algorithm to use
if variant == "BFS":
  result = bfs(safeLocations, safePaths, energyLimit)
elif variant == "UCS":
  result = ucs(safeLocations, safePaths, energyLimit)
elif variant == "A*":
  result = aStar(safeLocations, safePaths, energyLimit)

# Writing the result to the output file
with open(outputFile, "w") as file:
  file.write(" ".join(result))
