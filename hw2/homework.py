# Imports
import os
__AVERAGE_EVALUATION_TIME_IN_MY_MACHINE__ = 0.015601277351379395
# A function that calibrates the average execution time
def calibrate():
  # Read the average execution time from the calibration file if it exists
  global __AVERAGE_EVALUATION_TIME_IN_GRADING_MACHINE__
  current_dir = os.path.dirname(os.path.abspath(__file__))
  if os.path.exists(os.path.join(current_dir, 'calibration.txt')):
    with open(os.path.join(current_dir, 'calibration.txt'), 'r') as f:
      __AVERAGE_EVALUATION_TIME_IN_GRADING_MACHINE__ = float(f.read())
      return
  __AVERAGE_EVALUATION_TIME_IN_GRADING_MACHINE__ = 0

# A function that takes in a file and returns a dictionary with the input values
def readInput(file):
  # Open the file and read the input values
  current_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(current_dir, file)
  with open(file_path, 'r') as file:
    inputLines = file.readlines()

  # Parse the input values
  player = inputLines[0].strip()
  timeRemaining = float(inputLines[1].split(' ')[0])
  opponentTimeRemaining = float(inputLines[1].split(' ')[1])

  # Create the board
  # Converting characters to numbers
  # 0: Empty
  # 1: Player
  # -1: Opponent
  board = []
  for line in inputLines[2:]:
    row = []
    for char in line:
      if char != '\n':
        if char == '.':
          row.append(0)
        elif char == player:
          row.append(1)
        else:
          row.append(-1)
    board.append(row)

  return {
      'player': 1,
      'timeRemaining': timeRemaining,
      'opponentTimeRemaining': opponentTimeRemaining,
      'board': board
  }

# A function that takes in a file at the current directory and a string and writes the string to the file
def writeOutput(file, output):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(current_dir, file)
  with open(file_path, 'w') as file:
    file.write(output)

# A function that takes in a state, a player, and a position and returns True if the position is within the board and False otherwise
def isWithinBoard(state, row, column):
  if row < 0 or row >= len(state['board']):
    return False
  if column < 0 or column >= len(state['board'][row]):
    return False
  return True

# A function that takes in a state, a player, a position, and a direction and returns a list of valid moves for the player at the position in the state in the given direction
# The direction is a tuple with the row and column direction
# We will keep moving in the direction until we find a valid move or the position is not within the board
def findValidMoveInDirection(state, player, position, direction):
  validMoveInDirection = []
  directionRow, directionColumn = direction
  currentRow, currentColumn = position
  currentRow += directionRow
  currentColumn += directionColumn

  # If the position in the direction is not within the board or is the player's piece, return an empty list
  if not isWithinBoard(state, currentRow, currentColumn):
    return validMoveInDirection

  # If the position in the direction is the player's piece or is empty, return an empty list
  if state['board'][currentRow][currentColumn] == player or state['board'][currentRow][currentColumn] == 0:
    return validMoveInDirection

  # If the position in the direction is the opponent's piece, keep moving in the direction until the position is within the board and is the opponent's piece
  while isWithinBoard(state, currentRow, currentColumn) and state['board'][currentRow][currentColumn] == (0-player):
    currentRow += directionRow
    currentColumn += directionColumn

  # If the position in the direction is within the board and is empty, add the position to the list of valid moves
  if isWithinBoard(state, currentRow, currentColumn) and state['board'][currentRow][currentColumn] == 0:
    validMoveInDirection.append((currentRow, currentColumn))
  return validMoveInDirection

# A function that takes in a state, a player, and a position and returns a list of valid moves for the player at the position in the state in all directions
def findValidMoves(state, player, position):
  validMoves = []

  # Find all the valid moves for the player at the position in all directions
  # directions = [right, down, left, up, right-down, right-up, left-down, left-up]
  directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
  for direction in directions:
    validMoves.extend(findValidMoveInDirection(state, player, position, direction))

  return validMoves

# A function that takes in a state and a player and returns a map of all the available moves for the player
# The map is a dictionary where the key is the move and the value is a list of positions of the player on the board that can make the move
# This is done to avoid duplicate moves and to avoid recalculation when flipping the opponent's pieces
def getAvailableMoves(state, player):
  moves = {}

  # Find all the positions of the player on the board
  playerPositions = []
  for i in range(len(state['board'])):
    for j in range(len(state['board'][i])):
      if state['board'][i][j] == player:
        playerPositions.append((i, j))

  # Find all the valid moves for the player at each position
  # If the move is not in the map, add the move as a key and the position as a value
  # If the move is in the map, append the position to the list of positions
  for position in playerPositions:
    validMoves = findValidMoves(state, player, position)
    for move in validMoves:
      moves.setdefault(move, []).append(position)

  return moves


# A function that takes in a state, a move, and a player and returns a new state after the move has been made
# def makeMove(state, move):
  newState = {
      'player': 'O' if state['player'] == 'X' else 'X',
      'timeRemaining': state['timeRemaining'],
      'opponentTimeRemaining': state['opponentTimeRemaining'],
      'board': [row.copy() for row in state['board']]
  }

  # Make the move on the board
  i, j = move
  newState['board'][i][j] = state['player']
  # Flip the opponent's pieces

# def minimax(state, moves, depth, isMaximizing):
  # If the depth is 0, return the evaluation of the state
  if depth == 0:
    return evaluate(state)

  # If the player is maximizing, return the maximum value of the moves
  if isMaximizing:
    bestValue = float('-inf')
    for move in moves:
      newState = makeMove(state, move)
      value = minimax(newState, getAvailableMoves(newState, newState['player']), depth - 1, False)
      bestValue = max(bestValue, value)
    return bestValue

  # If the player is minimizing, return the minimum value of the moves
  else:
    bestValue = float('inf')
    for move in moves:
      newState = makeMove(state, move)
      value = minimax(newState, getAvailableMoves(newState, newState['opponent']), depth - 1, True)
      bestValue = min(bestValue, value)
    return bestValue

# A function that takes in a move as a tuple and returns the move as a string in the format a1, a2, ..., h8
# a...h: Columns
# 1...8: Rows
def formatMove(move):
  row, col = move
  formattedMove = str(chr(col + 97)) + str(row + 1)
  return formattedMove

# A function that takes in a move and a position and returns the direction from the move to the position
# direction = (row, column)
# row: 1 if the position is above the move, -1 if the position is below the move, 0 if the position is on the same row as the move
# column: 1 if the position is to the right of the move, -1 if the position is to the left of the move, 0 if the position is on the same column as the move
def getDirection(move, position):
  row, col = move
  i, j = position
  directionRow = i - row
  directionColumn = j - col
  directionRow = 1 if directionRow > 0 else -1 if directionRow < 0 else directionRow
  directionColumn = 1 if directionColumn > 0 else -1 if directionColumn < 0 else directionColumn
  return directionRow, directionColumn

# A function that takes in a state and returns the number of corners occupied by the player
# def calculate_corner_score(state):
#   corner_score = 0
#   for i in [0, 11]:
#     for j in [0, 11]:
#       if state['board'][i][j] == state['player']:
#         corner_score += 1
#   return corner_score

# # A function that takes in a state and returns the number of edges occupied by the player
# def calculate_edge_score(state):
#   edge_score = 0
#   for i in range(12):
#     if state['board'][0][i] == state['player']:
#       edge_score += 1
#     if state['board'][11][i] == state['player']:
#       edge_score += 1
#     if state['board'][i][0] == state['player']:
#       edge_score += 1
#     if state['board'][i][11] == state['player']:
#       edge_score += 1
#   return edge_score

# # A function that takes in a state and returns the number of available moves for the player
# def calculate_mobility_score(state):
#   return len(getAvailableMoves(state, state['player']))

# # A function that takes in a state and returns the number of pieces occupied by the player
# def calculate_piece_score(state):
  piece_score = 0
  for i in range(12):
    for j in range(12):
      if state['board'][i][j] == state['player']:
        piece_score += 1
  return piece_score

# A funtion that takes in a state and evaluates the state
def evaluate(state):
  # Define the weights for each feature
  weights = {
      'corner': 10,
      'edge': 5,
      'mobility': 2,
      'piece': 1
  }

  # Calculate the score for each feature
  corner_score = 0
  edge_score = 0
  piece_score = 0

  for i in range(12):
    for j in range(12):
      piece_at_position = state['board'][i][j]

      # If the position is a corner, add the piece to the corner score
      if (i == j == 0 or i == j == 11) and piece_at_position == state['player']:
        corner_score += piece_at_position

      # If the position is an edge, add the piece to the edge score
      if (i == 0 or j == 0 or i == 11 or j == 11) and piece_at_position == state['player']:
        edge_score += piece_at_position

      # Add the piece to the piece score
      if piece_at_position == state['player']:
        piece_score += piece_at_position

  # Calculate the score for each feature
  corner_score *= weights['corner']
  edge_score *= weights['edge']
  mobility_score = len(getAvailableMoves(state, state['player'])) * weights['mobility']
  # mobility_score = calculate_mobility_score(state) * weights['mobility'] # Not using mobility score
  piece_score *= weights['piece']

  # Calculate the total evaluation score
  # evaluation_score = corner_score + edge_score + mobility_score + piece_score
  evaluation_score = corner_score + edge_score + piece_score + mobility_score

  return evaluation_score

# A function that takes in a state and a move and returns the new state after the move has been made
def makeMove(state, move, positions):
  newState = {
      'player': 0-state['player'],
      'timeRemaining': state['timeRemaining'],
      'opponentTimeRemaining': state['opponentTimeRemaining'],
      'board': [row.copy() for row in state['board']]
  }

  # Make the move on the board
  i, j = move
  newState['board'][i][j] = state['player']

  # Flip the opponent's pieces
  # For each piece of the player that has the move as a valid move, flip the opponent's pieces in the direction of the move and flip them
  for position in positions:
    directionRow, directionColumn = getDirection(move, position)
    currentRow, currentColumn = move

    while isWithinBoard(state, currentRow, currentColumn) and state['board'][currentRow][currentColumn] != state['player']:
      currentRow += directionRow
      currentColumn += directionColumn
      newState['board'][currentRow][currentColumn] = state['player']

  return newState

# Minimax algorithm with alpha-beta pruning
# This function uses the minimax algorithm to find the best move for the player
# Also tried to implement alpha-beta pruning to improve the performance of the algorithm
def minimax(state, moves, depth, isMaximizing, alpha=float('-inf'), beta=float('inf')):
  # If the depth is 0, return the evaluation of the state
  if depth == 0:
    return None, evaluate(state)

  # If there are no available moves for the player, return the evaluation of the state
  if not moves:
    return None, evaluate(state)

  # If the player is maximizing, return the maximum value of the moves
  if isMaximizing:
    bestValue = float('-inf')
    bestMove = None
    for move, positions in moves.items():
      newState = makeMove(state, move, positions)
      _, value = minimax(newState, getAvailableMoves(newState, newState['player']), depth - 1, False, alpha, beta)
      if value > bestValue:
        bestValue = value
        bestMove = move
      alpha = max(alpha, bestValue)
      if beta <= alpha:
        break
    return bestMove, bestValue

  # If the player is minimizing, return the minimum value of the moves
  else:
    bestValue = float('inf')
    bestMove = None
    for move, positions in moves.items():
      newState = makeMove(state, move, positions)
      _, value = minimax(newState, getAvailableMoves(newState, newState['player']), depth - 1, True, alpha, beta)
      if value < bestValue:
        bestValue = value
        bestMove = move
      beta = min(beta, bestValue)
      if beta <= alpha:
        break
    return bestMove, bestValue

# A function that takes in a state and returns the best move
def findBestMove(state):
  # Get all the available moves for the player
  moves = getAvailableMoves(state, state['player'])

  # If there are no available moves for the player, return 'Pass'
  if not moves:
    return 'Pass'

  # If remaining time is less than 0.1, return the first available move
  if state['timeRemaining'] < 0.1:
    return formatMove(list(moves.keys())[0])

  # If there are available moves for the player, run the minimax algorithm to find the best move
  # Calibrate the depth of the minimax algorithm to find the best move
  remainingTime = state['timeRemaining']
  depthCalibration = 0
  if __AVERAGE_EVALUATION_TIME_IN_GRADING_MACHINE__ > __AVERAGE_EVALUATION_TIME_IN_MY_MACHINE__:  # Grading machine is slower
    depthCalibration = -1
  else:
    depthCalibration = 1

  if __AVERAGE_EVALUATION_TIME_IN_GRADING_MACHINE__ == 0:
    depthCalibration = 0

  depth = 1 if remainingTime < 20 else 2 if remainingTime < 60 else 3 if remainingTime < 150 else 4
  depth += depthCalibration

  if len(moves) < 10:
    depth += 1

  # If I have more time than the opponent, increase the depth
  if state['timeRemaining'] > state['opponentTimeRemaining']:
    depth += 1

  # Run the minimax algorithm to find the best move
  import time
  start_time = time.time()
  bestMove, bestValue = minimax(state, moves, depth, True)
  end_time = time.time()
  print("Evaluation Time:", end_time - start_time, "for depth:", depth, "and number of moves:", len(moves),
        "and remaining time:", remainingTime, "and best value:", bestValue, "and best move:", bestMove)

  return formatMove(bestMove)

# Main function
def main():
  # Calibrate the average execution time
  calibrate()

  # Read the input file
  state = readInput('input.txt')

  # Find the best move
  bestMove = findBestMove(state)
  print("Best Move:", bestMove)
  print("Average Evaluation Time in My Machine:", __AVERAGE_EVALUATION_TIME_IN_MY_MACHINE__)
  print("Average Execution time in grading machine:", __AVERAGE_EVALUATION_TIME_IN_GRADING_MACHINE__)

  # Write the best move to the output file
  writeOutput('output.txt', bestMove)


# Calling the main function
if __name__ == "__main__":
  main()