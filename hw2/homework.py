# Imports
import os

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
  playerPieces, opponentPieces, emptySpaces = set(), set(), set()

  # Create the board
  # Converting characters to numbers
  # 0: Empty
  # 1: Player
  # -1: Opponent
  board = []
  for i in range(2, 14):
    row = []
    for j in range(12):
      char = inputLines[i][j]
      if char != '\n':
        if char == '.':
          emptySpaces.add((i-2, j))
          row.append(0)
        elif char == player:
          playerPieces.add((i-2, j))
          row.append(1)
        else:
          opponentPieces.add((i-2, j))
          row.append(-1)
    board.append(row)

  return {
      'player': 1,
      'timeRemaining': timeRemaining,
      'opponentTimeRemaining': opponentTimeRemaining,
      'playerPieces': playerPieces,
      'opponentPieces': opponentPieces,
      'emptySpaces': emptySpaces,
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
def findValidMoveInDirection(state, player, position, direction, positionIsPlayerPiece):
  validMoveInDirection = []
  directionRow, directionColumn = direction
  currentRow, currentColumn = position
  currentRow += directionRow
  currentColumn += directionColumn

  # If the position in the direction is not within the board or is the player's piece or an empty space, return an empty list
  if not isWithinBoard(state, currentRow, currentColumn) or state['board'][currentRow][currentColumn] in [player, 0]:
    return validMoveInDirection

  # If the position in the direction is the opponent's piece, keep moving in the direction until the position is within the board and is the opponent's piece
  while isWithinBoard(state, currentRow, currentColumn) and state['board'][currentRow][currentColumn] == (0-player):
    currentRow += directionRow
    currentColumn += directionColumn

  # If the position is within the board
  if isWithinBoard(state, currentRow, currentColumn):
    # If the position is the player's piece, add the position to the valid moves
    if state['board'][currentRow][currentColumn] == player and not positionIsPlayerPiece:
      validMoveInDirection.append((currentRow, currentColumn))
    # If the position is an empty space, add the position to the valid moves
    elif state['board'][currentRow][currentColumn] == 0 and positionIsPlayerPiece:
      validMoveInDirection.append((currentRow, currentColumn))

  return validMoveInDirection

# A function that takes in a state, a player, and a position and returns a list of valid moves for the player at the position in the state in all directions
# The positionIsPlayerPiece is a boolean that is True if the position is the player's piece and False otherwise
def findValidMoves(state, player, position, positionIsPlayerPiece):
  validMoves = []

  # Find all the valid moves for the player at the position in all directions
  # directions = [right, down, left, up, right-down, right-up, left-down, left-up]
  directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

  for direction in directions:
    validMoves.extend(findValidMoveInDirection(state, player, position, direction, positionIsPlayerPiece))

  return validMoves

# A function that takes in a state and a player and returns a map of all the available moves for the player
# The map is a dictionary where the key is the move and the value is a list of positions of the player on the board that can make the move
# This is done to avoid duplicate moves and to avoid recalculation when flipping the opponent's pieces
def getAvailableMoves(state, player):
  moves = {}

  # If the number of empty spaces is less than the number of player pieces, use that to find the valid moves
  if state['emptySpaces'] < state['playerPieces']:
    for position in state['emptySpaces']:
      validMoves = findValidMoves(state, player, position, False)
      for move in validMoves:
        moves.setdefault(position, []).append(move)
    return moves

  # Find all the valid moves for the player at each position
  # If the move is not in the map, add the move as a key and the position as a value
  # If the move is in the map, append the position to the list of positions
  for position in state['playerPieces']:
    validMoves = findValidMoves(state, player, position, True)
    for move in validMoves:
      moves.setdefault(move, []).append(position)

  return moves

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

# A function that takes in a state and evaluates the state
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
      if (i == j == 0 or i == j == 11):
        corner_score += piece_at_position

      # If the position is an edge, add the piece to the edge score
      if (i == 0 or j == 0 or i == 11 or j == 11):
        edge_score += piece_at_position

      # Add the piece to the piece score
      # if piece_at_position == state['player']:
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
  # Create a new state with the player and the opponent switched
  # Switching the player and the opponent pieces because the player and the opponent have been switched
  newState = {
      'player': 0-state['player'],
      'timeRemaining': state['opponentTimeRemaining'],
      'opponentTimeRemaining': state['timeRemaining'],
      'board': [row.copy() for row in state['board']],
      'playerPieces': state['opponentPieces'].copy(),
      'opponentPieces': state['playerPieces'].copy(),
      'emptySpaces': state['emptySpaces'].copy()
  }

  # Make the move on the board
  # Add the move to the player pieces and remove it from the empty spaces
  i, j = move
  newState['board'][i][j] = state['player']
  newState['opponentPieces'].add(move)
  newState['emptySpaces'].remove(move)

  # Flip the opponent's pieces
  # For each piece of the player that has the move as a valid move, flip the opponent's pieces in the direction of the move and flip them
  for position in positions:
    directionRow, directionColumn = getDirection(move, position)
    currentRow, currentColumn = move
    currentRow += directionRow
    currentColumn += directionColumn

    while isWithinBoard(newState, currentRow, currentColumn) and newState['board'][currentRow][currentColumn] == (0-state['player']):
      newState['board'][currentRow][currentColumn] = state['player']
      newState['opponentPieces'].add((currentRow, currentColumn))
      newState['playerPieces'].remove((currentRow, currentColumn))
      currentRow += directionRow
      currentColumn += directionColumn

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

# A function that takes in a state and returns the best move in the format a1, a2, ..., h8
# a...h: Columns
# 1...8: Rows
def findBestMove(state):
  # Get all the available moves for the player
  moves = getAvailableMoves(state, state['player'])
  print('Available Moves:', moves, len(moves))

  # If there are no available moves for the player, return 'Pass'
  if not moves:
    return 'Pass'

  # If remaining time is less than 0.1, return the first available move
  if state['timeRemaining'] < 0.1:
    return formatMove(list(moves.keys())[0])

  # If there are available moves for the player, run the minimax algorithm to find the best move
  # Calibrate the depth of the minimax algorithm to find the best move
  remainingTime = state['timeRemaining']

  # If remaining time is less than 20 seconds, set the depth to 1
  # If remaining time is less than 60 seconds, set the depth to 2
  # If remaining time is less than 150 seconds, set the depth to 3
  # If remaining time is less than 300 seconds, set the depth to 4
  depth = 1 if remainingTime < 20 else 2 if remainingTime < 60 else 3 if remainingTime < 150 else 4

  # If there are less than 10 available moves, increase the depth
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
  # Read the input file
  state = readInput('input.txt')

  # import pprint
  # pprint.pprint(state)

  # Find the best move
  bestMove = findBestMove(state)
  print("Best Move:", bestMove)

  # Write the best move to the output file
  writeOutput('output.txt', bestMove)


# Calling the main function
if __name__ == "__main__":
  main()
