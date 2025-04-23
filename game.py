import numpy as np
import random

class Game2048:
    def __init__(self):
        """
        Set up the game board (4x4) and start with two random tiles (2 or 4).
        The score begins at 0.
        """
        self.board_size = 4
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.score = 0
        self._place_random_tile()
        self._place_random_tile()

    def _place_random_tile(self):
        """
        Pick a random empty spot and place either a 2 or a 4.
        There's a 90% chance for a 2, and 10% for a 4.
        """
        empty_cells = np.argwhere(self.board == 0)
        if empty_cells.size > 0:
            row, col = random.choice(empty_cells)
            self.board[row, col] = 2 if random.random() < 0.9 else 4

    def move(self, direction, simulate=False):
        """
        Move tiles in the given direction and update the score.
        If the board changes, add a new random tile.
        """
        prev_board = self.board.copy()
        prev_score = self.score

        if direction == 'left':
            self.board, points = self._move_left(self.board)
        elif direction == 'right':
            self.board, points = self._move_right(self.board)
        elif direction == 'up':
            self.board, points = self._move_up(self.board)
        elif direction == 'down':
            self.board, points = self._move_down(self.board)

        self.score += points

        # Only add a new tile if the board actually changed
        if not np.array_equal(prev_board, self.board):
            if not simulate:
                self._place_random_tile()
        else:
            self.score = prev_score

    def _move_left(self, board):
        """Move and merge tiles to the left."""
        new_board = np.zeros_like(board)
        points = 0
        for i in range(self.board_size):
            row = board[i][board[i] != 0]  # Remove zeros
            row, row_points = self._merge(row)  # Merge tiles if possible
            points += row_points
            new_board[i, :len(row)] = row
        return new_board, points

    def _move_right(self, board):
        """Move tiles to the right (reverse, move, reverse back)."""
        flipped_board = np.fliplr(board)
        new_board, points = self._move_left(flipped_board)
        new_board = np.fliplr(new_board)
        return new_board, points

    def _move_up(self, board):
        """Move tiles up (transpose, move, transpose back)."""
        transposed_board = board.T
        new_board, points = self._move_left(transposed_board)
        new_board = new_board.T
        return new_board, points

    def _move_down(self, board):
        """Move tiles down (transpose, reverse, move, transpose back)."""
        transposed_board = board.T
        flipped_board = np.fliplr(transposed_board)
        new_board, points = self._move_left(flipped_board)
        new_board = np.fliplr(new_board)
        new_board = new_board.T
        return new_board, points

    def _merge(self, row):
        """Merge tiles in a single row."""
        merged_row = []
        points = 0
        skip = False
        for i in range(len(row)):
            if skip:
                skip = False
                continue
            if i < len(row) - 1 and row[i] == row[i + 1]:  # Can merge
                merged_row.append(row[i] * 2)
                points += row[i] * 2
                skip = True
            else:
                merged_row.append(row[i])
        return np.array(merged_row), points

    def available_moves(self):
        """Check all possible moves and return a list of valid ones."""
        moves = []
        for direction in ['up', 'down', 'left', 'right']:
            temp_board = self.board.copy()
            temp_score = self.score
            self.move(direction)
            if not np.array_equal(temp_board, self.board):  # If the board changed
                moves.append(direction)
            self.board = temp_board
            self.score = temp_score
        return moves

    def is_game_over(self):
        """Check if there are no valid moves left."""
        return len(self.available_moves()) == 0

    def display(self):
        """Print out the current game board and score."""
        for row in self.board:
            print("\t".join(map(str, row)))
        print(f"Score: {self.score}\n")
