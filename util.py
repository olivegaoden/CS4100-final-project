import numpy as np

from game import Game2048

class ExpectimaxAI:
    def __init__(self, game: Game2048, depth=4, weights=None):
        self.game = game
        self.depth = depth
        self.weights = weights or {
            "max_tile": 0,
            "empty_cells": 1,
            "monotonicity": 2,
            "smoothness": 0,
            "corner_bonus": 0,
        }

    def get_best_move(self):
        best_move = None
        best_score = float('-inf')
        
        # Try all available moves and pick the one with the highest score
        for move in self.game.available_moves():
            temp_board = self.simulate_move(move)
            score = self.expectimax(temp_board, self.depth, False)
            
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def expectimax(self, board, depth, is_max):
        if depth == 0 or not self.get_available_moves(board):
            return self.evaluate_board(board)

        # Maximize score if AI's turn
        if is_max:
            best_score = float('-inf')
            for move in self.get_available_moves(board):
                temp_board = self.simulate_move_on_board(board, move)
                best_score = max(best_score, self.expectimax(temp_board, depth - 1, False))
            return best_score
        # Minimize score if random tile placement turn
        else:
            empty_cells = [(r, c) for r in range(4) for c in range(4) if board[r, c] == 0]
            if not empty_cells:
                return self.evaluate_board(board)

            expected_value = 0
            for (r, c) in empty_cells:
                for tile, prob in [(2, 0.9), (4, 0.1)]:
                    temp_board = board.copy()
                    temp_board[r, c] = tile
                    expected_value += prob * self.expectimax(temp_board, depth - 1, True)
            return expected_value / len(empty_cells)

    def simulate_move(self, move):
        temp_game = self.copy_game()
        temp_game.move(move, simulate=True)
        return temp_game.board

    def simulate_move_on_board(self, board, move):
        temp_game = self.copy_game_with_board(board)
        temp_game.move(move, simulate=True)
        return temp_game.board

    def evaluate_board(self, board):
        weights = self.weights
        max_tile = np.max(board)
        empty_cells = np.count_nonzero(board == 0)
        monotonicity = self.calculate_monotonicity(board)
        smoothness = self.calculate_smoothness(board)
        corner_bonus = self.corner_max_tile(board)

        score = (
            weights["max_tile"] * max_tile +
            weights["empty_cells"] * empty_cells +
            weights["monotonicity"] * monotonicity +
            weights["smoothness"] * smoothness +
            weights["corner_bonus"] * corner_bonus
        )
        return score

    def get_available_moves(self, board):
        temp_game = self.copy_game_with_board(board)
        return temp_game.available_moves()

    def copy_game(self):
        from game import Game2048 
        new_game = Game2048()
        new_game.board = self.game.board.copy()
        return new_game

    def copy_game_with_board(self, board, score=0):
        from game import Game2048 
        new_game = Game2048()
        new_game.board = board.copy()
        new_game.score = score
        return new_game
    
    def calculate_monotonicity(self, board):
        score = 0
        for row in board:
            vals = row[row > 0]
            if np.all(np.diff(vals) >= 0) or np.all(np.diff(vals) <= 0):
                score += 1
        for col in board.T:
            vals = col[col > 0]
            if np.all(np.diff(vals) >= 0) or np.all(np.diff(vals) <= 0):
                score += 1
        return score

    def calculate_smoothness(self, board):
        smoothness = 0
        for r in range(4):
            for c in range(3):
                if board[r][c] and board[r][c+1]:
                    smoothness -= abs(board[r][c] - board[r][c+1])
        for c in range(4):
            for r in range(3):
                if board[r][c] and board[r+1][c]:
                    smoothness -= abs(board[r][c] - board[r+1][c])
        return smoothness

    def corner_max_tile(self, board):
        max_tile = np.max(board)
        corners = [board[0][0], board[0][3], board[3][0], board[3][3]]
        return 1 if max_tile in corners else 0
