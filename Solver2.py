import numpy as np


class State:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.num_moves = 0

    def can_play(self, col):
        return self.board[0][col] == 0

    def play(self, col, piece):
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = piece
                break
        self.num_moves += 1
    def revert(self, col, piece):
        for row in range( 0,self.rows ):
            if self.board[row][col] == piece:
                self.board[row][col] = 0
                break
        self.num_moves -= 1
    def is_winning_move(self, piece):
        # Check horizontal locations
        for c in range(self.cols - 3):
            for r in range(self.rows):
                if all(self.board[r, c + i] == piece for i in range(4)):
                    return True
        # Check vertical locations
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if all(self.board[r + i, c] == piece for i in range(4)):
                    return True
        # Check positively sloped diagonals
        for c in range(self.cols - 3):
            for r in range(self.rows - 3):
                if all(self.board[r + i, c + i] == piece for i in range(4)):
                    return True
        # Check negatively sloped diagonals
        for c in range(self.cols - 3):
            for r in range(3, self.rows):
                if all(self.board[r - i, c + i] == piece for i in range(4)):
                    return True
        return False

    def get_valid_columns(self):
        return [c for c in range(self.cols) if self.can_play(c)]

    def copy(self):
        new_state = State(self.rows, self.cols)
        new_state.board = self.board.copy()
        new_state.num_moves = self.num_moves
        return new_state


# Solver class for AI
class Solver:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.transposition_table = {}

    def minimax(self, state, depth, alpha, beta, maximizingPlayer, piece):
        opponent_piece = 1 if piece == 2 else 2
        valid_columns = state.get_valid_columns()
        # valid_columns.sort(key=lambda c: abs(c - state.cols // 2))


        if depth == 0 or not valid_columns:
            return None,self.evaluate_board(state, piece)

        if state.is_winning_move(piece):
            return None, 1000000 - state.num_moves
        if state.is_winning_move(opponent_piece):
            return None, -1000000 + state.num_moves
        # board_hash = str(state.board)
        # if board_hash in self.transposition_table:
        #     return self.transposition_table[board_hash]
        best_column, best_value = 0, 0
        if maximizingPlayer:
            value = float('-inf')
            best_col = valid_columns[0]
            for col in valid_columns:
                # next_state = state.copy()
                state.play(col, piece)
                _, score = self.minimax(state, depth - 1, alpha, beta, False, piece)
                if score > value:
                    value = score
                    best_col = col
                state.revert(col,piece)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            best_column, best_value = best_col, value


        else:
            value = float('inf')
            best_col = valid_columns[0]
            for col in valid_columns:
                # next_state = state.copy()
                state.play(col, opponent_piece)
                _, score = self.minimax(state, depth - 1, alpha, beta, True, piece)
                if score < value:
                    value = score
                    best_col = col
                state.revert(col, opponent_piece)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            best_column, best_value = best_col, value

        # self.transposition_table[board_hash] = (best_column,best_value)
        # if depth == self.max_depth-1:
        #     state.play(best_column, 3-piece)
        #     print(state.board,best_column, best_value)
        #     state.revert(best_column, 3-piece)
        return best_column, best_value
    def evaluate_board(self, state, piece):
        score = self.evaluate_board2(state, piece)
        opponent_piece = 1 if piece == 2 else 2
        for col in range(state.cols):
            if state.can_play(col):
                # temp_state = state.copy()
                state.play(col, opponent_piece)
                if state.is_winning_move(opponent_piece):
                    score -= 100000
                state.revert(col, opponent_piece)
        center_array = state.board[:, state.cols // 2]
        center_count = np.count_nonzero(center_array == piece)
        score += center_count * 3

        for r in range(state.rows):
            row_array = state.board[r, :]
            for c in range(state.cols - 3):
                window = row_array[c:c + 4]
                score += self.evaluate_window(window, piece)

        for c in range(state.cols):
            col_array = state.board[:, c]
            for r in range(state.rows - 3):
                window = col_array[r:r + 4]
                score += self.evaluate_window(window, piece)

        for r in range(state.rows - 3):
            for c in range(state.cols - 3):
                window = [state.board[r + i][c + i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        for r in range(state.rows - 3):
            for c in range(state.cols - 3):
                window = [state.board[r + 3 - i][c + i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        return score

    def evaluate_window(self, window, piece):
        score = 0
        opponent_piece = 1 if piece == 2 else 2

        if np.count_nonzero(window == piece) == 4:
            score += 100
        elif np.count_nonzero(window == piece) == 3 and np.count_nonzero(window == 0) == 1:
            score += 15
        elif np.count_nonzero(window == piece) == 2 and np.count_nonzero(window == 0) == 2:
            score += 2

        if np.count_nonzero(window == opponent_piece) == 3 and np.count_nonzero(window == 0) == 1:
            score -= 4

        return score



    def evaluate_board2(self,state, player):
        board = state.board
        """Evaluate the Connect Four board using Heuristic-2."""
        opponent = 3 - player
        heuristic_matrix = np.array([
            [3, 4, 5, 7, 5, 4, 3],
            [4, 6, 8, 10, 8, 6, 4],
            [5, 8, 11, 13, 11, 8, 5],
            [5, 8, 11, 13, 11, 8, 5],
            [4, 6, 8, 10, 8, 6, 4],
            [3, 4, 5, 7, 5, 4, 3],
        ])

        score = 0
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row, col] == player:
                    score += heuristic_matrix[row, col]
                elif board[row, col] == opponent:
                    score -= heuristic_matrix[row, col]

        return score
if __name__ == '__main__':
    arr = np.array([[0. ,0., 0., 0., 0., 0., 0.],
                     [0., 2., 1., 2., 0., 0., 0.],
                     [0., 1., 1., 1., 0., 0. ,0.],
                     [0., 1. ,2. ,2., 0., 0. ,0.],
                     [1. ,2. ,2. ,2. ,0. ,0. ,0.],
                     [1. ,2. ,1., 1. ,0., 0. ,0.]])
    state = State(6,7)
    state.board = arr
    state.num_moves = 15
    sol = Solver(max_depth=3)
    print(sol.minimax(state, depth=3, alpha=float('-inf'), beta=float('inf'), maximizingPlayer=True,
                                    piece=2))