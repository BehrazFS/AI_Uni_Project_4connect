import numpy as np
import pygame
import sys
import math


class State:
    def __init__(self, ROW_COUNT, COLUMN_COUNT):
        self.width = COLUMN_COUNT
        self.height = ROW_COUNT
        self.board = np.zeros((self.width, self.height))
        self.heights = np.array([0 for _ in range(self.width)])
        self.num_moves = 0

    def get_area(self):
        return self.width * self.height

    def get_num_moves(self):
        return self.num_moves

    def can_play(self, col):
        return self.heights[col] < self.height

    def play(self, col):
        self.board[col][self.heights[col]] = 1 + self.num_moves % 2
        self.heights[col] += 1
        self.num_moves += 1

    def is_winning_move(self, col):
        cur_player = 1 + self.num_moves % 2
        if self.heights[col] >= 3 and self.board[col][self.heights[col] - 1] == cur_player and self.board[col][
            self.heights[col] - 2] == cur_player and self.board[col][self.heights[col] - 3] == cur_player:
            return True
        for dy in range(-1, 2):
            nb = 0
            for dx in range(-1, 2, 2):
                x = col + dx
                y = self.heights[col] + dx * dy
                while 0 <= x < self.width and 0 <= y < self.height and self.board[x][y] == cur_player:
                    x += dx
                    y += dy * dx
                    nb += 1
            if nb >= 3:
                return True
        return False

    def play_line(self, line):
        for i in range(len(line)):
            col = int(ord(line[i]) - ord('1'))
            if col < 0 or col >= self.width or not self.can_play(col) or self.is_winning_move(col):
                return i
            self.play(col)
        return len(line)

    def copy(self, state):
        self.board = state.board.copy()
        self.num_moves = state.num_moves
        self.heights = state.heights.copy()


class Solver:
    def __init__(self, max_depth,width):
        self.max_depth = max_depth
        # self.column_order = []
        # for i in range(width):
        #     self.column_order.append( state.width // 2 + (1 - 2 * (i % 2)) * (i + 1) // 2)
        # self.column_order = np.array(self.column_order)
        self.column_order = sorted(range(width), key=lambda x: abs(width // 2 - x))

    def solve(self, state):

        return self.negamax(state, 0, state.get_area() // 2, state.get_area() // 2)#


    def negamax(self, state: State, depth, alpha, beta):
        if state.get_num_moves() == state.get_area():
            return 0
        for x in range(state.width):
            if state.can_play(x) and state.is_winning_move(x):
                return (state.get_area() + 1 - state.get_num_moves()) // 2
        if depth >= self.max_depth:
            def evaluate_window(window, piece):
                opponent_piece = 3 - piece
                score = 0

                if window.count(piece) == 4:
                    score += 100
                elif window.count(piece) == 3 and window.count(0) == 1:
                    score += 10
                elif window.count(piece) == 2 and window.count(0) == 2:
                    score += 5

                if window.count(opponent_piece) == 3 and window.count(0) == 1:
                    score -= 80  # Block opponent's winning move

                return score

            def heuristic(state: State):
                piece = 1 + (state.num_moves % 2)  # Current player
                opponent_piece = 3 - piece
                score = 0

                # Center column preference
                center_array = [state.board[state.width // 2][i] for i in range(state.height)]
                center_count = center_array.count(piece)
                score += center_count * 6

                # Horizontal scoring
                for r in range(state.height):
                    row_array = [state.board[c][r] for c in range(state.width)]
                    for c in range(state.width - 3):
                        window = row_array[c:c + 4]
                        score += evaluate_window(window, piece)

                # Vertical scoring
                for c in range(state.width):
                    col_array = [state.board[c][r] for r in range(state.height)]
                    for r in range(state.height - 3):
                        window = col_array[r:r + 4]
                        score += evaluate_window(window, piece)

                # Positive diagonal scoring
                for c in range(state.width - 3):
                    for r in range(state.height - 3):
                        window = [state.board[c + i][r + i] for i in range(4)]
                        score += evaluate_window(window, piece)

                # Negative diagonal scoring
                for c in range(state.width - 3):
                    for r in range(3, state.height):
                        window = [state.board[c + i][r - i] for i in range(4)]
                        score += evaluate_window(window, piece)

                return score

            # print("ended")
            # print(depth)
            print("e")
            return heuristic(state)
        # print("not ended")

        max_score = (state.get_area() - 1 - state.get_num_moves()) // 2
        if beta > max_score:
            beta = max_score
            if alpha >= beta:
                return beta
        for col in self.column_order:
            if state.can_play(col):
                temp = State(state.height, state.width)
                temp.copy(state)

                temp.play(self.column_order[x])  # player one is us not the bot

                score = -self.negamax(temp, depth + 1, -beta, -alpha)

                if score >= beta:
                    return score
                if score > alpha:
                    alpha = score

        return alpha

gui = True
if gui:
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)

    ROW_COUNT = 6
    COLUMN_COUNT = 7


    def create_board():
        board = np.zeros((ROW_COUNT, COLUMN_COUNT))
        return board


    def drop_piece(board, row, col, piece):
        board[row][col] = piece


    def is_valid_location(board, col):
        return board[ROW_COUNT - 1][col] == 0


    def get_next_open_row(board, col):
        for r in range(ROW_COUNT):
            if board[r][col] == 0:
                return r


    def print_board(board):
        print(np.flip(board, 0))


    def winning_move(board, piece):
        # Check horizontal locations for win
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                    c + 3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                    c] == piece:
                    return True

        # Check positively sloped diaganols
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and \
                        board[r + 3][
                            c + 3] == piece:
                    return True

        # Check negatively sloped diaganols
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and \
                        board[r - 3][
                            c + 3] == piece:
                    return True


    def draw_board(board):
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(screen, BLACK, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                if board[r][c] == 1:
                    pygame.draw.circle(screen, RED, (
                        int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
                elif board[r][c] == 2:
                    pygame.draw.circle(screen, YELLOW, (
                        int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
        pygame.display.update()


    board = create_board()
    print_board(board)
    game_over = False
    turn = 0

    state = State(ROW_COUNT, COLUMN_COUNT)
    solve = Solver(3,state.width)
    pygame.init()

    SQUARESIZE = 100

    width = COLUMN_COUNT * SQUARESIZE
    height = (ROW_COUNT + 1) * SQUARESIZE

    size = (width, height)

    RADIUS = int(SQUARESIZE / 2 - 5)

    screen = pygame.display.set_mode(size)
    draw_board(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 75)

    while not game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                if turn == 0:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    posx = event.pos[0]
                    if turn == 0:
                        pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
                    else:
                        pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE / 2)), RADIUS)
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                # print(event.pos)
                # Ask for Player 1 Input
                if turn == 0:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    if is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, 1)
                        state.play(col)
                        if winning_move(board, 1):
                            label = myfont.render("Player 1 wins!!", 1, RED)
                            screen.blit(label, (40, 10))
                            game_over = True


                # # Ask for Player 2 Input
                else:
                    # posx = event.pos[0]
                    # col = int(math.floor(posx / SQUARESIZE))
                    best_move = -1
                    max_score = -float('inf')
                    for x in range(COLUMN_COUNT):
                        temp = State(ROW_COUNT, COLUMN_COUNT)
                        temp.copy(state)
                        if temp.can_play(x):
                            temp.play(x)
                            s = solve.solve(temp)
                            print(f"Column: {x}, Score: {s}")
                            if s > max_score:
                                max_score = s
                                best_move = x


                    col = best_move

                    if is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, 2)
                        state.play(col)
                        if winning_move(board, 2):
                            label = myfont.render("Player 2 wins!!", 1, YELLOW)
                            screen.blit(label, (40, 10))
                            game_over = True

                print_board(board)
                print(solve.solve(state))
                print(np.rot90(state.board))
                draw_board(board)

                turn += 1
                turn = turn % 2
                if state.num_moves == 6*7:
                    game_over = True
                if game_over:
                    pygame.time.wait(3000)
else:
    inp = input()
    state = State(6, 7)
    if state.play_line(inp) != len(inp):
        print("Invalid input")
    else:
        solve = Solver(1000)
        print(solve.solve(state))
