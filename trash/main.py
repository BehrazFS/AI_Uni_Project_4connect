# if turn == 0 and not game_over:
        #     pygame.time.wait(500)
        #     col, _ = solver.minimax(state, depth=5, alpha=float('-inf'), beta=float('inf'), maximizingPlayer=True,
        #                             piece=1)
        #     if is_valid_location(board, col):
        #         row = get_next_open_row(board, col)
        #         drop_piece(board, row, col, 1)
        #         state.play(col, 1)
        #         if winning_move(board, 1):
        #             label = myfont.render("Player 1 wins!!", 1, RED)
        #             screen.blit(label, (40, 10))
        #             game_over = True
        #         draw_board(board)
        #         turn = 1