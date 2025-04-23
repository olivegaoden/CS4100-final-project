import time
from game import Game2048
from util import ExpectimaxAI

def play_game():
    game = Game2048()
    ai = ExpectimaxAI(game)

    total_time = 0
    move_count = 0

    while True:
        # Print intial board state
        game.display()

        # Get time it takes for AI to decide on a move
        start_time = time.time()
        move = ai.get_best_move()
        end_time = time.time()

        elapsed_time = end_time - start_time
        total_time += elapsed_time
        move_count += 1

        
        if move:
            print(f"AI chooses: {move} (took {elapsed_time:.3f} seconds)")
            game.move(move)
        else:
            print("Game Over!")
            print("Score:", game.score)
            print("Total Moves:", move_count)
            print(f"Total Time: {total_time:.3f} seconds")
            print(f"Average Time per Move: {total_time / move_count:.3f} seconds")
            break 

if __name__ == "__main__":
    play_game()