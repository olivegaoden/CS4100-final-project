import csv
import itertools
import os
import numpy as np
from game import Game2048
from util import ExpectimaxAI

# Initialize file names
NO_HEURISTICS_FILENAME = "baseline.csv"
INDIVIDUAL_HEURISTICS_FILENAME = "heuristic_individual_results.csv"
HEURISTIC_COMBINATIONS_FILENAME = "heuristic_combinations_results.csv"
WEIGHTS_FILENAME = "weights_results.csv"
WEIGHTS_2_FILENAME = "weights_2_results.csv"
WEIGHTS_3_FILENAME = "weights_3_results.csv"
DEPTH_TUNING_FILENAME = "depth_tuning_results.csv"
DEPTH_TUNING_2_FILENAME = "depth_tuning_2_results.csv"
DEPTH_TUNING_3_FILENAME = "depth_tuning_3_results.csv"
BEST_HEURISTICS_FILENAME = "best_heuristics_results.csv"

# Run game with given parameters
def run_game_with_weights(weights, depth=3):
    game = Game2048()
    ai = ExpectimaxAI(game, depth=depth, weights=weights)

    while not game.is_game_over():
        move = ai.get_best_move()
        if move:
            game.move(move)
        else:
            break
    return game.score, np.max(game.board)

# Get results fro different weights and averages across trials
def evaluate_weights(weights, num_trials=3, depth=3):
    scores = []
    max_tiles = []

    for _ in range(num_trials):
        score, max_tile = run_game_with_weights(weights, depth=depth)
        scores.append(score)
        max_tiles.append(max_tile)

    return {
        "weights": weights,
        "avg_score": np.mean(scores),
        "avg_max_tile": np.mean(max_tiles),
        "scores": scores,
        "max_tiles": max_tiles,
    }

# Method to test the baseline of no heuristics
def test_no_heuristics():
    # If evaluation already run, don't run it again
    if not os.path.exists(NO_HEURISTICS_FILENAME) or not os.path.getsize(NO_HEURISTICS_FILENAME) > 0:
        weights = {
            "max_tile": 0,
            "empty_cells": 0,
            "monotonicity": 0,
            "smoothness": 0,
            "corner_bonus": 0
        }

        result = evaluate_weights(weights, num_trials=3)

        with open(NO_HEURISTICS_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["avg_score", "avg_max_tile", "scores", "max_tiles"])
            row = [result["avg_score"], result["avg_max_tile"], str(result["scores"]), str(result["max_tiles"])]
            writer.writerow(row)
    else:
        print(f"{NO_HEURISTICS_FILENAME} already contains results, skipping baseline evaluation.")

# Method to do evaluation on individual heuristics
def individual_heuristic_evaluation():
    heuristics = ["max_tile", "empty_cells", "monotonicity", "smoothness", "corner_bonus"]

    # If evaluation already run, don't run it again
    if not os.path.exists(INDIVIDUAL_HEURISTICS_FILENAME) or os.path.getsize(INDIVIDUAL_HEURISTICS_FILENAME) == 0:
        with open(INDIVIDUAL_HEURISTICS_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["heuristic", "weight", "avg_score", "avg_max_tile"])

            for heuristic in heuristics:
                # Only set current heuristic to 1 and others to 0 to make sure we're testing only that heuristic
                weights = {h: 0.0 for h in heuristics}
                weights[heuristic] = 1.0

                result = evaluate_weights(weights, num_trials=3)
                print(f"{heuristic:<15} → Score: {result['avg_score']:.1f}, Max Tile: {result['avg_max_tile']:.1f}")

                writer.writerow([heuristic, 1.0, result['avg_score'], result['avg_max_tile']])
    else:
        print(f"{INDIVIDUAL_HEURISTICS_FILENAME} already contains results, skipping individual heuristic evaluation.")

# Method to do evaluation on different combinations of heuristics
def test_heuristic_combinations():
    # If evaluation already run, don't run it again
    if not os.path.exists(HEURISTIC_COMBINATIONS_FILENAME) or os.path.getsize(HEURISTIC_COMBINATIONS_FILENAME) == 0:
        # From individual heuristics run, we know that empty_cells, monotonicity, and smoothness
        # were the best heuristics, so we only do combinations with those heuristics
        weight_grid = {
            "max_tile": [0],
            "empty_cells": [0, 1],
            "monotonicity": [0, 1],
            "smoothness": [0, 1],
            "corner_bonus": [0],
        }

        keys = list(weight_grid.keys())
        combinations = list(itertools.product(*[weight_grid[k] for k in keys]))

        all_results = []
        total_combos = len(combinations)

        with open(HEURISTIC_COMBINATIONS_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["avg_score", "avg_max_tile"] + keys + ["scores", "max_tiles"])

            for i, values in enumerate(combinations, 1):
                weights = dict(zip(keys, values))
                result = evaluate_weights(weights, num_trials=3)
                result["weights"] = weights
                all_results.append(result)

                print(f"[{i}/{total_combos}] Avg Score: {result['avg_score']:.1f} | Max Tile: {result['avg_max_tile']:.1f}")
                print(f"   Weights: " + ", ".join(f"{k}={v}" for k, v in weights.items()))

                row = [result["avg_score"], result["avg_max_tile"]] + [weights[k] for k in keys] + [str(result["scores"]), str(result["max_tiles"])]
                writer.writerow(row)
    else:
        print(f"{HEURISTIC_COMBINATIONS_FILENAME} already contains results, skipping heuristic combinations evaluation.")

# Method to do evaluation on what weights the heuristics should be set to
def test_heuristic_weights():
    # If evaluation already run, don't run it again
    if not os.path.exists(WEIGHTS_FILENAME) or not os.path.getsize(WEIGHTS_FILENAME) > 0:
        # From heuristics combinations run, we know that empty cells and mmonotonicity was the best
        # combination of heuristics, so we only do combinations with those heuristics
        weight_grid = {
            "max_tile": [0],
            "empty_cells": [0],
            "monotonicity": [0.5, 1, 2],
            "smoothness": [0.5, 1, 2],
            "corner_bonus": [0]
        }

        keys = list(weight_grid.keys())
        combinations = list(itertools.product(*[weight_grid[k] for k in keys]))

        all_results = []
        total_combos = len(combinations)

        with open(WEIGHTS_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["avg_score", "avg_max_tile", "monotonicity", "smoothness", "weights"])

            for i, values in enumerate(combinations, 1):
                weights = dict(zip(keys, values))
                result = evaluate_weights(weights, num_trials=3)
                result["weights"] = weights  # Store for top 5 printing
                all_results.append(result)

                result["monotonicity"] = weights["monotonicity"]
                result["smoothness"] = weights["smoothness"]

                print(f"[{i}/{total_combos}] Avg Score: {result['avg_score']:.1f} | Max Tile: {result['avg_max_tile']:.1f}")
                print(f"   Weights: " + ", ".join(f"{k}={v}" for k, v in weights.items()))

                row = [result["avg_score"], result["avg_max_tile"]] + [weights[k] for k in keys] + [str(result["scores"]), str(result["max_tiles"])]
                writer.writerow(row)
    else:
        print(f"{WEIGHTS_FILENAME} already contains results, skipping heuristic tuning.")

# Method to do evaluation on what weights the heuristics should be set to for a different combination of heuristics
def test_heuristic_weights_2():
    # If evaluation already run, don't run it again
    if not os.path.exists(WEIGHTS_2_FILENAME) or not os.path.getsize(WEIGHTS_2_FILENAME) > 0:
        # From heuristics combinations run, we know that empty cells and mmonotonicity was the best
        # combination of heuristics, so we only do combinations with those heuristics
        weight_grid = {
            "max_tile": [0],
            "empty_cells": [0.5, 1, 2],
            "monotonicity": [0.5, 1, 2],
            "smoothness": [0],
            "corner_bonus": [0]
        }

        keys = list(weight_grid.keys())
        combinations = list(itertools.product(*[weight_grid[k] for k in keys]))

        all_results = []
        total_combos = len(combinations)

        with open(WEIGHTS_2_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["avg_score", "avg_max_tile", "monotonicity", "smoothness", "weights"])

            for i, values in enumerate(combinations, 1):
                weights = dict(zip(keys, values))
                result = evaluate_weights(weights, num_trials=3)
                result["weights"] = weights  # Store for top 5 printing
                all_results.append(result)

                result["monotonicity"] = weights["monotonicity"]
                result["smoothness"] = weights["smoothness"]

                print(f"[{i}/{total_combos}] Avg Score: {result['avg_score']:.1f} | Max Tile: {result['avg_max_tile']:.1f}")
                print(f"   Weights: " + ", ".join(f"{k}={v}" for k, v in weights.items()))

                row = [result["avg_score"], result["avg_max_tile"]] + [weights[k] for k in keys] + [str(result["scores"]), str(result["max_tiles"])]
                writer.writerow(row)
    else:
        print(f"{WEIGHTS_2_FILENAME} already contains results, skipping heuristic tuning.")

# Method to do evaluation on what weights the heuristics should be set to for a second different combination of heuristics
def test_heuristic_weights_3():
    # If evaluation already run, don't run it again
    if not os.path.exists(WEIGHTS_3_FILENAME) or not os.path.getsize(WEIGHTS_3_FILENAME) > 0:
        # From heuristics combinations run, we know that empty cells and mmonotonicity was the best
        # combination of heuristics, so we only do combinations with those heuristics
        weight_grid = {
            "max_tile": [0],
            "empty_cells": [0, 0.5, 1, 2],
            "monotonicity": [0, 0.5, 1, 2],
            "smoothness": [0, 0.5, 1, 2],
            "corner_bonus": [0]
        }

        keys = list(weight_grid.keys())
        combinations = list(itertools.product(*[weight_grid[k] for k in keys]))

        all_results = []
        total_combos = len(combinations)

        with open(WEIGHTS_3_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["avg_score", "avg_max_tile", "monotonicity", "smoothness", "weights"])

            for i, values in enumerate(combinations, 1):
                weights = dict(zip(keys, values))
                result = evaluate_weights(weights, num_trials=3)
                result["weights"] = weights  # Store for top 5 printing
                all_results.append(result)

                result["monotonicity"] = weights["monotonicity"]
                result["smoothness"] = weights["smoothness"]

                print(f"[{i}/{total_combos}] Avg Score: {result['avg_score']:.1f} | Max Tile: {result['avg_max_tile']:.1f}")
                print(f"   Weights: " + ", ".join(f"{k}={v}" for k, v in weights.items()))

                row = [result["avg_score"], result["avg_max_tile"]] + [weights[k] for k in keys] + [str(result["scores"]), str(result["max_tiles"])]
                writer.writerow(row)
    else:
        print(f"{WEIGHTS_3_FILENAME} already contains results, skipping heuristic tuning.")

# Method to do evaluation on what depth is the most effective
def test_different_depths():
    # If evaluation already run, don't run it again
    if not os.path.exists(DEPTH_TUNING_FILENAME) or not os.path.getsize(DEPTH_TUNING_FILENAME) > 0:
        depth_values = [2, 3, 4]

        # From heuristics weights run, we know that empty cells as 1 and monotonicity as 2 is the best
        # combination and weight of heuristics, so we only do different depths with those heuristics
        weights = {
            "max_tile": 0,
            "empty_cells": 1,
            "monotonicity": 2,
            "smoothness": 0,
            "corner_bonus": 0
        }

        all_results = []

        with open(DEPTH_TUNING_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["avg_score", "avg_max_tile", "depth", "weights"])

            for depth in depth_values:
                result = evaluate_weights(weights, num_trials=3, depth=depth)
                result["depth"] = depth
                result["weights"] = weights
                all_results.append(result)

                print(f"Depth {depth}: Avg Score: {result['avg_score']:.1f} | Max Tile: {result['avg_max_tile']:.1f}")

                row = [result["avg_score"], result["avg_max_tile"], depth] + [weights[k] for k in weights]
                writer.writerow(row)
    else:
        print(f"{DEPTH_TUNING_FILENAME} already contains results, skipping depth tuning.")

def test_different_depths_2():
    # If evaluation already run, don't run it again
    if not os.path.exists(DEPTH_TUNING_2_FILENAME) or not os.path.getsize(DEPTH_TUNING_2_FILENAME) > 0:
        depth_values = [2, 3, 4, 5]

        # From heuristics weights run, we know that empty cells as 1 and monotonicity as 2 is the best
        # combination and weight of heuristics, so we only do different depths with those heuristics
        weights = {
            "max_tile": 0,
            "empty_cells": 1,
            "monotonicity": 2,
            "smoothness": 0.5,
            "corner_bonus": 0
        }

        all_results = []

        with open(DEPTH_TUNING_2_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["avg_score", "avg_max_tile", "depth", "weights"])

            for depth in depth_values:
                result = evaluate_weights(weights, num_trials=3, depth=depth)
                result["depth"] = depth
                result["weights"] = weights
                all_results.append(result)

                print(f"Depth {depth}: Avg Score: {result['avg_score']:.1f} | Max Tile: {result['avg_max_tile']:.1f}")

                row = [result["avg_score"], result["avg_max_tile"], depth] + [weights[k] for k in weights]
                writer.writerow(row)
    else:
        print(f"{DEPTH_TUNING_2_FILENAME} already contains results, skipping depth tuning.")

def test_different_depths_3():
    # If evaluation already run, don't run it again
    if not os.path.exists(DEPTH_TUNING_3_FILENAME) or not os.path.getsize(DEPTH_TUNING_3_FILENAME) > 0:
        depth_values = [2, 3, 4]

        # From heuristics weights run, we know that monotonicity as 2 and smoothness as 1 is the best
        # combination and weight of heuristics, so we only do different depths with those heuristics
        weights = {
            "max_tile": 0,
            "empty_cells": 0,
            "monotonicity": 2,
            "smoothness": 1,
            "corner_bonus": 0
        }

        all_results = []

        with open(DEPTH_TUNING_3_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["avg_score", "avg_max_tile", "depth", "weights"])

            for depth in depth_values:
                result = evaluate_weights(weights, num_trials=3, depth=depth)
                result["depth"] = depth
                result["weights"] = weights
                all_results.append(result)

                print(f"Depth {depth}: Avg Score: {result['avg_score']:.1f} | Max Tile: {result['avg_max_tile']:.1f}")

                row = [result["avg_score"], result["avg_max_tile"], depth] + [weights[k] for k in weights]
                writer.writerow(row)
    else:
        print(f"{DEPTH_TUNING_3_FILENAME} already contains results, skipping depth tuning.")

def test_best_combination():
    # If evaluation already run, don't run it again
    if not os.path.exists(BEST_HEURISTICS_FILENAME) or not os.path.getsize(BEST_HEURISTICS_FILENAME) > 0:
        weights = {
            "max_tile": 0,
            "empty_cells": 1,
            "monotonicity": 2,
            "smoothness": 0,
            "corner_bonus": 0
        }

        depth = 4

        with open(BEST_HEURISTICS_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["avg_score", "avg_max_tile"])
            result = evaluate_weights(weights, num_trials=5, depth=depth)

            print(f"Avg Score: {result['avg_score']:.1f} | Max Tile: {result['avg_max_tile']:.1f}")

            row = [result["avg_score"], result["avg_max_tile"]]
            writer.writerow(row)
    else:
        print(f"{BEST_HEURISTICS_FILENAME} already contains results, skipping depth tuning.")

def test_ai_selects_optimal_move_in_various_scenarios():
    test_cases = [
        {
            "board": np.array([
                [2, 2, 4, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            "expected_move": "left"
        },
        {
            "board": np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [2, 0, 0, 0],
                [2, 0, 0, 0]
            ]),
            "expected_move": "up"
        },
        {
            "board": np.array([
                [4, 0, 0, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            "expected_move": ["left", "right"]
        },
    ]

    weights = {
        "max_tile": 0,
        "empty_cells": 1,
        "monotonicity": 2,
        "smoothness": 0,
        "corner_bonus": 0
    }

    print("AI with heuristics")
    for i, test_case in enumerate(test_cases):
        game = Game2048()
        game.board = test_case["board"]
        ai = ExpectimaxAI(game, depth=4, weights=weights)
        best_move = ai.get_best_move()
        print(f"Test {i+1}: AI chose '{best_move}' (expected '{test_case['expected_move']}')")

def test_ai_no_heuristics_doesnt_select_optimal_moves():
    test_cases = [
        {
            "board": np.array([
                [2, 2, 4, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            "expected_move": "left"
        },
        {
            "board": np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [2, 0, 0, 0],
                [2, 0, 0, 0]
            ]),
            "expected_move": "up"
        },
        {
            "board": np.array([
                [4, 0, 0, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            "expected_move": ["left", "right"]
        },
    ]

    weights = {
        "max_tile": 0,
        "empty_cells": 0,
        "monotonicity": 0,
        "smoothness": 0,
        "corner_bonus": 0
    }

    print("AI without heuristics")
    for i, test_case in enumerate(test_cases):
        game = Game2048()
        game.board = test_case["board"]
        ai = ExpectimaxAI(game, depth=4, weights=weights)
        best_move = ai.get_best_move()
        print(f"Test {i+1}: AI chose '{best_move}' (expected '{test_case['expected_move']}')")

if __name__ == "__main__":
    test_no_heuristics()
    individual_heuristic_evaluation()
    test_heuristic_combinations()
    test_heuristic_weights()
    test_heuristic_weights_2()
    test_heuristic_weights_3()
    test_different_depths()
    test_different_depths_2()
    test_different_depths_3()
    test_best_combination()
    test_ai_selects_optimal_move_in_various_scenarios()
    test_ai_no_heuristics_doesnt_select_optimal_moves()