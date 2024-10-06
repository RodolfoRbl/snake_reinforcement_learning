import multiprocessing as mp
from game import SnakeGameAI
import sys
import time
import pygame
from agent import Agent
import os


def get_best_model_path():
    """
    Retrieve the model file with the highest score.
    """
    model_files = [f for f in os.listdir("./model/") if f.startswith("model_score_")]
    if not model_files:
        return None, 0  # No model file found

    # Extract scores from filenames and find the maximum
    scores = [int(f.split("_")[-1].split(".")[0]) for f in model_files]
    max_score = max(scores)
    best_model_path = os.path.join("./model/", f"model_score_{max_score}.pth")
    return best_model_path, max_score


def run_snake_instance(agent: Agent, game_idx: int, memory_queue: mp.Queue):
    """
    Adding experiences to memory
    """
    pygame.init()  # Initialize pygame in each process
    game = SnakeGameAI()
    iteration = 0
    best_score = 0
    experiences = []
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        # Send experience tuple (state, action, reward, next_state, done) to the memory queue
        experiences.append((state_old, final_move, reward, state_new, done))
        if done:
            print(f'length experiences proc {game_idx}: {len(experiences)}')
            memory_queue.put(experiences)
            experiences = []
            game.reset()
            iteration += 1
            agent.n_games += 2
            if score > best_score:
                best_score = score
                if f"model_score_{score}.pth" not in [i for i in os.listdir("./model/")]:
                    agent.model.save(f"./model/model_score_{score}.pth")
                    print(f"Saved model with score {game_idx}")
            print("Process: ", game_idx, "Iteration", iteration, "Score", score, "Memory size", len(agent.memory))


def train_agent(agent: Agent, memory_queue):
    best_model_path, max_score = get_best_model_path()
    if best_model_path:
        agent.model.load(best_model_path)
    while True:
        if not memory_queue.empty():
            game_experiences = memory_queue.get()
            [agent.remember(state_old, final_move, reward, state_new, done) for state_old, final_move, reward, state_new, done in game_experiences]
            agent.train_long_memory()
        else:
            time.sleep(1)


if __name__ == "__main__":

    agent = Agent()
    agent.model.share_memory()
    memory_queue = mp.Queue()
    stop_event = mp.Event()
    # Create multiple processes for the Snake game instances
    num_games = 8
    processes = []

    try:
        for i in range(num_games):
            p = mp.Process(target=run_snake_instance, args=(agent, i, memory_queue))
            processes.append(p)
            p.start()

        # Start the trainin g process
        train_agent(agent, memory_queue)

        # Wait for all processes to complete
        for p in processes:
            p.join()

    except Exception as e:
        print(e)
    finally:
        stop_event.set()
        print("finally")
        for p in processes:
            p.terminate()  # Ensure all processes are terminated in any case
            p.join()
        print("end")
