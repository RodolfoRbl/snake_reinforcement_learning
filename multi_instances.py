import multiprocessing as mp
import torch
import random
from game import SnakeGameAI
from model import Linear_QNet, QTrainer
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


def run_snake_instance(agent: Agent, game_idx, memory_queue):
    """
    Adding experiences to memory
    """
    game = SnakeGameAI()
    reload_interval = 5
    iteration = 0
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Send experience tuple (state, action, reward, next_state, done) to the memory queue
        memory_queue.put((state_old, final_move, reward, state_new, done))

        if done:
            game.reset()
            iteration += 1
            if iteration % reload_interval == 0:
                # Reload the model to get the updated weights
                best_model_path, best_score = get_best_model_path()
                if best_model_path:
                    agent.model.load(best_model_path)
                    print(
                        f"Instance {game_idx} reloaded the best model: {best_model_path}"
                    )
                    if score > best_score and not os.path.exists(f'./model/model_score_{score}.pth'):
                        agent.model.save(f'./model/model_score_{score}.pth')

            print("Process", game_idx, "Iteration", iteration, "Score", score, "Model_score_used", best_score)


def train_agent(agent: Agent, memory_queue):
    best_model_path, max_score = get_best_model_path()
    if best_model_path:
        agent.model.load(best_model_path)
        while True:
            if not memory_queue.empty():
                experience = memory_queue.get()
                agent.remember(*experience)  # Unpack the experience tuple
                agent.train_long_memory()  # Training on experiences from multiple games

def prueba(agent,i,other):
    import time
    print(f'iniciando {i}')
    time.sleep(3)
    print(f'finalizo {i}')

if __name__ == "__main__":

    agent = Agent()
    memory_queue = mp.Queue()

    # Create multiple processes for the Snake game instances
    num_games = 2
    processes = []

    try:
        for i in range(num_games):
            p = mp.Process(target=run_snake_instance, args=(agent, i, memory_queue))
            processes.append(p)
            p.start()

        # Start the training process
        #train_agent(agent, memory_queue)


        # Wait for all processes to complete
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        for p in processes:
            p.terminate()