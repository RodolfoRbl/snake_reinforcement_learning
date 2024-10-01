import pygame
import random
import numpy as np
from collections import deque
import keras

from .snake_game_human import SnakeGame, Direction

# Initialize pygame
pygame.init()
font = pygame.font.SysFont("arial", 25)


# DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay memory
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        # Build the neural network model
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experiences in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Choose an action based on the exploration-exploitation tradeoff
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        act_values = self.model.predict(state)  # Exploit
        return np.argmax(act_values[0])  # Return the action with max value

    def replay(self, batch_size):
        # Train the model using a batch of experiences
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    @classmethod
    def simulate_game(cls, max_steps=500):
        game = SnakeGame(speed=20, parallel_food=1, wall_collision=True, is_auto=False)  # Game setup
        state = game.get_state()  # Get initial state
        steps = 0
        total_reward = 0
        game_over = False
        
        while not game_over and steps < max_steps:
            action = agent.get_action(state)  # Agent selects action based on current state

            # Map the action to the game direction
            if action == 0:
                game.direction = Direction.LEFT
            elif action == 1:
                game.direction = Direction.RIGHT
            elif action == 2:
                game.direction = Direction.UP
            elif action == 3:
                game.direction = Direction.DOWN

            # Perform a game step based on the action and get the new state
            game_over, score = game.play_step()

            # Get the next state after the action
            next_state = game.get_state()

            # Update the current state to the new state for the next iteration
            state = next_state
            steps += 1
            total_reward += score

        print(f"Game over. Total steps: {steps}, Final score: {score}")
        return score


if __name__ == "__main__":
    # Initialize game and DQN Agent
    game = SnakeGame(speed=20, parallel_food=1, wall_collision=True, is_auto=False)
    agent = DQNAgent(state_size=8, action_size=4)  # Update according to your game action space
    batch_size = 32

    for e in range(10):  # Number of episodes
        state = game.get_state()
        state = np.reshape(state, [1, agent.state_size])

        for time in range(500):  # Maximum time steps per episode
            action = agent.act(state)
            done, score = game.play_step()  # Update this function to return new state and reward
            reward = score  # You may want to adjust the reward based on the game logic
            next_state = game.get_state()
            next_state = np.reshape(next_state, [1, agent.state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode: {e + 1}/{1000}, Score: {score}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    pygame.quit()
