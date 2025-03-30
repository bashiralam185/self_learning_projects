import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from collections import deque

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the board to start a new game."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player 1 = X, Player -1 = O
        return self.board.flatten()

    def is_winner(self, player):
        """Check if a player has won the game."""
        for i in range(3):
            if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                return True
        if all(np.diag(self.board) == player) or all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def get_available_actions(self):
        """Return a list of available positions."""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def step(self, action):
        """Perform an action (place X or O)."""
        if self.board[action] != 0:
            return self.board.flatten(), -10, True  # Invalid move penalty
        
        self.board[action] = self.current_player
        
        # Check win condition
        if self.is_winner(self.current_player):
            return self.board.flatten(), 1, True  # Win reward
        
        # Check draw condition
        if not self.get_available_actions():
            return self.board.flatten(), 0, True  # Draw
        
        # Switch player and continue
        self.current_player *= -1
        return self.board.flatten(), 0, False

    def render(self):
        """Print the board state."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        """Build a neural network to approximate Q-values."""
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss=keras.losses.MeanSquaredError(),  # Use the explicit loss function
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experiences in memory for replay."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size):
        """Train the model using experience replay."""
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
