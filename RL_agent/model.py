import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from collections import deque

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the board for a new game."""
        self.board = np.zeros((3, 3), dtype=float)
        self.current_player = 1  # Player 1 = X, Player -1 = O
        self.done = False
        return self.board.flatten() # Return a copy to prevent external modification

    def is_winner(self, player):
        """Check if a player has won the game."""
        # Check rows and columns
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        # Check diagonals
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def get_available_actions(self):
        """Return a list of available positions as (row, col) tuples."""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def step(self, action):
        """Perform an action (place X or O)."""
        if self.done:
            raise ValueError("Game has already ended")
            
        if self.board[action] != 0:
            return self.board.copy(), -10, True  # Invalid move penalty
        
        self.board[action] = self.current_player
        
        # Check win condition
        if self.is_winner(self.current_player):
            self.done = True
            return self.board.copy(), 10, True  # Increased win reward
        
        # Check draw condition
        if not self.get_available_actions():
            self.done = True
            return self.board.copy(), 1, True  # Draw reward
        
        # Switch player and continue
        self.current_player *= -1
        return self.board.copy(), -0.1, False  # Small penalty for not winning immediately

    def render(self):
        """Print the board state."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in self.board:
            print(' '.join(symbols[int(cell)] for cell in row))
        print()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Build a neural network to approximate Q-values."""
        model = keras.Sequential([
            keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update the target model weights with the main model weights."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experiences in memory for replay."""
        # Ensure states are flattened before storing
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))
    
    def act(self, state, available_actions):
        """Choose an action using an epsilon-greedy policy with valid moves."""
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)
        
        # Ensure state is flattened before prediction
        flat_state = state.flatten()
        q_values = self.model.predict(flat_state.reshape(1, -1), verbose=0)[0]
        
        # Convert available actions to indices
        action_indices = [a[0] * 3 + a[1] for a in available_actions]
        best_action_index = np.argmax([q_values[i] for i in action_indices])
        return available_actions[best_action_index]
    
    def replay(self, batch_size):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            # States are already flattened when stored in memory
            states[i] = state
            next_states[i] = next_state
            
            # Convert action to index
            action_idx = action[0] * 3 + action[1]
            
            # Predict Q-values for current state
            target = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            
            if done:
                target[action_idx] = reward
            else:
                q_future = np.amax(self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0])
                target[action_idx] = reward + self.gamma * q_future
            
            targets[i] = target

        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
