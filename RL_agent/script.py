import tkinter as tk
from tkinter import messagebox
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import h5py
import os

class TicTacToe:
    """Tic-Tac-Toe game environment"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.board.flatten()

    def is_winner(self, player):
        for i in range(3):
            if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                return True
        if all(np.diag(self.board) == player) or all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def get_available_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def step(self, action):
        if self.board[action] != 0:
            return self.board.flatten(), -10, True
        
        self.board[action] = self.current_player
        
        if self.is_winner(self.current_player):
            return self.board.flatten(), 1, True
        
        if not self.get_available_actions():
            return self.board.flatten(), 0, True
        
        self.current_player *= -1
        return self.board.flatten(), 0, False

class DQNAgent:
    """Deep Q-Network agent"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0  # Exploration rate (0 during testing)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        available_actions = [(i, j) for i in range(3) for j in range(3) if state[i*3+j] == 0]
        if not available_actions:
            return None
            
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        available_indices = [a[0]*3 + a[1] for a in available_actions]
        action_idx = available_indices[np.argmax(q_values[available_indices])]
        return (action_idx // 3, action_idx % 3)

class TicTacToeGUI:
    """GUI for playing against the AI"""
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe vs AI")
        self.game = TicTacToe()
        self.agent = self.load_agent()
        self.buttons = []
        self.create_board()
        self.reset_game()

    def load_agent(self):
        """Robust model loading with multiple fallback strategies"""
        agent = DQNAgent(9, 9)
        
        # Strategy 1: Try loading the complete model
        try:
            from tensorflow.keras.models import load_model
            agent.model = load_model("tic_tac_toe_model.h5", compile=False)
            print("Model loaded successfully with standard approach")
            return agent
        except Exception as e:
            print(f"Standard load failed: {str(e)}")

        # Strategy 2: Try loading architecture + weights separately
        try:
            agent.model.load_weights("tic_tac_toe_model.h5")
            print("Model loaded successfully with weights-only approach")
            return agent
        except Exception as e:
            print(f"Weights-only load failed: {str(e)}")

        # Strategy 3: Try converting the model
        try:
            with h5py.File("/home/uca/Documents/self_learning_projects/RL_agent/tic_tac_toe_model_final.h5", 'r') as f:
                model_config = f.attrs.get('model_config')
                if model_config:
                    agent.model = model_from_json(model_config.decode('utf-8'))
                    agent.model.set_weights([np.array(f[weight]) for weight in f['model_weights']])
                    print("Model loaded successfully with JSON config approach")
                    return agent
        except Exception as e:
            print(f"JSON config load failed: {str(e)}")

        # Fallback: Create new untrained model
        messagebox.showwarning("Model Load", "Using new untrained model")
        return agent

    def create_board(self):
        """Create the game board UI"""
        for i in range(3):
            row = []
            for j in range(3):
                btn = tk.Button(
                    self.root, text="", font=('Arial', 30),
                    height=2, width=5, bg='lightgray',
                    command=lambda i=i, j=j: self.on_click(i, j)
                )
                btn.grid(row=i, column=j, sticky="nsew")
                row.append(btn)
            self.buttons.append(row)
        
        # Status label
        self.status = tk.Label(self.root, text="", font=('Arial', 14))
        self.status.grid(row=3, column=0, columnspan=3)

        # Reset button
        tk.Button(
            self.root, text="New Game", font=('Arial', 14),
            command=self.reset_game
        ).grid(row=4, column=0, columnspan=3, sticky="nsew")

    def reset_game(self):
        """Reset the game state"""
        self.game.reset()
        self.update_board()
        self.status.config(text="Your turn (X)")

    def update_board(self):
        """Update the UI to match game state"""
        for i in range(3):
            for j in range(3):
                cell = self.game.board[i][j]
                btn = self.buttons[i][j]
                btn.config(state='normal')
                if cell == 1:
                    btn.config(text="X", fg='red', state='disabled')
                elif cell == -1:
                    btn.config(text="O", fg='blue', state='disabled')
                else:
                    btn.config(text="", state='normal')

    def on_click(self, row, col):
        """Handle human player move"""
        if self.game.board[row][col] != 0:
            return

        # Human move
        _, _, done = self.game.step((row, col))
        self.update_board()

        if done:
            self.game_over()
            return

        # AI move after short delay
        self.root.after(500, self.ai_move)

    def ai_move(self):
        """Handle AI move"""
        state = self.game.board.flatten()
        action = self.agent.act(state)
        
        if action:
            _, _, done = self.game.step(action)
            self.update_board()
            
            if done:
                self.game_over()

    def game_over(self):
        """Handle game end conditions"""
        if self.game.is_winner(1):
            self.status.config(text="You won!")
        elif self.game.is_winner(-1):
            self.status.config(text="AI won!")
        else:
            self.status.config(text="It's a draw!")
        
        # Disable all buttons
        for row in self.buttons:
            for btn in row:
                btn.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToeGUI(root)
    
    # Configure grid weights for resizing
    for i in range(3):
        root.grid_rowconfigure(i, weight=1)
        root.grid_columnconfigure(i, weight=1)
    
    root.mainloop()