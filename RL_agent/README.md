## Tic-Tac-Toe Deep Q-Network (DQN) AI
This project consists of three Python scripts that together implement a reinforcement learning (RL) agent playing Tic-Tac-Toe. The agent uses Deep Q-Learning (DQN) to train and optimize its strategy against a random opponent. Additionally, a graphical user interface (GUI) allows a human player to play against the trained AI.

#### Files:
- [model.py](model.py): Contains the definition of the Tic-Tac-Toe environment and the DQN agent model.
- [training.py](training.py): Handles the training process, including agent training, saving checkpoints, and recording the training history.
- [script.py](script.py): A GUI implementation where a user can play Tic-Tac-Toe against the AI using the trained DQN model.
- [model_analysis.ipynb](model_analysis.ipynb): Provide usefull visualizations to understand model training

#### Overview of Scripts
##### 1. model.py
This script defines the core components of the Tic-Tac-Toe environment and the agent that plays the game.

***TicTacToe Class:***
The environment that simulates a Tic-Tac-Toe game.
- reset: Resets the board to its initial state and returns the flattened game state.
- is_winner: Checks if a given player has won.
- get_available_actions: Returns a list of all available actions (empty cells).
- step: Takes an action (a player places their mark) and returns the new board, reward, and whether the game is over.
- render: Prints the current state of the game in the console.

***DQNAgent Class***:
Implements the Deep Q-Network agent that interacts with the game environment.

- _build_model: Builds a neural network model to approximate Q-values for each state-action pair.
- remember: Stores experiences (state, action, reward, next_state, done) for experience replay.
- act: Chooses an action based on the epsilon-greedy policy.
- replay: Trains the model using a batch of stored experiences.
- update_target_model: Periodically updates the target model with the weights of the main model.

***2. training.py***
This script is responsible for training the DQN agent. It also records the training progress, including win rates and rewards, and saves checkpoints.

- EnhancedDQNAgent Class:
Inherits from DQNAgent and extends it with the ability to track training statistics such as win/loss/draw counts, win rate, and average reward.

- train_agent function:
Trains the agent over a specified number of episodes.

- It performs training updates, opponent moves, and evaluates the agent's performance.
- After each episode, it updates the training history, logs results, and saves checkpoints.
- Training data is saved at regular intervals, including:
- The model's weights (model_ep{episode}.h5)
- Training statistics (stats_ep{episode}.json)
- At the end of the training, the final models and statistics are saved.

***3. script.py***
This script implements a graphical user interface (GUI) using Tkinter where a human player can play against the trained AI.

- TicTacToe Class:
Similar to the class in model.py but with a GUI component to track and display the game state.

The game state is updated after every move, and the board is visually represented in the GUI.

- DQNAgent Class:
A trained model is loaded to allow the AI to make moves against the player.

- act: Predicts the best move based on the current board state.

- TicTacToeGUI Class:
The graphical user interface that allows the user to play against the AI.

It provides buttons for each cell in the Tic-Tac-Toe grid, updates the board based on player and AI moves, and shows the game result (win/loss/draw).
