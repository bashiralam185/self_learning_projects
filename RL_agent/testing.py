from model import TicTacToe, DQNAgent
import numpy as np
import random
import time
from tensorflow.keras.models import load_model
import keras

def test_agent(agent, games=10, visualize=False):
    env = TicTacToe()
    wins, losses, draws = 0, 0, 0
    
    for _ in range(games):
        state = env.reset()
        done = False
        if visualize:
            env.render()
            time.sleep(1)
        
        while not done:
            available_actions = env.get_available_actions()
            action = agent.act(state, available_actions)  # Pass available actions
            
            state, reward, done = env.step(action)
            
            # Opponent move (Random)
            if not done:
                opponent_action = random.choice(env.get_available_actions())
                state, _, done = env.step(opponent_action)
            
            if visualize:
                env.render()
                time.sleep(1)
        
        # Determine the result correctly
        if env.is_winner(1):
            wins += 1
        elif env.is_winner(-1):
            losses += 1
        else:
            draws += 1
    
    print(f"Results after {games} games: Wins: {wins}, Losses: {losses}, Draws: {draws}")

# Load the saved model
agent = DQNAgent(state_size=9, action_size=9)
agent.model = load_model("tic_tac_toe_model_final.h5", custom_objects={"MeanSquaredError": keras.losses.MeanSquaredError})

# Run the test
test_agent(agent, games=10, visualize=True)
