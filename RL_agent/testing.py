from model import TicTacToe, DQNAgent
import numpy as np
from tensorflow.keras.models import load_model
import time
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
            action = agent.act(state)
            available_actions = env.get_available_actions()
            if action >= len(available_actions):
                action = np.random.choice(range(len(available_actions)))
            
            action = available_actions[action]
            state, reward, done = env.step(action)
            
            if visualize:
                env.render()
                time.sleep(1)
        
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1
    
    print(f"Results after {games} games: Wins: {wins}, Losses: {losses}, Draws: {draws}")

# Load the saved model
agent = DQNAgent(state_size=9, action_size=9)
agent.model = load_model("tic_tac_toe_model.h5", custom_objects={"MeanSquaredError": keras.losses.MeanSquaredError})

test_agent(agent, visualize=True)
