from model import TicTacToe, DQNAgent
import numpy as np

def train_agent(episodes=100, batch_size=16):
    env = TicTacToe()
    state_size = 9  # 3x3 board
    action_size = 9  # 9 possible moves
    agent = DQNAgent(state_size, action_size)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            available_actions = env.get_available_actions()
            if action >= len(available_actions):
                action = np.random.choice(range(len(available_actions)))
            
            action = available_actions[action]
            next_state, reward, done = env.step(action)
            agent.remember(state, action[0] * 3 + action[1], reward, next_state, done)
            state = next_state
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{episodes} - Epsilon: {agent.epsilon:.2f}")
    
    print("Training completed!")
    agent.model.save("tic_tac_toe_model.h5")  # Save the trained model
    return agent

trained_agent = train_agent()

