import numpy as np
import json
import os
import random
from model import TicTacToe, DQNAgent
from tensorflow.keras.models import load_model
import time

class EnhancedDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.training_history = {
            'episodes': [],
            'epsilon': [],
            'rewards': [],
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate': [],
            'avg_reward': []
        }
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience and track rewards"""
        # States will be flattened by parent class
        super().remember(state, action, reward, next_state, done)
        self.training_history['rewards'].append(reward)
        
    def record_episode_result(self, result):
        """Track win/loss/draw stats"""
        if result == 1:
            self.training_history['wins'] += 1
        elif result == -1:
            self.training_history['losses'] += 1
        else:
            self.training_history['draws'] += 1

def train_agent(episodes=500, batch_size=64, save_interval=50, target_update_interval=25):
    env = TicTacToe()
    state_size = 9  # Flattened board size
    action_size = 9  # 3x3 board positions
    agent = EnhancedDQNAgent(state_size, action_size)
    
    os.makedirs('training_data', exist_ok=True)
    warmup_episodes = 100
    
    for episode in range(episodes):
        state = env.reset()  # Already returns flattened state
        done = False
        total_reward = 0
        episode_start_time = time.time()
        
        while not done:
            available_actions = env.get_available_actions()
            action = agent.act(state, available_actions)
            
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state.flatten()  # Ensure state is flattened
            total_reward += reward
            
            # Opponent move (Random)
            if not done:
                opponent_action = random.choice(env.get_available_actions())
                next_state, _, done = env.step(opponent_action)
                state = next_state.flatten()  # Ensure state is flattened
        
        # Record episode results
        if env.is_winner(1):
            agent.record_episode_result(1)
        elif env.is_winner(-1):
            agent.record_episode_result(-1)
        else:
            agent.record_episode_result(0)
        
        # Train if enough experiences
        if len(agent.memory) > batch_size and episode >= warmup_episodes:
            agent.replay(batch_size)
        
        # Update target network periodically
        if episode % target_update_interval == 0 and episode > warmup_episodes:
            agent.update_target_model()
        
        # Calculate metrics
        win_rate = agent.training_history['wins'] / (episode + 1)
        avg_reward = np.mean(agent.training_history['rewards'][-50:]) if len(agent.training_history['rewards']) > 0 else 0
        
        # Update training history
        agent.training_history['episodes'].append(episode)
        agent.training_history['epsilon'].append(agent.epsilon)
        agent.training_history['win_rate'].append(win_rate)
        agent.training_history['avg_reward'].append(avg_reward)
        
        # Save checkpoint
        if episode % save_interval == 0 or episode == episodes - 1:
            agent.model.save(f"training_data/model_ep{episode}.h5")
            with open(f"training_data/stats_ep{episode}.json", 'w') as f:
                json.dump(agent.training_history, f)
            
            episode_time = time.time() - episode_start_time
            print(f"Episode {episode}/{episodes} - "
                  f"Time: {episode_time:.2f}s - "
                  f"Epsilon: {agent.epsilon:.3f} - "
                  f"Win Rate: {win_rate:.2f} - "
                  f"Avg Reward: {avg_reward:.2f} - "
                  f"W/L/D: {agent.training_history['wins']}/{agent.training_history['losses']}/{agent.training_history['draws']} - "
                  f"Memory: {len(agent.memory)}")
    
    # Save final models
    agent.model.save("tic_tac_toe_model_final.h5")
    agent.target_model.save("tic_tac_toe_target_model_final.h5")
    with open("training_data/final_stats.json", 'w') as f:
        json.dump(agent.training_history, f)
    
    print("\nTraining completed!")
    print(f"Final Stats - Wins: {agent.training_history['wins']} | "
          f"Losses: {agent.training_history['losses']} | "
          f"Draws: {agent.training_history['draws']}")
    print(f"Final Win Rate: {agent.training_history['win_rate'][-1]:.2f}")
    
    return agent

if __name__ == "__main__":
    trained_agent = train_agent(
        episodes=500,
        batch_size=64,
        save_interval=50,
        target_update_interval=25
    )