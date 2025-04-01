import numpy as np
import json
import os
from collections import deque
from model import TicTacToe, DQNAgent
from tensorflow.keras.models import load_model

class EnhancedDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.training_history = {
            'episodes': [],
            'epsilon': [],
            'rewards': [],
            'wins': 0,
            'losses': 0,
            'draws': 0
        }
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience and track rewards"""
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

def train_agent(episodes=500, batch_size=32, save_interval=50):
    env = TicTacToe()
    state_size = 9
    action_size = 9
    agent = EnhancedDQNAgent(state_size, action_size)
    
    # Create directory for saving models and stats
    os.makedirs('training_data', exist_ok=True)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            available_actions = env.get_available_actions()
            available_indices = [a[0]*3 + a[1] for a in available_actions]
            
            # Convert action to valid move
            if action >= len(available_indices):
                action = np.random.choice(range(len(available_indices)))
            action_pos = available_actions[action]
            
            next_state, reward, done = env.step(action_pos)
            agent.remember(state, available_indices[action], reward, next_state, done)
            state = next_state
            total_reward += reward
        
        # Record episode results
        if env.is_winner(1):  # Agent won
            agent.record_episode_result(1)
        elif env.is_winner(-1):  # Opponent won
            agent.record_episode_result(-1)
        else:  # Draw
            agent.record_episode_result(0)
        
        # Train if enough experiences
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        # Update training history
        agent.training_history['episodes'].append(episode)
        agent.training_history['epsilon'].append(agent.epsilon)
        
        # Save checkpoint periodically
        if episode % save_interval == 0 or episode == episodes - 1:
            # Save model
            model_path = f'training_data/model_ep{episode}.h5'
            agent.model.save(model_path)
            
            # Save training stats
            stats_path = f'training_data/stats_ep{episode}.json'
            with open(stats_path, 'w') as f:
                json.dump(agent.training_history, f)
            
            # Print progress
            win_rate = agent.training_history['wins'] / (episode + 1)
            print(f"Episode {episode}/{episodes} - "
                  f"Epsilon: {agent.epsilon:.3f} - "
                  f"Win Rate: {win_rate:.2f} - "
                  f"Total Reward: {total_reward:.1f} - "
                  f"Memory: {len(agent.memory)}")
    
    # Save final model and stats
    agent.model.save("tic_tac_toe_model_final.h5")
    with open("training_data/final_stats.json", 'w') as f:
        json.dump(agent.training_history, f)
    
    print("\nTraining completed!")
    print(f"Final Stats - Wins: {agent.training_history['wins']} | "
          f"Losses: {agent.training_history['losses']} | "
          f"Draws: {agent.training_history['draws']}")
    
    return agent

# Enhanced training with visualization
if __name__ == "__main__":
    trained_agent = train_agent(episodes=500, batch_size=64)
    
    # # Generate training report
    # import matplotlib.pyplot as plt
    
    # # Load final stats
    # with open("training_data/final_stats.json") as f:
    #     stats = json.load(f)
    
    # # Plot training progress
    # plt.figure(figsize=(12, 4))
    
    # # Epsilon decay
    # plt.subplot(1, 2, 1)
    # plt.plot(stats['episodes'], stats['epsilon'])
    # plt.title('Epsilon Decay')
    # plt.xlabel('Episode')
    # plt.ylabel('Epsilon')
    
    # # Win rate
    # win_rates = [stats['wins']/(i+1) for i in range(len(stats['episodes']))]
    # plt.subplot(1, 2, 2)
    # plt.plot(stats['episodes'], win_rates)
    # plt.title('Win Rate Progress')
    # plt.xlabel('Episode')
    # plt.ylabel('Win Rate')
    
    # plt.tight_layout()
    # plt.savefig('training_data/training_progress.png')
    # plt.show()