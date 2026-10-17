
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agent import QLearningAgent

def main():
    # 1. Initialize Environment
    env = gym.make("Taxi-v4")
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # 2. Hyperparameters
    n_episodes = 3000
    max_steps = 100
    
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )
    
    rewards_history = []
    
    print("Training started...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            agent.update(state, action, reward, next_state, terminated)
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
                
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{n_episodes} - Average Reward (last 100): {np.mean(rewards_history[-100:]):.2f}")
            
    print("Training finished!")
    
    # Save Q-table weights
    np.save("taxi_q_table.npy", agent.q_table)
    print("Model weights saved as 'taxi_q_table.npy'.")
    
    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title("Training Rewards over Episodes (Taxi-v3)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("training_rewards.png")
    print("Training plot saved as 'training_rewards.png'.")
    
    env.close()

if __name__ == "__main__":
    main()