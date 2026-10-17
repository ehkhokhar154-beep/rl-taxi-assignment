import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import os

def record_trained_agent(q_table_path="taxi_q_table.npy", video_folder="recorded_video"):
    # 1. Environment initialize karein with rgb_array render mode
    env = gym.make("Taxi-v4", render_mode="rgb_array")
    
    # 2. RecordVideo wrapper lagayein taaki video automatically save ho jaye
    env = RecordVideo(
        env, 
        video_folder=video_folder, 
        episode_trigger=lambda episode_id: True, # Har episode record karega
        name_prefix="taxi-agent"
    )
    
    # 3. Saved Q-table load karein
    if not os.path.exists(q_table_path):
        print(f"Error: {q_table_path} nahi mila! Pehle main.py run karke training karein.")
        return

    q_table = np.load(q_table_path)
    
    print("Recording trained agent...")
    # 4. Agent ko test karein aur video record karein
    for episode in range(3): # 3 test episodes record honge
        state, _ = env.reset()
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            # Greedy action select karein (trained policy ke mutabiq)
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            
    env.close()
    print(f"Video successfully saved in folder: '{video_folder}'!")

if __name__ == "__main__":
    record_trained_agent()