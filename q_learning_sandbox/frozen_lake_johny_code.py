import gymnasium as  gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(episodes=1000,render=False):
    
        env = gym.make("FrozenLake-v1", render_mode="human" if render else None, map_name="4x4", is_slippery=False)
        
        q = np.zeros((env.observation_space.n, env.action_space.n))
        
        learning_rate = 0.1
        discount_factor = 0.9
        
        epsilon = 1
        epsilon_decay_rate = 0.001 # 1/episodes
        rng = np.random.default_rng()
        
        reward_per_episode = np.zeros(episodes)
        
        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False
        

            while not (terminated or truncated):
                if rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q[state,:])
                    
                new_state, reward, terminated, truncated, _ = env.step(action)
                
                # Q[estado_actual_idx, accion] = Q[estado_actual_idx, accion] + alpha * (recompensa + gamma * mejor_Q_siguiente - Q[estado_actual_idx, accion])

                q[state, action] = q[state, action]  + learning_rate * (
                    reward + discount_factor * np.max(q[new_state])- q[state, action]
                )
                
                
                state = new_state
                
            epsilon = max(epsilon - epsilon_decay_rate,0)

            if (epsilon ==0):
                learning_rate = 0.01 # reduce learning rate to avoid oscillations
                
            if reward == 1:
                reward_per_episode[i] = 1
            
        env.close()
        
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_rewards[i] = np.sum(reward_per_episode[max(0,t-100):t+1])
        plt.plot(sum_rewards)
        plt.savefig("frozen_lake_rewards.png")
        
        f = open("frozen_lake_q_table.txt", "wb")
        pickle.dump(q, f)
        f.close()
        
if __name__ == "__main__":
    run()
    
