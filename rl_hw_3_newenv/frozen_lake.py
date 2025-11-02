import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
import time

def run():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode=None)

    q = np.zeros((env.observation_space.n, env.action_space.n))
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1
    epsilon_decay = 0.0001
    episodes = 10000
    rng = np.random.default_rng()
    
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        max_steps = 200
        steps = 0

        while not terminated and not truncated and steps < max_steps:

            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])
            new_state, reward, terminated, truncated, info = env.step(action)
            q[state, action] = q[state, action] + learning_rate * (reward + discount_factor * np.max(q[new_state, :]) - q[state, action])
            state = new_state
            steps += 1

        
        epsilon = max(epsilon - epsilon_decay, 0)
        if(epsilon == 0):
            learning_rate = 0.1
        if (i + 1) % 500 == 0:
            print(f"Episode {i+1}/{episodes} completed.")
    env.close()
    return q

def evaluate(q_table, episodes=1, delay=0.1):
    """
    用訓練好的 Q-table 在 FrozenLake 上進行測試。
    會顯示視窗並計算平均報酬。
    """
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human")

    total_reward = 0
    success_count = 0

    for ep in range(episodes):
        state, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0

        print(f"\n🎮 Episode {ep+1}/{episodes}")
        while not terminated and not truncated:
            # 根據 Q-table 選擇動作（不再隨機）
            action = np.argmax(q_table[state])
            new_state, reward, terminated, truncated, _ = env.step(action)

            state = new_state
            episode_reward += reward

            # 加一點延遲讓人類看得清楚
            time.sleep(delay)

        total_reward += episode_reward
        if episode_reward > 0:
            success_count += 1

    env.close()

    avg_reward = total_reward / episodes
    success_rate = success_count / episodes
    print(f"\n✅ average reward: {avg_reward:.2f}, success rate: {success_rate*100:.1f}%")

    return avg_reward, success_rate

if __name__ == "__main__":
    q = run()
    evaluate(q)