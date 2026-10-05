import gymnasium as gym 

env=gym.make('CartPole-v1',render_mode="human")

observation,info=env.reset()

print("Starting observation:{observation}")

episode_over=False
total_reward=0

while not episode_over:
    action=env.action_space.sample()
    observation,reward,terminated,truncated,info=env.step(action)
    total_reward+=reward
    if terminated or truncated:
        episode_over=True

print(f"Total reward: {total_reward}")
env.close()