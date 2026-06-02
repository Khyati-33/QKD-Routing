from rl_agent.train import train

agent, logs = train(
    config_path='experiments/configs/network_3node.yaml',
    n_episodes=500,
    checkpoint_dir='checkpoints/'
)
