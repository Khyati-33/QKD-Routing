from rl_agent.evaluate import compare_policies

results = compare_policies(
    agent=agent,
    config_path='experiments/configs/network_3node.yaml',
    policies=['shortest', 'max_skr', 'pool_aware', 'random', 'rl', 'dp_optimal'],
    n_seeds=30,
    T_hours=24
)
# Returns DataFrame: policy × metric (mean ± std)
