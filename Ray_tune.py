import multiprocessing

import tensorflow as tf

import ray
from ray import tune

n_cpus = multiprocessing.cpu_count()
n_cpus

n_gpus = len(tf.config.list_physical_devices('GPU'))
n_gpus

# initialize ray
# https://ray.readthedocs.io/en/latest/package-ref.html#ray.init
ray.init(ignore_reinit_error=True, log_to_driver=False, webui_host='0.0.0.0')

# run one training iteration
# https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

env_name = 'CartPole-v1'

ppo_config = DEFAULT_CONFIG.copy()
if n_gpus:
    ppo_config['num_gpus'] = n_gpus
    ppo_config['tf_session_args']['device_count']['GPU'] = n_gpus

ppo_config['num_workers'] = 1
ppo_config['num_sgd_iter'] = 2
ppo_config['sgd_minibatch_size'] = 128
ppo_config['lr'] = 0.0003
ppo_config['gamma'] = 0.99
ppo_config['model']['fcnet_hiddens'] = [64, 64]
ppo_config['timesteps_per_iteration'] = 2000
ppo_config['train_batch_size'] = 8000
ppo_config['num_cpus_per_worker'] = 0  # This avoids running out of resources in the notebook environment when this cell is re-executed

agent = PPOTrainer(ppo_config, env_name)
result = agent.train()

result

# tune hyperparamters with grid search
# https://github.com/ray-project/ray/blob/master/python/ray/tune/tune.py
ray.init(ignore_reinit_error=True)
env_name = 'CartPole-v1'
ppo_config = {
    "env": env_name,
    "num_workers": 1,
    'model': {
        'fcnet_hiddens': tune.grid_search([
                                           [16, 16], [32, 32], [64, 64], [128, 128],
                                          ])
    },        
    'train_batch_size': 1000,
    "lr": tune.grid_search([0.0003, 0.0001]),
    'gamma': tune.grid_search([0.99, 0.999]),
    "eager": False,
    'num_gpus': n_gpus  
}
                      
analysis = tune.run(
    "PPO",
    name='cartpole_test',
    verbose=1,

    stop={"episode_reward_mean": 300},  # stop when a parameter set is able to reach 300 timesteps
    config = ppo_config,
    checkpoint_freq=10,
    checkpoint_at_end=True,
    checkpoint_score_attr='episode_reward_mean',
    num_samples=1,  # for grid search, number of times to run each hyperparameter combo
    #     with_server=True,
    #     server_port=8267,
)

