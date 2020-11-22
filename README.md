Notebooks and code for [Alpha Architect](https://alphaarchitect.com/2020/02/26/reinforcement-learning-for-trading/) post on reinforcement learning.

- [Tic-Tac-Toe.ipynb](Tic-Tac-Toe.ipynb) - Table-based reinforcement learning to play Tic-Tac-Toe, and analogous if pointless deep learning algo
- [Cart-Pole.ipynb](Cart-Pole.ipynb) - Building deep reinforcement learning algos from scratch with Keras for OpenAI environments like Cartpole and LunarLander. 
  - DQN
  - Policy Gradient (REINFORCE)
  - REINFORCE with baseline
  - [Run_CartPole.ipynb](Run_CartPole.ipynb), [Run_LunarLander.ipynb](Run_LunarLander.ipynb) - only run saved good models, don't train
- [Ray_tune.ipynb](Ray_tune.ipynb) - Similar but with state of the art RL from UC Berkeley Ray project
- [Trading_with_RL.ipynb](Trading_with_RL.ipynb) - Algos to trade fake market data, inspired by Gordon Ritter paper Machine Learning for Trading. This *should* run in [Google Colab](https://colab.research.google.com/github/druce/rl/blob/master/Trading_with_RL.ipynb).

Typical installation procedure:

- Install [Anaconda](https://docs.anaconda.com/anaconda/install/) python data science distribution 

- Make an environment like 

  ```bash
  conda create --name tf tensorflow
  ```

  or if you have Nvidia GPU

  ```bash
  conda create --name tf_gpu tensorflow-gpu 
  ```

  This should install requirements like working Nvidia drivers

- Upgrade [TensorFlow](https://www.tensorflow.org/install/pip?lang=python3) to latest version with 

  ```bash
  pip install --upgrade tensorflow
  ```

- Install additional requirements as necessary - [requirements.txt](requirements.txt) has python modules installed at time of testing.
  ```bash
  pip install -r requirements.txt
  ```

- TensorFlow Docker install may also be a good way to start but has not been tested.

- Run notebooks using 

  ```bash
  jupyter notebook
  ```

  
