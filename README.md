Notebooks and code for Alpha Architect post on reinforcement learning.

- [Tic-Tac-Toe.ipynb](Tic-Tac-Toe.ipynb) - table-based reinforcement learning to play Tic-Tac-Toe, and analogous if pointless deep learning algo
- [Cart-Pole.ipynb](Cart-Pole.ipynb) - building deep reinforcement learning algos from scratch with Keras for OpenAI environments like Cartpole and LunarLander. 
  - DQN
  - Policy Gradient (REINFORCE)
  - REINFORCE with baseline
  - [Run CartPole.ipynb](Run CartPole.ipynb), [Run LunarLander.ipynb](Run LunarLander.ipynb) - only run saved good models, don't train
- [Ray_tune.ipynb](Ray_tune.ipynb) - similar but with state of the art RL from UC Berkeley Ray project
- [Trading with RL.ipynb](Trading with RL.ipynb) - algos to trade fake market data, inspired by Gordon Ritter paper Machine Learning for Trading

Typical installation procedure:

- Install [Anaconda](https://docs.anaconda.com/anaconda/install/) python data science distribution 

- Make an environment like 

  ```python
  conda create --name tf tensorflow
  ```

  or if you have Nvidia GPU

  ```python
  conda create --name tf_gpu tensorflow-gpu 
  ```

  This should install requirements like working Nvidia drivers

- Upgrade [TensorFlow](https://www.tensorflow.org/install/pip?lang=python3) to latest version with 

  ```python
  pip install --upgrade tensorflow
  ```

- Install additional requirements as necessary - requirements.txt has python modules installed at time of testing, but most of these aren't necessary and more recent versions may be preferable.

- TensorFlow Docker install may also be a good way to start but has not been tested.

- run notebooks using 

  ```python
  jupyter notebook
  ```

  