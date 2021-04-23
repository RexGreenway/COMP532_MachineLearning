"""
COMP532 - CA2
=============
By Thomas Rex Greenway, Kit Bower-Morris, Nicholas Bryan

Implementation of a Deep Q-Learning model on an OpenAI Gym environment interacting with a TensorFlow 
neural newtwork.

Dependencies
------------
Python 3.8.4rc1
TensorFlow 2.4.1
OpenAI gym 0.18.0
matplotlib 3.4.1
numpy 1.19.5
"""

# Suppress Annoying Tensorflow warnings.
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# print("\n")

# Import Numpy, OpenAI Gym + Tensorflow
import gym
import tensorflow as tf
import numpy as np


# Establish Neural Network HELPER class/ model function whatever.... thing
class Network(tf.keras.Model):
    """
    Tensorflow neural network helper class for use in deep q-network model.

    Parameters
    ----------
    state_shape : int
        Size of state data inputs into network.
    layers : list
        List of integers specifying the structure of the network's hidden layers.
    num_actions : int
        Number of possible actions to be taken, corresponding to network outputs.
    """
    def __init__(self, state_shape, layers, num_actions): 
        """
        Initialise a Neural Network with given state space, layers, and possible actions.
        """
        super(Network, self).__init__()

        # Input Layer
        self.input_layer = tf.keras.layers.InputLayer(input_shape = (state_shape,))

        # Hidden Layers
        self.hidden_layers = []
        for nodes in layers:
            hidden = tf.keras.layers.Dense(nodes, activation = "relu")  # ReLU activation for hidden layers
            self.hidden_layers.append(hidden)
        
        # Output Layer (Node for each possible action)
        self.output_layer = tf.keras.layers.Dense(num_actions)
    
    def call(self, inputs):
        """
        Forward pass through the network.
        """
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output


class DQ_Model():
    """
    Deep Q-Network Model for reinforcment learning on an OpenAi Gym environment.

    Parameters
    ----------
    state_shape : int
        Size of state data inputs into model.
    layers : list
        list of integers speciying the structure of the underlying network's hidden layers.
    num_actions : int
        Number of possible actions to be taken in the environment.
    min_buf_size : int
        Minimum size of the experience replay buffer.
    max_buf_size : int
        Maximum size of the experience replay buffer.
    batch_size : int
        Size of batches to be passed to the underlying network for training.
    gamma : float
        Discount factor.
    """
    def __init__(self, state_shape, layers, num_actions, min_buf_size, max_buf_size, batch_size, gamma):
        """
        Initialise Deep Q-Network Model.
        """
        self.num_actions = num_actions
        self.gamma = gamma

        # Optimizer
        self.optimizer = tf.optimizers.Adam(0.001)   # Learning rate defaults to 0.001

        # Establish network
        self.network = Network(state_shape, layers, num_actions)

        # Experience Replay dictionary for storing secific moents in the game to use for upadting weights
        # We randomly selct a bacth from this to update the model...
        self.experience = {"s" : [], "a" : [], "r" : [], "s'" : [], "done" : []}

        # Set max and min size for experience buffer
        self.min_buf_size = min_buf_size
        self.max_buf_size = max_buf_size 

        # Batch size for training
        self.batch_size = batch_size

        # Random generator for epsilon-greedy method
        self.rand_gen = np.random.default_rng()
    
    # Predict method to perfrom a foward pass over our Primary network and recieve Q-value estimates 
    def predict(self, inputs):
        """
        Peforms a forward pass on the primary model network to estimate action values.
        """
        # Tensorflow only accepts 
        return self.network(np.atleast_2d(inputs))
    
    def get_sample_batch(self):
        """
        Returns random sample batch for training, of a given size, from experience buffer.
        """
        # Gets random sample of indexes from experience buffer
        ids = self.rand_gen.integers(low=0, high=len(self.experience['s']), size=self.batch_size)
        
        # Retrieves the corresponding experience moments
        states = np.asarray([self.experience["s"][i] for i in ids])
        actions = np.asarray([self.experience["a"][i] for i in ids])
        rewards = np.asarray([self.experience["r"][i] for i in ids])
        next_states = np.asarray([self.experience["s'"][i] for i in ids])
        dones = np.asarray([self.experience["done"][i] for i in ids])
        return states, actions, rewards, next_states, dones

    def train(self, TargetNet):
        """
        Performs Q-Learning training on the primary network using a secondary network
        as the off-policy weight updates.

        Parameters
        ----------
        TargetNet : Network()
            Target Network (a near copy of the primary network, TrainNet) used to generate
            ground truth/ actual values.

        Returns
        -------
        loss : 
            Loss metric after batch forward pass.
        """
        # Only train once the experience buffer is above a certain size
        if len(self.experience["s"]) < self.min_buf_size:
            return 0

        # Recieve random sample batch
        states, actions, rewards, next_states, dones = self.get_sample_batch()

        # Use Target Network to generate y_actual for selected batch
        next_value = np.max(TargetNet.predict(next_states), axis = 1)
        y_actual = np.where(dones, rewards, rewards + self.gamma * next_value)

        with tf.GradientTape() as tape:
            # Foward Pass on primary network (i.e. TrainNet)
            y_pred = tf.math.reduce_sum(self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            # Calculate Loss (MSE)
            loss = tf.math.reduce_mean(tf.square(y_actual - y_pred))
        
        # Compute gradients
        variables = self.network.trainable_variables
        gradients = tape.gradient(loss, variables)
        # Update Weights (Backpropagation)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return loss

    def get_action(self, states, epsilon):
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        states : NumPy Array / Tensor
            State data taken from an environemnt used to generate actions.

        Returns
        -------
        action : 0 or 1 <-> 'left' or 'right'
            Action moving cart in direction left or right.
        """
        if self.rand_gen.random() < epsilon:
            return self.rand_gen.choice(self.num_actions)
        else:        
            return np.argmax(self.predict(states))

    def add_experience(self, exp):
        """
        Add experience (s, a, r, s', done) into experience buffer.
        """
        if len(self.experience['s']) >= self.max_buf_size:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)
    
    def copy_weights(self, net_to_copy):
        """
        Copy weights from one network model to another.
        """
        variables1 = self.network.trainable_variables
        variables2 = net_to_copy.network.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

def run_episode(env, TrainNet, TargetNet, epsilon, steps_to_copy, train = True, render = False):
    """
    Runs a single full episode of the CartPole Environment.

    Parameters
    ----------
    env : OpenAI Gym environment
        An environemt taken from OpenAI Gym.
    TrainNet : DQ_Model()
        Primary Model of Deep Q-Network to estimate state-action values.
    TargetNet : DQ_Model()
        Secondary network with identical architechure to the primary network model
        used to generate ground-truth/ actual state-action values.
    epsilon : float
        Probability of the model to select a non-greedy action.
    steps_to_copy : int
        Number of steps taken before primary network weights are copied to the
        secondary Target Network.
    train : bool (Default = True)
        True if performing a model training episode, Flase otherwise.
    render : bool (Default = False)
        True to render envirnoment visualiser window, Flase otherwise. 
    
    Returns
    -------
    reward_total : int
        The total reward for the episode.
    """
    # Set up initial variables 
    reward_total = 0
    i = 0
    done = False
    # Get initial state of environment
    state = env.reset()

    # Run through episode untill done bool returns True
    while not done:
        if render:
            env.render()

        # Store current state
        current_state = state
        # Select nect action form primary network
        action = TrainNet.get_action(state, epsilon)
        # Take that action in the encironemt and recieve the next state, reward, and done bool
        state, reward, done, _ = env.step(action)
        # Add reward to reward total
        reward_total += reward
        
        # Done check
        if done:            
            reward = -200
            env.reset()

        # Add experience moment to experience buffer
        exp = {"s": current_state, "a": action, "r": reward, "s'": state, "done": done}
        TrainNet.add_experience(exp)

        # Train primary network using secondary network as off-policy
        if train:
            TrainNet.train(TargetNet)

        # Copy weights to target network from primary network at desired steps
        i += 1
        if i % steps_to_copy == 0:
            TargetNet.copy_weights(TrainNet)

    return reward_total

def movingAvg(sumList):
    """
    Returns the array of moving averages for a given array.
    """
    movingAvg = []
    for itemIndex in range(len(sumList)):
        if itemIndex < 5:
            movingAvg.append(-100)
        elif itemIndex > len(sumList) - 5:
            movingAvg.append(None)
        else:
            mean = sum(sumList[itemIndex - 5 : itemIndex + 5])/10
            movingAvg.append(mean)
    
    return movingAvg

def main():
    # Hyper-Parameters
    env = gym.make("CartPole-v0")
    gamma = 0.99
    copy_step = 25
    num_states = env.observation_space.shape[0]     # Box --> 4
    num_actions = env.action_space.n                # Discrete -- > 2
    hidden_layers = [32, 32]
    max_buf_size = 10000
    min_buf_size = 100
    batch_size = 32

    epsilon = 0.9

    # Establish Target Network
    TargetNet = DQ_Model(num_states, hidden_layers, num_actions, min_buf_size, max_buf_size, batch_size, gamma)
    # Establish Model
    TrainNet = DQ_Model(num_states, hidden_layers, num_actions, min_buf_size, max_buf_size, batch_size, gamma)

    episodes = 1000
    rewards = np.zeros(episodes)
    
    for ep in range(episodes):
        epsilon = max(0.995 * epsilon, 0.1)
        final_reward = run_episode(env, TrainNet, TargetNet, epsilon, copy_step)
        rewards[ep] = final_reward

        avg_rewards = rewards[max(0, ep - 100):(ep + 1)].mean()

        if ep % 100 == 0:
            print(
                "epidode: ", ep,
                "ep reward: ", final_reward,
                "epsilon: ", epsilon,
                "avg reward (prev 100): ", avg_rewards
            )
    print("avg reward for last 100 episodes:", avg_rewards)

    # Plot the rewards
    import matplotlib.pyplot as plt

    plt.plot(movingAvg(rewards))
    plt.ylabel("Reward")
    plt.xlabel("Episodes")
    plt.ylim(0, 210)
    plt.show()

    # Run on trained network
    for _ in range(50):
        run_episode(env, TrainNet, TargetNet, epsilon, copy_step, train = False, render = True)
    
    env.close()

if __name__ == "__main__":
    main()
