"""
ASSIGNMENT 2
------------
By Thomas Rex Greenway, Kit Bower-Morris, Nicholas Bryan

Implementation of a Deep Q-Learning model on an OpenAI Gym environment interacting with a TensorFlow 
neural newtwork.  
"""

# Suppress Annoying Tensorflow warnings.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("\n")

# Import Numpy, OpenAI Gym + Tensorflow
import gym
import tensorflow as tf
import numpy as np


# Establish Neural Network HELPER class/ model function whatever.... thing
class Network(tf.keras.Model):
    """
    Tensorflow neural network class...
    """
    def __init__(self, state_shape, layers, num_actions):  # inputs are gonna be the states 
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
        Call function (i.e. foward pass...?)
        """
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output


class DQ_Model():
    """
    the agent...? or model or what??
    """
    def __init__(self, state_shape, layers, num_actions, min_buf_size, max_buf_size, batch_size, gamma):
        """
        """
        self.num_actions = num_actions
        self.gamma = gamma

        # Optimizer
        self.optimizer = tf.optimizers.Adam()   # Learning rate defaults to 0.001

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
        """
        # print(np.atleast_2d(inputs.astype('float32')))
        return self.network(np.atleast_2d(inputs))
    
    # def get_sample_batch(self):
    #     """
    #     Returns sample batch for training from experience buffer
    #     """
    #     # Gets random sample of indexes from experience buffer
    #     ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        
    #     # Retrieves the corresponding experience moments
    #     states = np.asarray([self.experience["s"][i] for i in ids])
    #     actions = np.asarray([self.experience["a"][i] for i in ids])
    #     rewards = np.asarray([self.experience["r"][i] for i in ids])
    #     next_states = np.asarray([self.experience["s'"][i] for i in ids])
    #     dones = np.asarray([self.experience["done"][i] for i in ids])
    #     return states, actions, rewards, next_states, dones

    def train(self, off_policy):
        """
        Performs Q-Learning training on the primary network
        using a secondary network as the off-policy weight updates.

        Parameters
        ----------
        off_policy : Network()
            Target Network imitating 
        """
        # Only train once the expericence buffer is above a certain size
        if len(self.experience["s"]) < self.min_buf_size:
            return 0
        
        ids = self.rand_gen.integers(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience["s"][i] for i in ids])
        actions = np.asarray([self.experience["a"][i] for i in ids])
        rewards = np.asarray([self.experience["r"][i] for i in ids])
        next_states = np.asarray([self.experience["s'"][i] for i in ids])
        dones = np.asarray([self.experience["done"][i] for i in ids])

        # Recieve random sample batch
        # states, actions, rewards, next_states, dones = self.get_sample_batch()

        # Use off-policy Target Network to generate y_actual
        next_value = np.max(off_policy.predict(next_states), axis = 1)
        y_actual = np.where(dones, rewards, rewards + self.gamma * next_value)

        with tf.GradientTape() as tape:
            # Forward pass on states
            # y_pred = self.predict(states)   # This may not work  for some reason???

            y_pred = tf.math.reduce_sum(self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)

            # Loss (MSE)
            loss = tf.math.reduce_mean(tf.square(y_actual - y_pred))
        
        # Compute gradients
        variables = self.network.trainable_variables
        gradients = tape.gradient(loss, variables)
        # Update Weights (Backpropagation)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return loss

    def get_action(self, states, epsilon):
        """
        Epsilon Greedy action selection
        """
        if self.rand_gen.random() < epsilon:
            a = self.rand_gen.choice(self.num_actions)
            return a
        else:        
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        """
        Add experience (s, a, r, s', done) into experience buffer 
        """
        if len(self.experience['s']) >= self.max_buf_size:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)
    
    def copy_weights(self, net_to_copy):
        """
        Copy weights from one network to another.
        """
        variables1 = self.network.trainable_variables
        variables2 = net_to_copy.network.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

def run_episode(env, TrainNet, TargetNet, epsilon, steps_to_copy):
    """
    """
    # Set up initial variables 
    reward_total = 0
    i = 0
    done = False
    # Reset State
    state = env.reset()

    while not done:
        # env.render()

        # Store current state
        current_state = state
        # Select nect action form primary network
        action = TrainNet.get_action(state, epsilon)
        # Take that action in the encironemt and recieve the next state, reward, and done bool
        state, reward, done, _ = env.step(action)
        # Add reward to reward total
        
        reward_total += reward
        # Done check ?????
        if done:
            # print(done, reward_total)
            # print("END OF EP\n")
            
            reward = -200
            env.reset()

        # Experience Buffer adding...
        exp = {"s": current_state, "a": action, "r": reward, "s'": state, "done": done}
        TrainNet.add_experience(exp)

        # TRAINAINAINAINAIN
        TrainNet.train(TargetNet)

        # Copy weights to target network from primary network
        i += 1
        if i % steps_to_copy == 0:
            TargetNet.copy_weights(TrainNet)

    return reward_total

def main():
    # Hyperparameters
    env = gym.make("CartPole-v0")
    gamma = 0.99
    copy_step = 25
    num_states = env.observation_space.shape[0]     # Box --> 4
    num_actions = env.action_space.n                # Discrete -- > 2
    layers = [200, 200]
    max_buf_size = 10000
    min_buf_size = 100
    batch_size = 32

    epsilon = 0.5

    # Establish Target Network
    TargetNet = DQ_Model(num_states, layers, num_actions, min_buf_size, max_buf_size, batch_size, gamma)
    # Establish Model
    TrainNet = DQ_Model(num_states, layers, num_actions, min_buf_size, max_buf_size, batch_size, gamma)

    episodes = 2000
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        epsilon = max(0.999 * epsilon, 0.1)
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
    
    env.close()

if __name__ == "__main__":
    main()
