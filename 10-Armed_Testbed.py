"""
COMP532 - Assignment 1
----------------------
Program to reimplement the 10-Armed Testbed Example present within 'Reinforcement Learning: An 
Introduction' by Richard S. Sutton and Andrew G. Barto (Second Edition), Figure 2.2 [Pg. 23].

Authors
-------
Thomas Rex Greenway, 201198319
Kit Bower-Morris,
Nicholas Bryan, 
"""

import numpy as np
import matplotlib.pyplot as plt


class Agent():
    """
    Agent class that selects actions (which arm to pull) from the environment and interprets
    the value of the reward produced when specific action is taken.

    Parameters
    ----------
    arms : int
        Number of arms for the agent to choose between (expected number of potential actions).
    eNum : float < 1
            Probability for the agent to not choose the optimum action in order to 'explore'.
            A value of 0 result in a Greedy-Method Agent.

    Attributes
    ----------
    previousAct : int < arms
        The last performed action in the environment, established with each call of the
        interpreter method. Actions are represented as an index value.
    numActs : NumPy Array
        Number of times an each action is selected.
    sumActs : NumPy Array
        Sum of rewards for each action.
    actVals : NumPy Array
        Action value (Q value) estimates for each action.

    Notes
    -----
    Actions are not stored explicitly and are instead represented by their index position in
    the various related value arrays.
    """
    def __init__(self, arms, eNum):
        """
        Initialises an agent with given arms and episilon value.
        """
        self.arms = arms
        self.eNum = eNum

        # Variables used to adapt action value estimates.
        self.previousAct = None
        self.numActs = np.zeros(self.arms)
        self.sumActs = np.zeros(self.arms)

        # Stored Action Value Estimates
        self.actVals = np.zeros(self.arms)

        # Numpy Random Generator
        self.randGen = np.random.default_rng()

    def __str__(self):
        """
        Returns the type of agent with respect to its exploratory probability value, epsilon (eNum).
        """
        if self.eNum == 0:
            return "\u03B5 = 0 (greedy)"
        else:
            return f"\u03B5 = {self.eNum}"
    
    def action(self):
        """
        The policy, or action selection class method. Returns the action for the agent to take based
        upon its current understanding of the environment.

        Returns
        -------
        action : action
            The selected action based on the Agent's current action value estimates.
        """
        # Epsilon Choice (If eNum = 0 then Greedy-Method)
        eNumSelector = self.randGen.random()
        
        if eNumSelector < self.eNum:
            # picks random arm from all arms
            # action = randGenerator.choice(self.arms)

            action = np.random.choice(self.arms)

        # Greedy Choice
        else: 
            action = np.argmax(self.actVals)                                 # Select index (action) of largest actVal
            maxActs = np.where(self.actVals == self.actVals[action])[0]      # Array of indices (actions) with max actVal

            # When there are multiple greedy actions, choose one at random.
            if len(maxActs) > 1:
                # action = randGenerator.choice(maxActs)
                action = np.random.choice(maxActs)

        # Encode the previous action
        self.previousAct = action

        return action
    
    def interpreter(self, reward):
        """
        Re-evaluates value estimates based upon recieved reward.

        Parameters
        ----------
        reward : float
            Number recieved from the environment after performing a particular action.
        """
        # Increment action counter
        self.numActs[self.previousAct] += 1

        # Add reward for the previous action.
        self.sumActs[self.previousAct] += reward

        # New actVal estimate
        qEstimate = self.sumActs[self.previousAct]/self.numActs[self.previousAct]

        # Set new action value
        self.actVals[self.previousAct] = qEstimate

    def reset(self):
        """
        Resets the Agent class to default values.
        """
        self.previousAct = None
        self.numActs[:] = 0
        self.sumActs[:] = 0
        self.actVals[:] = 0


class Environment():
    """
    Environment class interacts with Agents to take actions and return rewards. This environment also
    sets the Testbed according to the standard normal distribution.

    Parameters
    ----------
    plays : int
        Number of times agents perform an action upon the Testbed within an iteration.
    iterations : int 
        Number of different Testbeds for the agents to play/ perform on.
    """
    def __init__(self, plays, iterations):
        """
        Initialises an environment with set plays and iterations.
        """
        self.plays = plays
        self.iterations = iterations

        # Numpy Random Generator
        self.randGen = np.random.default_rng()
    
    def setTestbed(self, arms):
        """
        Sets a Testbed with standard normally distributed action values for a given number of actions
        (arms).

        Parameters
        ----------
        arms : int
            Number of actions in the Testbed.

        Attributes
        ----------
        actionVals : NumPy Array
            The random 'true' action values for this Testbed.
        bestAct : action
            The definitive optimum action in the current testbed.
        """
        # Set random TestBed action values from Standard Normal Distribution
        self.actionVals = self.randGen.normal(0, 1, arms)
        self.bestAct = np.argmax(self.actionVals)

    def getReward(self, action):
        """
        Returns a normally distributed reward about the true action value.

        Returns
        -------
        reward : float
            Reward selected from normal distribution with mean = 'true' action value, and standard
            deviation = 1. 
        """
        # Get random reward from Normal Distribution with action value mean.    
        return self.randGen.normal(self.actionVals[action], 1)
    
    def play(self, agents):
        """
        Performs a series of actions upon the environment according to agent policies, passing reward values
        back to agents after each action.

        Parameters
        ----------
        agents : list
            List of Agent class objects to interact with the environment.
        
        Returns
        -------
        scoreAvg : NumPy Array
            "plays X num. of agents" Array with the average score of each play across all iterations.
        optimumAvg : NumPy Array
            "plays X num. of agents" Array with the average number of optimum action choices across all
            iterations.

        Notes
        -----
        The play method returns the score and optimum average arrays to be utilisied in the drawing of the
        agent performance graphs.
        """
        scores = np.zeros((self.plays, len(agents)))
        optimumCount = np.zeros((self.plays, len(agents)))

        for i in range(self.iterations):

            # Tracks Progress
            if (i%100) == 0:
                print("Completed Iterations: ", i)

            # Reset agents
            for agent in agents:
                agent.reset()
            # Set new Testbed for this iteration with num of actions equal to the agents expected amount (arms).
            # All agents should have equal expected actions.
            self.setTestbed(agents[0].arms)

            # Play this testbed the given number of times.
            for j in range(self.plays):
                agentCount = 0

                for agent in agents:
                    # Pull arm/ do action
                    action = agent.action()
                    # Get reward
                    reward = self.getReward(action)
                    # Interpret Reward
                    agent.interpreter(reward)

                    # Score and Optimum tracking
                    scores[j, agentCount] += reward
                    if action == self.bestAct:
                        optimumCount[j, agentCount] += 1
                    
                    #increment agent counter
                    agentCount += 1
        
        scoreAvg = scores/self.iterations
        optimumAvg = 100 * (optimumCount/self.iterations)

        return scoreAvg, optimumAvg



if __name__ == "__main__":
    # Time tracking

    agents = [Agent(10, 0), Agent(10, 0.1), Agent(10, 0.01)]
    env = Environment(1000, 2000)

    print("Running -->")
    score, opt = env.play(agents)

    print("\nDrawing --> ")

    # Graphing Score Averages
    plt.plot(score[: , 0], "g")
    plt.plot(score[: , 2], "r")
    plt.plot(score[: , 1], "k")
    plt.yticks(np.arange(0, 1.6, step=0.5))
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')
    plt.legend(agents, loc=4)
    plt.show()


    # Graphing Optimum Averages
    plt.plot(opt[: , 0], "g")
    plt.plot(opt[: , 2], "r")
    plt.plot(opt[: , 1], "k")
    plt.ylim(0, 100)
    plt.ylabel('% Optimal Action')
    plt.xlabel('Steps')
    plt.legend(agents, loc=4)
    plt.show()