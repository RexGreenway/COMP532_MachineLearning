"""
COMP532 - Assignment 1
----------------------
Program to reimplement the Cliff Walking Example present within 'Reinforcement Learning: An 
Introduction' by Richard S. Sutton and Andrew G. Barto (Second Edition), Figure 6.4 [Pg. 108].

Authors
-------
Thomas Rex Greenway, 201198319
Kit Bower-Morris, 201532917
Nicholas Bryan, 201531951
"""

import numpy as np
import matplotlib.pyplot as plt

class Agent():
    """
    Agent class that selects actions (movements on a grid) in the environment and interprets
    the reward to produce state-action value estimates.

    Parameters
    ----------
    eNum : float < 1
        Probability for the agent to not choose the optimum action in order to 'explore'.
        A value of 0 result in a Greedy-Method Agent.

    Attributes
    ----------
    actions : list
        List of actions (movements) that the agent can take within the environment. 
    actValEstimates : 3-Dimensional NumPy Array
        Array storing the state-action value estimates with the state (position) represented
        by the first 2 indcices, and the action in the last dimension.

    """
    def __init__(self, eNum = 0.1):
        """
        Initialises the agent with given epsilon value.
        """
        self.eNum = eNum

        # Possible actions
        self.actions = ["up", "right", "down", "left"]

        # Establish 3 dimensional array for storing Q(s, a) estimates.
        # First 2 axis point to a state i.e. a coordinate position on the board.
        # Last position points to the various potential actions that can be chosen from that state.
        self.actValEstimates = np.zeros((4, 12, len(self.actions)))

        # Numpy Random Generator
        self.randGen = np.random.default_rng()

    def action(self, state):
        """
        The policy, or action selector, that returns the desired action when in a given state.

        Parameters
        ----------
        state : position (x, y)
            The state of the environemnt (or the position of the Agent within the grid).

        Returns
        -------
        action : action
            The selected action based on the given state and the Agent's current action value
            estimates.
        """
        # Epsilon Choice (If eNum = 0 then Greedy-Method)
        eNumSelector = self.randGen.random()
        
        if eNumSelector < self.eNum:
            # picks random arm from all arms
            action = self.randGen.choice(self.actions)
        
        # Greedy Choice
        else:
            # Gets the action values for current position
            stateActVals = self.actValEstimates[state[0], state[1], :]

            # Index of action with max actVal (greedy action)
            actionIndex = np.argmax(stateActVals)

            # Check for other greedy actions.
            maxActs = np.where(stateActVals == stateActVals[actionIndex])[0]
            # When there are multiple greedy actions, choose one at random.
            if len(maxActs) > 1:
                actionIndex = self.randGen.choice(maxActs)

            action = self.actions[actionIndex]
        
        return action

    def interpreterSarsa(self, prevState, prevAct, reward, nxtState, nxtAct):
        """
        Re-evaluates value estimates using SARSA.

        Parameters
        ----------
        prevState : position (x, y)
            The starting state of the environment.
        prevAct : action
            The action taken from the starting state.
        reward : int
            The reward produced when prevAct is taken from prevState.
        nxtState :  position (x, y)
            The state reached when prevAct is taken.
        nxtAct : action
            The next action to be made selected from existing state-action value estimates.

        """
        # Get action indices
        a = self.actions.index(prevAct)
        aPrime = self.actions.index(nxtAct)
        
        # Get action values
        currentActVal = self.actValEstimates[prevState[0], prevState[1], a]
        nxtActVal = self.actValEstimates[nxtState[0], nxtState[1], aPrime]

        # Update action value estimate
        self.actValEstimates[prevState[0], prevState[1], a] += self.eNum * (reward + nxtActVal - currentActVal)

    def interpreterQLearn(self, prevState, prevAct, reward, nxtState):
        """
        Re-evaluates value estimates using Q-Learning.

        Parameters
        ----------
        prevState : position (x, y)
            The starting state of the environment.
        prevAct : action
            The action taken from the starting state.
        reward : int
            The reward produced when prevAct is taken from prevState.
        nxtState :  position (x, y)
            The state reached when prevAct is taken.

        """
        # Get previous action nand state
        a = self.actions.index(prevAct)

        # Get action values
        currentActVal = self.actValEstimates[prevState[0], prevState[1], a]
        nxtActVal = max(self.actValEstimates[nxtState[0], nxtState[1], :])      # max action val for next state

        # Update action value estimate
        self.actValEstimates[prevState[0], prevState[1], a] += self.eNum * (reward + nxtActVal - currentActVal)

    def reset(self):
        """
        Resets the Agent class to default values.
        """
        self.previousActIndex = None
        self.previousState = None
        self.actValEstimates = np.zeros((4, 12, len(self.actions)))



# CLIFF --> Board, positionMoves, reward return, PLAY???? (play in the environment, interpret in the agent)
class Environment():
    """
    Environment class interacts with Agents to take actions, adapt states and return rewards.

    Parameters
    ----------
    episodes : int
        Number of times agents work through the cliff-walking (Learning).
    runs : int 
        Number of times to run through episodes.

    Attributes
    ----------
    end : boolean
        True if the environent is in an end state (at the goal or off the cliff). False if not.
    board : 2-Dimensional NumPy Array
        The grid of the envirnoment with value -1 where the cliff is defined to be.
    S : position (x, y)
        The starting position for agent.
    G : position (x, y)
        The end goal position for the agent.
    
    """
    def __init__(self, episodes, runs):
        """
        Initialises an environment with set episodes and runs.
        """
        self.episodes = episodes
        self.runs = runs

        self.end = False

        self.board = np.zeros((4, 12))
        # Cliff face descibed by -1 grid values.
        self.board[3, 1:11] = -1

        # Start and Goal positions        
        self.S = (3, 0)
        self.G = (3, 11)

        # Set Environment state:
        self.state = self.S

    def movePlayer(self, action):
        """
        Takes an action to change the stae of the environment, i.e moves the agent on the grid.

        Parameters
        ----------
        action : action 
            The action to move the agent in the grid by.

        Notes
        -----
        The end of episode detection also takes place within this method. When the state of the
        environment is at the goal, or off ther cliff, the end attribute is set to be True.
        """
        # Move around the board according to given action.
        if action == "up":
            nxtPos = (self.state[0] - 1, self.state[1])
        elif action == "down":
            nxtPos = (self.state[0] + 1, self.state[1])
        elif action == "left":
            nxtPos = (self.state[0], self.state[1] - 1)
        else:
            nxtPos = (self.state[0], self.state[1] + 1)
        
        # Check that the next position is on the board. If not: do not move.
        if nxtPos[0] >= 0 and nxtPos[0] <= 3:
            if nxtPos[1] >= 0 and nxtPos[1] <= 11:
                self.state = nxtPos

        # Leave the goal or cliff check for the play part....
        if self.state == self.G:
            self.end = True
            # print("REACHED GOAL at: ", self.state)
        if self.board[self.state] == -1:
            self.end = True
            # print("FALLS OFF CLIFF at: ", self.state)

    def getReward(self, position):
        """
        Returns a reward give the state of the environment.

        Returns
        -------
        reward : int
            Reward is defined to be -1 for all spaces except for when the state "falls" off the
            cliff, when this happens the reward is -100. 
        """
        # If the player is on the cliff return -100
        if self.board[position] == -1:
            return -100

        # Else, for all other positions, return -1
        else:
            return -1

    def play(self, agent, method = "sarsa"):
        """
        Performs a series of episodes of the cliff-walking game, estimates state-action values using
        the given method with a given agent.

        Parameters
        ----------
        agent : Agent()
            Agent class to learn the cliff walking game.
        method : "sarsa" (Default) or "qlearn"
            The desired policy control method.

        Returns
        -------
        rewardSum : NumPy Array
            Averaged array of cumulative rewards for each episode. 
        """
        rewardSum = np.zeros(self.episodes)

        # Iterate through given number of runs
        for j in range(self.runs):
            # Tracker (Runs)
            print("RUN: ", j)

            # Complete given number of episodes for each run
            for i in range(self.episodes):

                # Initilialise starting state
                self.state = self.S
                self.end = False

                # SARSA
                if method == "sarsa":
                    # Choose initial action
                    action = agent.action(self.state)

                    # Iterate for each step of the episode
                    while 1:
                        # Store Current Sate and Action
                        previousState = self.state
                        previousAct = action

                        # Move agent and recieve reward
                        self.movePlayer(action)
                        reward = self.getReward(self.state)

                        # Add reward to sum tracker
                        rewardSum[i] += reward

                        # Next Action (based on existing action values)
                        action = agent.action(self.state)
                        
                        # Re-evaluate action value estimates and grab next action.
                        agent.interpreterSarsa(previousState, previousAct, reward, self.state, action)

                        if self.end:
                            break
                
                # Q-Learning
                if method == "qlearn":
                    # Iterate for each step of the episode
                    while 1:
                        # Get action
                        action = agent.action(self.state)

                        # Store Current Sate and Action
                        previousState = self.state
                        previousAct = action

                        # Move agent and recieve reward
                        self.movePlayer(action)
                        reward = self.getReward(self.state)

                        # Add reward to sum tracker
                        rewardSum[i] += reward
                        
                        # Re-evaluate action value estimates
                        agent.interpreterQLearn(previousState, previousAct, reward, self.state)

                        if self.end:
                            break
                
        return rewardSum/self.runs
                

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

            

if __name__ == "__main__":
    agent = Agent()
    env = Environment(500, 10)

    # Play SARSA
    rewardSarsa = env.play(agent, "sarsa")

    agent.reset()

    # PLAY Q-LEARNING
    rewardQLearn = env.play(agent, "qlearn")

    # PLOT
    plt.plot(movingAvg(rewardSarsa))
    plt.plot(movingAvg(rewardQLearn))
    plt.ylim(-100, 0)
    plt.legend(["SARSA", "Q-LEARNING"], loc = 4, frameon = False)
    plt.show()