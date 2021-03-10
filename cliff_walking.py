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

EPISODES = 500      # the episode ends when the agent ether falls off the cliff or reaches the goal.
ROUNDS = 10


# AGENT --> Its position, policy, interpreter(SARSA) and interpreter(Q-Learning)
class Agent():
    """
    IMPLEMENT
    """
    def __init__(self, eNum = 0.1):
        """
        IMPLEMENT
        """
        self.eNum = eNum
        
        # Agent Position (temporary origin untill esatblished upond play in environment)
        self.pos = (3, 0)

        # Possible actions
        self.actions = ["up", "right", "down", "left"]

        self.previousActIndex = None
        self.previousState = None

        # Establish 3 dimensional array for storing Q(s, a) estimates.
        # First 2 axis point to a state i.e. a coordinate position on the board.
        # Last position points to the various potential actions that can be chosen from that state.
        self.actValEstimates = np.zeros((4, 12, len(self.actions)))

        # Numpy Random Generator
        self.randGen = np.random.default_rng()

    def action(self):
        """
        IMPLEMENT
        """
        # Encode current state for interpreter
        self.previousState = self.pos

        # Epsilon Choice (If eNum = 0 then Greedy-Method)
        eNumSelector = self.randGen.random()
        
        if eNumSelector < self.eNum:
            # picks random arm from all arms
            action = self.randGen.choice(self.actions)
        
        # Greedy Choice
        else:
            # Gets the action values for current position
            stateActVals = self.actValEstimates[self.pos[0], self.pos[1], :]

            # Index of action with max actVal (greedy action)
            actionIndex = np.argmax(stateActVals)

            # Check for other greedy actions.
            maxActs = np.where(stateActVals == stateActVals[actionIndex])[0]
            # When there are multiple greedy actions, choose one at random.
            if len(maxActs) > 1:
                actionIndex = self.randGen.choice(maxActs)

            action = self.actions[actionIndex]

        # Encode the previous action (as index)
        self.previousActIndex = self.actions.index(action)
        
        return action

    def interpreterSarsa(self, reward):
        """
        state, action, reward, nxtState, nxtAct 
        """
        # Get previous action nand state
        a = self.previousActIndex
        s = self.previousState
        
        # State after action
        nxtState = self.pos

        # Action index for next action.
        nxtAct = self.actions.index(self.action())
        
        currentActVal = self.actValEstimates[s[0], s[1], a]
        nxtActVal = self.actValEstimates[nxtState[0], nxtState[1], nxtAct]

        # Update action value estimate
        self.actValEstimates[s[0], s[1], a] += self.eNum * (reward + nxtActVal - currentActVal)
        
        # Return next action to be used next....
        return nxtAct

    def interpreterQLearn(self, reward):
        """
        IMPLEMENT
        """
        # Get previous action nand state
        a = self.previousActIndex
        s = self.previousState

        # State after action
        nxtState = self.pos

        currentActVal = self.actValEstimates[s[0], s[1], a]
        nxtActVal = max(self.actValEstimates[nxtState[0], nxtState[1], :])      # max action val for next state

        # Update action value estimate
        self.actValEstimates[s[0], s[1], a] += self.eNum * (reward + nxtActVal - currentActVal)

    def reset(self):
        """
        IMMPLEMENT
        """
        self.previousActIndex = None
        self.previousState = None
        self.actValEstimates = np.zeros((4, 12, len(self.actions)))



# CLIFF --> Board, positionMoves, reward return, PLAY???? (play in the environment, interpret in the agent)
class Environment():
    """
    IMPLEMENT
    """
    def __init__(self, episodes):
        """
        IMPLEMENT
        """
        self.episodes = episodes

        self.end = False

        self.board = np.zeros((4, 12))
        # Cliff face descibed by -1 grid values.
        self.board[3, 1:11] = -1

        # Start and Goal positions        
        self.S = (3, 0)
        self.G = (3, 11)

    def movePlayer(self, position, action):
        """
        IMPLEMENT
        """
        # Move around the board according to given action.
        if action == "up":
            nxtPos = (position[0] - 1, position[1])
        elif action == "down":
            nxtPos = (position[0] + 1, position[1])
        elif action == "left":
            nxtPos = (position[0], position[1] - 1)
        else:
            nxtPos = (position[0], position[1] + 1)
        
        # Check that the next position is on the board. If not: do not move.
        if nxtPos[0] >= 0 and nxtPos[0] <= 3:
            if nxtPos[1] >= 0 and nxtPos[1] <= 11:
                position = nxtPos

        # Leave the goal or cliff check for the play part....
        if position == self.G:
            self.end = True
            print("REACHED GOAL")
        if self.board[position] == -1:
            self.end = True
            print("FALLS OFF CLIFF")

        return position
    
    def endOfEpisode(self, position):
        """
        IMPLEMENT
        """
        if position == self.G or self.board[position] == -1:
            return True
        else:
            return False

    def getReward(self, position):
        """
        IMPLEMENT
        """
        # If the player is on the cliff return -100
        if self.board[position] == -1:
            return -100

        # Else, for all other positions, return -1
        else:
            return -1

    def play(self, agent, method = "sarsa"):
        """
        IMPLEMENT
        """
        rewardSum = np.zeros(self.episodes)
        
        for i in range(self.episodes):

            # Tracker
            if (i%5) == 0:
                print(f"Episodes Completed: {i}")

            # Initilialise starting state
            agent.pos = self.S

            # SARSA
            if method == "sarsa":
                # Choose initial action
                a = agent.action()

                # Iterate for each step of the episode
                while 1:
                    agent.pos = self.movePlayer(agent.pos, a)
                    reward = self.getReward(agent.pos)

                    # Add reward to sum tracker
                    rewardSum[i] += reward
                    
                    # Re-evaluate action value estimates and grab next action.
                    a = agent.interpreterSarsa(reward)

                    if self.end:
                        break
            
            # Q-Learning
            elif method == "qlearn":
                # Iterate for each step of the episode
                while 1:
                    action = agent.action()
                    agent.pos = self.movePlayer(agent.pos, action)
                    reward = self.getReward(agent.pos)

                    # Add reward to sum tracker
                    rewardSum[i] += reward
                    
                    # Re-evaluate action value estimates
                    agent.interpreterQLearn(reward)

                    if self.end:
                        break

        return rewardSum
                
            

if __name__ == "__main__":
    agent = Agent()
    env = Environment(100)


    env.play(agent, "qlearn")
    print(agent.actValEstimates)