# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # From lecture slides 3/22/22
        # python autograder.py -q q1
        for t in range(iterations):  # for each iteration
            valuesCopy = self.values.copy()  # copy of the counter values to help avoid bugs
            for state in self.mdp.getStates():  # get each state
                maxQ = None
                actions = self.mdp.getPossibleActions(state)
                for action in actions:  # get the possible qvalues for each action in that state
                    currentQ = self.getQValue(state, action)
                    if maxQ is None or currentQ > maxQ:  # find the maxQ action
                        maxQ = currentQ
                if maxQ is None:  # if there are no actions for the current state, then no future rewards
                    maxQ = 0
                valuesCopy[state] = maxQ  # store a maxQvalue in copy of counter value for that state
            self.values = valuesCopy  # copy all values over

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        val = 0
        transition = self.mdp.getTransitionStatesAndProbs(state, action)
        discount = self.discount
        for transitionState, transitionProb in transition:
            reward = self.mdp.getReward(state, action, transitionState)
            val += transitionProb * (reward + discount * self.values[transitionState])

        return val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        maxValue = None
        actions = self.mdp.getPossibleActions(state)  # all possible actions for the state
        if self.mdp.isTerminal(state) or len(actions) == 0:  # if we reach a terminal state
            return None

        for action in actions:  # go through each action and find the best Qvalue one
            value = self.getQValue(state, action)  # self.values[state]
            if maxValue is None or value > maxValue:
                maxValue = value
                bestAction = action

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
