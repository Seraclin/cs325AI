# multiAgents.py
# --------------
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
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        currentPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        closestDist = -10000000
        foodlist = currentFood.asList()
        # Basic strat: eat closest food, and avoid any spaces with ghosts, keep moving
        # don't ever want to stop moving
        if action == 'Stop':
            return -10000000
        # avoid any spaces with ghosts
        for ghosts in newGhostStates:
            if ghosts.getPosition() == newPos:
                return -10000000

        # find the closest food
        for food in foodlist:
            current = -1*(util.manhattanDistance(newPos, food)) # negative so lower dist = higher evalue
            if current > closestDist:
                closestDist = current

        return closestDist
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
            Very picky about how many times this is called.

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        # Based on the lecture slides 2/9/22; agentIndex: 0 = pacman, >0 = ghost
        result = self.minmaxdecision(gameState, 0, 0) # returns an evalue, action
        return result[1]  # should return only an action

    def minmaxdecision(self, gameState, agentIndex, depth):
        # from lecture slides 2/9/22; but with agentIndex and depth value
        # either return terminal state, max-value, or min-value
        # reset agentIndex if all players have went, we've traversed one level
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            depth = depth + 1
        # terminal state; returns score, action
        if len(gameState.getLegalActions(agentIndex)) == 0 or self.depth == depth:
            return gameState.getScore(), ""
        # pacman's turn (MAX)
        if agentIndex == 0:
            v = self.maxvalue(gameState, agentIndex, depth)
            return v
        else: # ghost turn > 0 (MIN)
            v = self.minvalue(gameState, agentIndex, depth)
            return v
    # return utility, action for max/min agent states
    def maxvalue(self, gameState, agentIndex, depth):
        v = float("-inf") # negative inf
        maxAction = ""
        pacmanAction = gameState.getLegalActions(agentIndex)
        for action in pacmanAction:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorValue = self.minmaxdecision(successor, agentIndex + 1, depth)[0]
            if successorValue > v:
                v = successorValue
                maxAction = action
        return v, maxAction

    def minvalue(self, gameState, agentIndex, depth):
        v = float("inf")  # pos inf
        minAction = ""
        ghostAction = gameState.getLegalActions(agentIndex)
        for action in ghostAction:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorValue = self.minmaxdecision(successor, agentIndex + 1, depth)[0]
            if successorValue < v:
                v = successorValue
                minAction = action
        return v, minAction



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # from lecture slides 2/14/22, basically the minmax code but with alpha/beta and not pruning on equality
        inf = float("inf")
        ninf = float("-inf")
        result = self.minmaxdecision(gameState, 0, 0, ninf, inf)  # returns evalue, action
        return result[1]  # should return only an action

    def minmaxdecision(self, gameState, agentIndex, depth, alpha, beta):
        # from lecture slides 2/9/22; but with alpha/beta and no equality pruning
        # either return terminal state, max-value, or min-value
        # reset agentIndex if all players have went, we've traversed one level
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            depth = depth + 1
        # terminal state; returns score, action
        if len(gameState.getLegalActions(agentIndex)) == 0 or self.depth == depth:
            return gameState.getScore(), ""
        # pacman's turn (MAX)
        if agentIndex == 0:
            v = self.maxvalue(gameState, agentIndex, depth, alpha, beta)
            return v
        else:  # ghost turn > 0 (MIN)
            v = self.minvalue(gameState, agentIndex, depth, alpha, beta)
            return v
        # return utility, action for max/min agent states

    def maxvalue(self, gameState, agentIndex, depth, alpha, beta):
        v = float("-inf")  # negative inf
        maxAction = ""
        pacmanAction = gameState.getLegalActions(agentIndex)
        for action in pacmanAction:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorValue = self.minmaxdecision(successor, agentIndex + 1, depth, alpha, beta)[0]
            if successorValue > v:
                v = successorValue
                maxAction = action
            if v > beta: # do not prune on equality!
                return v, maxAction
            alpha = max(alpha, v)
        return v, maxAction

    def minvalue(self, gameState, agentIndex, depth, alpha, beta):
        v = float("inf")  # pos inf
        minAction = ""
        ghostAction = gameState.getLegalActions(agentIndex)
        for action in ghostAction:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorValue = self.minmaxdecision(successor, agentIndex + 1, depth, alpha, beta)[0]
            if successorValue < v:
                v = successorValue
                minAction = action
            if v < alpha: # don't prune on equality
                return v, minAction
            beta = min(beta, v)
        return v, minAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        # from lecture slides 2/14/22; basically like minmax but replacing min-value with exp-value
        result = self.expectidecision(gameState, 0, 0)  # get evalue, action
        return result[1]  # should return only an action

    def expectidecision(self, gameState, agentIndex, depth):
        # from lecture slides 2/14/22; basically like minmax but replacing min-value with exp-value
        # either return terminal state, max-value, or min-value
        # reset agentIndex if all players have went, we've traversed one level
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            depth = depth + 1
        # terminal state; returns score, action
        if len(gameState.getLegalActions(agentIndex)) == 0 or self.depth == depth:
            return self.evaluationFunction(gameState), ""
        # pacman's turn (MAX)
        if agentIndex == 0:
            v = self.maxvalue(gameState, agentIndex, depth)
            return v
        else:  # ghost turn > 0 (expected value)
            v = self.expvalue(gameState, agentIndex, depth)
            return v
        # return utility, action for max/min agent states

    def maxvalue(self, gameState, agentIndex, depth):
        v = float("-inf")  # negative inf
        maxAction = ""
        pacmanAction = gameState.getLegalActions(agentIndex)
        for action in pacmanAction:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorValue = self.expectidecision(successor, agentIndex + 1, depth)[0]
            if successorValue > v:
                v = successorValue
                maxAction = action
        return v, maxAction

    def expvalue(self, gameState, agentIndex, depth):
        v = 0
        expaction = ""
        ghostAction = gameState.getLegalActions(agentIndex)
        p = 1.0 / len(ghostAction) # assuming uniform probability
        for action in ghostAction:
            # we return an expected value = probability * value instead of a minvalue
            successor = gameState.generateSuccessor(agentIndex, action)
            successorValue = self.expectidecision(successor, agentIndex + 1, depth)[0]
            v += p*successorValue
        return v, expaction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: linear combo of: score + inverse of closest food dist + negative food count + negative # of capsules
      Focus on closest food and not leaving any food behind.
    """
    pacPos = currentGameState.getPacmanPosition()
    ghostPos = currentGameState.getGhostPositions() # could probably improve with ghost distances, but works well enough
    foodList = currentGameState.getFood().asList()
    foodCount = len(foodList)
    capsules = currentGameState.getCapsules()

    # finding the closest food
    closestFood = float("inf")
    for food in foodList:
        currentDist = util.manhattanDistance(pacPos, food)
        if currentDist < closestFood:
            closestFood = currentDist
    # linear function for evalue; negative means larger = bad;
    # emphasize getting closest food, and not leaving any food left. Then focus on score and capsules left.
    evalue = 300*(1.0/closestFood) + 20*scoreEvaluationFunction(currentGameState) + -100*foodCount + -1*len(capsules)
    return evalue


# Abbreviation
better = betterEvaluationFunction

# contest, I don't have any code below here
class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

