# search.py
# ---------
# Modify this file
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


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    # from lecture notes 1/19/22
    # Graph search from lecture slides 1/24/22 with slight modifications
    closed = set()
    fringe = util.Stack()
    # fringe is stack of tuple (state, actions) where actions are a list
    fringe.push((problem.getStartState(), []))
    while fringe:
        node, moves = fringe.pop()
        if problem.isGoalState(node):
            return moves
        if node not in closed:
            closed.add(node)
            for s1, a1, c1 in problem.getSuccessors(node):
                # getSuccessors(state) returns (successorState, action, cost) list triple
                actions = moves + [a1]
                fringe.push((s1, actions))


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    # 1/24/22 lecture slides graph search; basically the same as DFS, but we use a Queue instead
    closed = set()
    fringe = util.Queue()
    # fringe is stack of tuple (state, actions) where actions are a list
    fringe.push((problem.getStartState(), []))
    while fringe:
        node, moves = fringe.pop()
        if problem.isGoalState(node):
            return moves
        if node not in closed:
            closed.add(node)
            for s1, a1, c1 in problem.getSuccessors(node):
                # getSuccessors(state) returns (successorState, action, cost) list triple
                actions = moves + [a1]
                fringe.push((s1, actions))


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    # 1/24/22 lecture slides graph search; similar to BFS, but we use a PriorityQueue instead with costs
    closed = set()
    fringe = util.PriorityQueue()
    # fringe is stack of tuple ((state, actions, cost), priority) where actions are a list
    fringe.push((problem.getStartState(), [], 0), 0)
    while fringe:
        node, moves, cost = fringe.pop()
        if problem.isGoalState(node):
            return moves
        if node not in closed:
            closed.add(node)
            for s1, a1, c1 in problem.getSuccessors(node):
                # getSuccessors(state) returns (successorState, action, cost) list triple
                actions = moves + [a1]
                prior = problem.getCostOfActions(actions)
                fringe.push((s1, actions, prior), prior)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    # lecture slides 1/26/22; like UCS, but we also use a heuristic cost PriorityQueue along with the regular cost
    closed = set()
    fringe = util.PriorityQueue()
    # fringe is stack of tuple ((state, actions, cost), heuristicPriority where actions are a list
    fringe.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem))
    while fringe:
        node, moves, cost = fringe.pop()
        if problem.isGoalState(node):
            return moves
        if node not in closed:
            closed.add(node)
            for s1, a1, c1 in problem.getSuccessors(node):
                # getSuccessors(state) returns (successorState, action, cost) list triple
                actions = moves + [a1]
                newcost = problem.getCostOfActions(actions)
                # this is the only difference from the UCS, we use heuristic+cost for our priority
                hcost = problem.getCostOfActions(actions) + heuristic(s1, problem)
                fringe.push((s1, actions, newcost), hcost)

# # Node class from lecture slides 1/24/22; this is not used anywhere in this program
# class Node:
#
#     def __init__(self, state, action, cost, parent):
#         self.actions = []  # e.g., to store full sequence of actions to get to node from root
#         self.actionFromParent = action
#         self.cost = cost
#         self.state = state
#         self.parent = parent
#         self.depth = 0
#
#         if parent:
#             self.depth = parent.depth + 1
#         # more code needed here to maintain cost, depth, etc. (see lecture slides)
#
#     def getActions(self):
#         # code here to compute path back to root
#         return self.actions
#
#     def expand(self, problem):
#         return [self.successor(problem, action)
#                     for action in problem.actions(self.state)]
#
#     def successor(self, problem, action):
#         next = problem.result(self.state, action)
#         return Node(next, self, action)
#
#     def path(self):
#         node, path_back = self, []
#         while node:
#             path_back.append(node)
#             node = node.parent
#         return list(reversed(path_back))
#
#         # can have other useful methods, like cost updates, support methods for hashing, etc (see lecture slides)
#
#     def __eq__(self, other):
#         return isinstance(other, Node) and self.state == other.state
#
#     def __hash__(self):
#         return hash(self.state)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
