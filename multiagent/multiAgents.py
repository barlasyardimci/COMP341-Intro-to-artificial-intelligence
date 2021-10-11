# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food_pos = 0
        ghost_pos = 0
        food_poss = list()

        if successorGameState.isWin():
            return float("Inf")

        for states in newGhostStates:
            for times in newScaredTimes:
                if times < 1:
                    if util.manhattanDistance(states.getPosition(), newPos) < 2:
                        return float("-Inf")
                    else:
                        ghost_pos = -util.manhattanDistance(states.getPosition(), newPos)
                else:
                    ghost_pos = 200
        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y] == True:
                    food_poss.append(util.manhattanDistance(newPos, (x, y)))
        food_pos = min(food_poss)
        food_val = 0
        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            food_val = 100
        return -2 * food_pos + ghost_pos + food_val


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        possibleMoves = gameState.getLegalActions(0)
        pacmanScores = list()
        numbers = list()

        for moves in possibleMoves:
            nextState = gameState.generateSuccessor(0, moves)
            pacmanScores.append(self.minimax(nextState, 1, 0))
        best = max(pacmanScores)
        for i in range(len(pacmanScores)):
            if pacmanScores[i] == best:
                numbers.append(i)
        bestMoveChoice = random.choice(numbers)
        return possibleMoves[bestMoveChoice]

    def minimax(self, state, agentIndex, depth):
        if agentIndex == state.getNumAgents():
            depth += 1

        agentIndex = agentIndex % state.getNumAgents()
        if depth == self.depth:
            return self.evaluationFunction(state)

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if not state.getLegalActions(agentIndex):
            return self.evaluationFunction(state)

        if agentIndex == 0:
            v0 = -999999
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                v0 = max(v0, self.minimax(nextState, agentIndex + 1, depth))
            return v0
        else:
            v1 = 999999
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                v1 = min(v1, self.minimax(nextState, agentIndex + 1, depth))
            return v1



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        possibleMoves = gameState.getLegalActions(0)
        pacmanScores = list()
        numbers = list()
        alpha = -999999
        beta = 999999

        for moves in possibleMoves:
            nextState = gameState.generateSuccessor(0, moves)
            notPruned = self.abpruning(nextState, 1, 0, alpha, beta)
            pacmanScores.append(notPruned)
            alpha = max(alpha,notPruned)
        best = max(pacmanScores)
        for i in range(len(pacmanScores)):
            if pacmanScores[i] == best:
                numbers.append(i)
        bestMoveChoice = random.choice(numbers)
        return possibleMoves[bestMoveChoice]

    def abpruning(self, state, agentIndex, depth, alpha, beta):
        if agentIndex == state.getNumAgents():
            depth += 1

        agentIndex = agentIndex % state.getNumAgents()
        if depth == self.depth:
            return self.evaluationFunction(state)

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if not state.getLegalActions(agentIndex):
            return self.evaluationFunction(state)

        if agentIndex == 0:
            v0 = -999999
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                v0 = max(v0, self.abpruning(nextState, agentIndex + 1, depth, alpha, beta))
                if v0 >= beta:
                    return v0
                alpha = max(alpha, v0)
            return v0
        else:
            v1 = 999999
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                v1 = min(v1, self.abpruning(nextState, agentIndex + 1, depth, alpha, beta))
                if v1 <= alpha:
                    return v1
                beta = min(beta, v1)
            return v1


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
        "*** YOUR CODE HERE ***"
        possibleMoves = gameState.getLegalActions(0)
        pacmanScores = list()
        numbers = list()

        for moves in possibleMoves:
            nextState = gameState.generateSuccessor(0, moves)
            pacmanScores.append(self.Expectimax(nextState, 1, 0))
        best = max(pacmanScores)
        for i in range(len(pacmanScores)):
            if pacmanScores[i] == best:
                numbers.append(i)
        bestMoveChoice = random.choice(numbers)
        return possibleMoves[bestMoveChoice]

    def Expectimax(self, state, agentIndex, depth):
        if agentIndex == state.getNumAgents():
            depth += 1

        agentIndex = agentIndex % state.getNumAgents()
        if depth == self.depth:
            return self.evaluationFunction(state)

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if not state.getLegalActions(agentIndex):
            return self.evaluationFunction(state)

        if agentIndex == 0:
            v0 = -999999
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                v0 = max(v0, self.Expectimax(nextState, agentIndex + 1, depth))
            return v0
        else:
            v1 = 0
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                p = float(1/len(state.getLegalActions(agentIndex)))
                v1 = float(p * self.Expectimax(nextState, agentIndex + 1, depth))
            return v1


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currScore = currentGameState.getScore() * 3

    if currentGameState.isWin():
        return float("Inf")
    if currentGameState.isLose():
        return float("-Inf")

    currPos = currentGameState.getPacmanPosition()
    currFoods = currentGameState.getFood()
    currCapsulesLocations = currentGameState.getCapsules()
    if currCapsulesLocations:
        nearestCapsuleDistance = min([manhattanDistance(capsule, currPos) for capsule in currCapsulesLocations])
    else:
        nearestCapsuleDistance = 0
    food_poss = list()
    for x in range(currFoods.width):
        for y in range(currFoods.height):
            if currFoods[x][y] == True:
                food_poss.append(util.manhattanDistance(currPos, (x, y)))
    food_pos = min(food_poss)
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    for states in currGhostStates:
        for times in currScaredTimes:
            if times < 1:
                if util.manhattanDistance(states.getPosition(), currPos) < 2:
                    return float("-Inf")
                else:
                    ghost_pos = -util.manhattanDistance(states.getPosition(), currPos)
            else:
                ghost_pos = 200
    return currScore + 1.5*ghost_pos - 2*food_pos - 0.75*nearestCapsuleDistance


# Abbreviation
better = betterEvaluationFunction
