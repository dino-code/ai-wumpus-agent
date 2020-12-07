"""
Modified from wwagent.py written by Greg Scott

Modified to only do random motions so that this can be the base
for building various kinds of agent that work with the wwsim.py 
wumpus world simulation -----  dml Fordham 2019

# FACING KEY:
#    0 = up
#    1 = right
#    2 = down
#    3 = left

# Actions
# 'move' 'grab' 'shoot' 'left' right'

"""

"""
Artificial Intelligence Final Project by Dino Becaj
Began on 11/26/2020

Summary: Currently, the base agent selects an action randomly with the action() method. My goal here is to implement the model-checking approach 
(truth-table enumeration-based entailment) as the means by which the agent estimates the probabilities of the outcomes of its actions and then
selects an action.

Using truth-table enumeration, the agent will decide how safe it is in a given location. If there is a move that is 100% guaranteed as safe,
the agent will make that move. If there is no guaranteed safe move, the agent will make the next safest move.

self.map is the agent's knowledge base (KB) and is what will be used for entailment.
"""

from itertools import product # used to enumerate pit, wumpus, and gold configurations
import copy
from random import randint
import numpy as np

# This is the class that represents an agent

class WWAgent:

    def __init__(self):
        self.max=4 # number of cells in one side of square world
        self.stopTheAgent=False # set to true to stop th agent at end of episode
        self.position = (3, 0) # top is (0,0). The default was (0, 3), but I changed it so it would align with the actual simulation.
        self.directions=['up','right','down','left']
        self.facing = 'right'
        self.arrow = 1
        self.frontier = [((3, 1), False, False), ((2, 0), False, False)] # ((row, col), hasPit, hasWumpus). The 2 default frontier elements will always be the case
        self.known = set([((3,0), False, False)]) # all the squares appended to this list will have values of False (otherwise the game would end).
        self.percepts = (None, None, None, None, None)
        self.map = [[ self.percepts for i in range(self.max) ] for j in range(self.max)]
        self.probabilities = {}
        self.models = []
        print("New agent created")
    
    # Add the latest percepts to list of percepts received so far
    # This function is called by the wumpus simulation and will
    # update the sensory data. The sensor data is placed into a
    # map structured KB for later use
    def update(self, percept):
        self.percepts=percept
        #[stench, breeze, glitter, bump, scream]

        # puts the percept at the spot in the map where sensed
        if self.position[0] in range(self.max) and self.position[1] in range(self.max):
            self.map[ self.position[0]][self.position[1]]=self.percepts
        
        # update the squares seen in the frontier
        self.updateFrontier()

    # Since there is no percept for location, the agent has to predict
    # what location it is in based on the direction it was facing
    # when it moved
    def calculateNextPosition(self,action):
        # I had to reverse the positions because in CS, the first number means the row and
        # the second number means the column (row, column)
        # I did this because the agent technically starts in (3, 0), not (0, 3)
        if self.facing=='up':
            self.position = (max(0,self.position[0]-1), self.position[1])
        elif self.facing =='down':
            self.position = (min(self.max-1,self.position[0]+1), self.position[1])
        elif self.facing =='right':
            self.position = (self.position[0], min(self.max-1, self.position[1]+1))
        elif self.facing =='left':
            self.position = (self.position[0], max(0,self.position[1]-1))
        
        if (self.position, False, False) not in self.known:
                self.known.add((self.position, False, False)) # append the new position to known positions

        return self.position

    # and the same is true for the direction the agent is facing, it also
    # needs to be calculated based on whether the agent turned left/right
    # and what direction it was facing when it did
    def calculateNextDirection(self,action):
        if self.facing=='up':
            if action=='left':
                self.facing = 'left'
            else:
                self.facing = 'right'
        elif self.facing=='down':
            if action=='left':
                self.facing = 'right'
            else:
                self.facing = 'left'
        elif self.facing=='right':
            if action=='left':
                self.facing = 'up'
            else:
                self.facing = 'down'
        elif self.facing=='left':
            if action=='left':
                self.facing = 'down'
            else:
                self.facing = 'up'

    # returns a list of squares that are adjacent to the given list of squares
    # if known == True, find squares within self.known that are adjacent
    # to the square in question. If known == False, then find squares in the
    # frontier that are adjacent to the square in question
    def find_adjacents(self, square, frontier_model, known):
        adjacents_list = set([])
        
        if len(square) > 2:
            square = square[0]

        above = (square[0]-1, square[1])
        below = (square[0]+1, square[1])
        left = (square[0], square[1]-1)
        right = (square[0], square[1]+1)

        if known == True:
            # find adjacents that are also in known

            if (above, False, False) in self.known:
                adjacents_list.add(above)
            if (below, False, False) in self.known:
                adjacents_list.add(below)
            if (left, False, False) in self.known:
                adjacents_list.add(left)
            if (right, False, False) in self.known:
                adjacents_list.add(right)
        else:
            # find adjacents that are in the frontier
            
            for sq in frontier_model:
                if above in sq:
                    adjacents_list.add(sq)
                if below in sq:
                    adjacents_list.add(sq)
                if left in sq:
                    adjacents_list.add(sq)
                if right in sq:
                    adjacents_list.add(sq)
        
        return adjacents_list

    # check generated models using the sensor data.
    # Returns valid models (that will then be used in
    # the calculation of probabilities for pits and 
    # wumpuses)
    def checkModels(self, models):
        valid_models = []  # key=query variant, value=models where this variant is valid
                                # this will be used later when calculating probabilities

        for model in models:
            isValid = True           # we begin by assuming the model is valid
            
            # known squares that the frontier borders. 
            # Set bc we don't need duplicate values and order doesn't matter
            known_border = set([])   
        
            for frontier_square in model:
                frontier_square_adjacents = self.find_adjacents(frontier_square, model, True)
                
                # add each adjacent square to the known_border
                for known_square in frontier_square_adjacents:
                    known_border.add(known_square)

            for known_square in known_border:
                percept = self.map[known_square[0]][known_square[1]]
                
                # now we find the squares in the frontier adjacent to the border square
                # we're going to check their status in the model and compare it to the
                # percept for the known border square
                frontier_squares_adjacent_to_border = self.find_adjacents(known_square, model, False)

                pit_count = 0
                wump_count = 0
                for frontier_square in frontier_squares_adjacent_to_border:
                    if frontier_square[1] == True:
                        pit_count += 1
                    if frontier_square[2] == True:
                        wump_count += 1
                
                # here we check conditions for determining whether the model is correct
                if "breeze" in percept and pit_count == 0:
                    # if there is a breeze in the percepts but none of the 
                    # frontier squares adjacent to the square with the breeze have a pit, 
                    # then we invalidate the model
                    isValid = False
                if "breeze" not in percept and pit_count > 0:
                    isValid = False
                if wump_count > 1:
                    isValid = False
                if "stench" in percept and wump_count == 0:
                    isValid = False
                if "stench" not in percept and wump_count != 0:
                    isValid = False
            
            if isValid == True:
                valid_models.append(model)

        return valid_models

    # calls enumerateModels, checkModels, and probabilityFormula
    # and returns the danger probability for each square in the 
    # frontier
    def calculateProbabilities(self):
        models = enumerateModels(self.frontier)

        validModels = self.checkModels(models)
        
        return self.probabilityFormula(validModels)        # returns probability that a square is dangerous for each square in frontier
    
    # this is the function that actually calculates probabilities
    def probabilityFormula(self, valid_models):
        square_danger = []

        #counts[valid_models[0][0]]
        for i in range(len(valid_models[0])):
            pit_count = 0
            wumpus_count = 0

            for model in valid_models:    
                if model[i][1] == True:
                    pit_count += 1
                if model[i][2] == True:
                    wumpus_count += 1

            # probability that there's a pit in square
            prob_pit = pit_count/len(valid_models) 

            # probability that there's a wumpus in square     
            prob_wump = wumpus_count/len(valid_models) 

            # probability that there is a pit or that there is a wumpus p(A) + p(B) - p(A ^ B)
            prob_danger = prob_pit+prob_wump-(prob_pit*prob_wump) 

            square_danger.append((model[i][0], prob_danger))
        
        # now we sort by probability of danger
        square_danger = sorted(square_danger, key=lambda x: x[1])
        
        return square_danger

    # this function finds the squares adjacent to the current square
    # and appends them to self.frontier
    def updateFrontier(self):
        temp = []

        # first, we must remove values from the frontier if
        # we have visited them already (because then they)
        # are no longer in the frontier
        
        duplicates = [i for i in self.frontier if i in self.known]

        for duplicate in duplicates:
            self.frontier.pop(self.frontier.index(duplicate)) # remove visited squares from frontier 
        
        above = (self.position[0]-1, self.position[1])
        below = (self.position[0]+1, self.position[1])
        left = (self.position[0], self.position[1]-1)
        right = (self.position[0], self.position[1]+1)
        
        # check to see if the dimension are between 
        # 0 and self.max, otherwise, they aren't
        # valid dimensions
        if above[0] >= 0:
            temp.append(above)
        if below[0] < self.max:
            temp.append(below)
        if left[1] >= 0:
            temp.append(left)
        if right[1] < self.max:
            temp.append(right)

        # go through all adjacent squares and check to see if
        # the adjacent squares are not already in the frontier
        # and haven't already been visited
        for square in temp:
            if (square, False, False) not in self.frontier and (square, False, False) not in self.known:
                self.frontier.append((square, False, False))
    
    # this is the function that will pick the next action of
    # the agent
    def action(self): 
        # test for controlled exit at end of successful gui episode
        if self.stopTheAgent:
            print("Agent has won this episode.")
            return 'exit' # will cause the episide to end
            
        #reflect action -- get the gold!
        if 'glitter' in self.percepts:
            print("Agent will grab the gold!")
            self.stopTheAgent=True
            return 'grab'
        
        danger_probabilities = self.calculateProbabilities()
        
        # first, we check to see if any of the best squares are adjaceent to the agent
        isAdjacent = False
        action = ""

        for danger in danger_probabilities:
            adjacents = self.find_adjacents(danger[0], self.frontier, True) # find squares in known that are adjacent to the chosen frontier square
            # if the current square is adjacent to the frontier square and if the frontier square's danger is equal to the minimum danger,
            # then that square is the next square
            if self.position in adjacents and danger[1] <= danger_probabilities[0][1]:
                next_square = danger[0]
                isAdjacent = True
                break
        
        if isAdjacent == False:
            fringe = [[self.position, []]]
            path = findPath(fringe, danger_probabilities[0][0], self.known)  # this function used breadth-first search to find a path to the desired square.
        
            next_square = path[1]
            
        if next_square[0] - self.position[0] == -1:     # next square is above
            if self.facing != "up":
                action = 'left'
                self.calculateNextDirection(action)
            else:
                action = 'move'
                self.calculateNextPosition(action)
        if next_square[0] - self.position[0] == 1:      # next square is below
            if self.facing != "down":
                action = 'right'
                self.calculateNextDirection(action)
            else:
                action = 'move'
                self.calculateNextPosition(action)
        if next_square[1] - self.position[1] == -1:     # next square is left
            if self.facing != "left":
                action = 'left'
                self.calculateNextDirection(action)
            else:
                action = 'move'
                self.calculateNextPosition(action)
        if next_square[1] - self.position[1] == 1:      # next square is right
            if self.facing != "right":
                action = 'right'
                self.calculateNextDirection(action)
            else:
                action = 'move'
                self.calculateNextPosition(action)

        return action    

# this function enumerates all the possibilities of the frontier.
# In other words, it will generate all the possible configurations
# of the squares within the frontier
def enumerateModels(frontier):
    model_list = []

    model_configs = product([True, False], repeat=len(frontier)*2) # this will produce 2^(len(frontier)*2) models

    for model_config in model_configs:  # for each model configuration
        model = []
        pos = 0
        for square in frontier:
            model.append((square[0], model_config[pos], model_config[pos+1])) # append the generated hasPit and hasWumpus
            pos = pos+2                                     # increment pos by 2
        model_list.append(model)                            # append the model to the model list -- each model appended is a possible frontier
    '''

    For a frontier with 2 squares, each element of enums will have 4 configurations slots
    each containing a True or False. For example, a possible configuration is: [False, True, True, True].
    The way to interpret this is that the first 2 numbers correspond to the following 
    values for the first square in frontier (hasPit, hasWumpus) and the last 2
    correspond to the respective values for the second square.
    '''

    return model_list

# finds the successors from each known square
def successor(known, root, goal):
    adjacents_list = []
    square = root

    above = (square[0]-1, square[1])
    below = (square[0]+1, square[1])
    left = (square[0], square[1]-1)
    right = (square[0], square[1]+1)

    if (above, False, False) in known or above == goal:
        adjacents_list.append(above)
    if (below, False, False) in known or below == goal:
        adjacents_list.append(below)
    if (left, False, False) in known or left == goal:
        adjacents_list.append(left)
    if (right, False, False) in known or right == goal:
        adjacents_list.append(right)
    
    return adjacents_list

# performs breadth-first search to find a path through known
# squares to the goal square
def findPath(fringe, goal, known):
    visited = []

    while len(fringe) > 0:
        rootnode = fringe.pop(0)
        print("ROOTNODE", rootnode)
        root = rootnode[0]

        if root == goal:
            return rootnode[1] + [goal]
        
        next_square_list = successor(known, root, goal)
        visited.append(root)
        fringe_square = [i[0] for i in fringe]

        for square in next_square_list:
            
            if not square in fringe_square and not square in visited:
                new_node = [square, rootnode[1] + [root]]
                fringe.append(new_node)
    
    return "NO PATH TO " + str(goal)
