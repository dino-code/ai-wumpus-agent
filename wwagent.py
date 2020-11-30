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
        self.known = [((3,0), False, False)] # all the squares appended to this list will have values of False (otherwise the game would end).
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
        if self.position[0] in range(self.max) and self.position[1] in range(self.max):
            self.map[ self.position[0]][self.position[1]]=self.percepts
        self.updateFrontier()
        print("-"*40)                                                                                           # ************************
        print("frontier:", self.frontier)
        print("-"*40)
        self.calculateProbabilities()

        # puts the percept at the spot in the map where sensed

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
                self.known.append((self.position, False, False)) # append the new position to known positions

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

    # check generated models using the sensor data.
    # Returns valid models (that will then be used in
    # the calculation of probabilities for pits and 
    # wumpuses)
    def checkModels(self, query_variants, models):
        valid_models_dict = {}  # key=query variant, value=models where this variant is valid
                                # this will be used later when calculating probabilities

        # for each query variant
        for query in query_variants:
            valid_models_dict[query] = []

            query_adjacents_in_known = []

            above_query = (query[0][0]-1, query[0][1])
            below_query = (query[0][0]+1, query[0][1])
            left_query = (query[0][0], query[0][1]-1)
            right_query = (query[0][0], query[0][1]+1)

            if (above_query, False, False) in self.known:
                query_adjacents_in_known.append(above_query)
            if (below_query, False, False) in self.known:
                query_adjacents_in_known.append(below_query)
            if (left_query, False, False) in self.known:
                query_adjacents_in_known.append(left_query)
            if (right_query, False, False) in self.known:
                query_adjacents_in_known.append(right_query)

            # within each model
            for model in models:
                bad = False
                # if ((row, col), True, False) there is a pit and no wumpus -- check that at least one adjacent square is in self.known, has breeze and has no stench
                # if ((row, col), False, True) there is no pit and a wumpus -- check that at least one adjacent square is in self.known, has no breeze and has a stench
                # if ((row, col), False, False) there is no pit and no wumpus -- check that all adjacent squares that are in known have no breeze or stench
                # if ((row, col), True, True) there is a pit and a wumpus -- check that all adjacent squares that are in known have a breeze and a stench
                for square in model:
                    # find all known squares adjacent to a square in the model. If any of these known squares contradicts a model, then the model must be false
                    # for example, if our model says (2, 1) has a pit, but we have visited (2, 0) and seen that there is no breeze, then there is no way that
                    # (2, 1) has a pit
                    
                    above = (square[0][0]-1, square[0][1])
                    below = (square[0][0]+1, square[0][1])
                    left = (square[0][0], square[0][1]-1)
                    right = (square[0][0], square[0][1]+1)

                    model_adjacents_in_known = [] # list of the squares we will have to check for breezes and stenches

                    if (above, False, False) in self.known:
                        model_adjacents_in_known.append(above)
                    if (below, False, False) in self.known:
                        model_adjacents_in_known.append(below)
                    if (left, False, False) in self.known:
                        model_adjacents_in_known.append(left)
                    if (right, False, False) in self.known:
                        model_adjacents_in_known.append(right)
                    
                    # now we iterate through sensor data to find whether there is are breezes or stenches in squares adjacent to the square in question
                    for adjacent in model_adjacents_in_known:
                        percept = self.map[adjacent[0]][adjacent[1]]

                        condition = (square[1] == False and 'breeze' in percept) or (square[1] == True and 'breeze' not in percept) or (square[2] == False and 'stench' in percept) or (square[2] == True and 'stench' not in percept)

                        if condition:
                            # this if statement checks for the case where the model has a pit, but no breeze in an adjacent square or if the 
                            # model has a Wumpus, but no stench in adjacent squares
                            # if this is the case, we invalidate the model
                            bad = True
                    
                    for adjacent in query_adjacents_in_known:
                        percept = self.map[adjacent[0]][adjacent[1]]

                        condition = (query[1] == False and 'breeze' in percept) or (query[1] == True and 'breeze' not in percept) or (query[2] == False and 'stench' in query) or (query[2] == True and 'stench' not in percept)

                        if condition:
                            # this if statement checks for the case where the model has a pit, but no breeze in an adjacent square or if the 
                            # model has a Wumpus, but no stench in adjacent squares
                            # if this is the case, we invalidate the model
                            bad = True
                    
                    # now we do the same for squares adjacent to the query
                    

                    model_adjacents_in_known = [] # list of the squares we will have to check for breezes and stenches

                    if (above, False, False) in self.known:
                        model_adjacents_in_known.append(above)
                    if (below, False, False) in self.known:
                        model_adjacents_in_known.append(below)
                    if (left, False, False) in self.known:
                        model_adjacents_in_known.append(left)
                    if (right, False, False) in self.known:
                        model_adjacents_in_known.append(right)
                    
                    # now we iterate through sensor data to find whether there is are breezes or stenches in squares adjacent to the square in question
                    for adjacent in model_adjacents_in_known:
                        percept = self.map[adjacent[0]][adjacent[1]]

                        if (square[1] == False and 'breeze' in percept) or (square[1] == True and 'breeze' not in percept) or (square[2] == False and 'stench' in percept) or (square[2] == True and 'stench' not in percept):
                            # this if statement checks for the case where the model has a pit, but no breeze in an adjacent square or if the 
                            # model has a Wumpus, but no stench in adjacent squares
                            # if this is the case, we invalidate the model
                            bad = True
                
                if bad == False:
                    valid_models_dict[query].append(model)
        
        return valid_models_dict

    def calculateProbabilities(self):
        '''
        here we need to remember that there are 4 variants of each query
        if ((row, col), True, False) there is a pit and no wumpus -- check that at least one adjacent square is in self.known, has breeze and has no stench
        if ((row, col), True, True) there is a pit and a wumpus -- check that all adjacent squares that are in known have a breeze and a stench
        if ((row, col), False, True) there is no pit and a wumpus -- check that at least one adjacent square is in self.known, has no breeze and has a stench
        if ((row, col), False, False) there is no pit and no wumpus -- check that all adjacent squares that are in known have no breeze or stench
        '''
        fron = copy.deepcopy(self.frontier)
        probabilities = {}      # this dict contains the probability of a pit for each square

        print("Queries and valid models:")
        for query in self.frontier:
            fron.pop(fron.index(query)) # remove the query from the frontier
            query_variants = [(query[0], True, False), (query[0], True, True), (query[0], False, True), (query[0], False, False)] # each of these impacts the probability calculation

            models = enumerateModels(fron)  # model enumeration
            fron.append(query)

            validModels = self.checkModels(query_variants, models)
            print(query)
            for i in validModels:
                print(i, validModels[i])

            probabilities[query[0]] = self.probabilityFormula(query, validModels)
            print("probabilities:", probabilities)

            # now, we evaluate the generated models and leave only the ones
            # that are possible given the observations (model-checking)

    def probabilityFormula(self, query, validModels):
        hasPit = (query[0], True, False)
        noPit = (query[0], False, False)

        probability = {}

        prob = 0
        
        for square in validModels:
            if square[0][1] == True:
                prob += 0.2
            else:
                prob += 0.8
        probability['hasPit'] = prob * 0.2

        prob = 0
        for square in validModels:
            if square[0][1] == True:
                prob += 0.2
            else:
                prob += 0.8
        probability['noPit'] = prob * 0.8

        return (probability['hasPit'], probability['noPit'])
            

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
    # the agent. This is the main function that needs to be
    # modified when you design your new intelligent agent
    # right now it is just a random choice agent
    
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
        
        # choose a random direction, and move          
        actionSelection = randint(0,1)
        if actionSelection>0: # there is an 50% chance of moving forward 
            action = 'move'
            # predict the effect of this
            self.calculateNextPosition(action)

        else: # pick left or right 50%
            actionSelection=randint(0,1)
            if actionSelection>0:
                action = 'left'
            else:
                action='right'
            # predict the effect of this
            self.calculateNextDirection(action)
        print ("Random agent:",action, "-->",self.position[0],
               self.position[1], self.facing)
        return action


# this function enumerates all the possibilities of the frontier.
# In other words, it will generate all the possible configurations
# of the squares within the frontier.
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