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
        self.frontier = [((3, 1), 0, 0), ((2, 0), 0, 0)] # ((row, col), hasPit, hasWumpus). The 2 default frontier elements will always be the case
        self.known = [((3,0), 0, 0)] # all the squares appended to this list will have values of 0 (otherwise the game would end).
        self.percepts = (None, None, None, None, None)
        self.map = [[ self.percepts for i in range(self.max) ] for j in range(self.max)]
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
        
        if (self.position, 0, 0) not in self.known:
                self.known.append((self.position, 0, 0)) # append the new position to known positions

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

        above = (self.position[0]+1, self.position[1])
        below = (self.position[0]-1, self.position[1])
        left = (self.position[0], self.position[1]-1)
        right = (self.position[0], self.position[1]+1)
        
        # check to see if the dimension are between 
        # 0 and self.max, otherwise, they aren't
        # valid dimensions
        if above[0] < self.max:
            temp.append(above)
        if below[0] > -1:
            temp.append(below)
        if left[1] > -1:
            temp.append(left)
        if right[1] < self.max:
            temp.append(right)

        # go through all adjacent squares and check to see if
        # the adjacent squares are not already in the frontier
        # and haven't already been visited
        for square in temp:
            if (square, 0, 0) not in self.frontier and (square, 0, 0) not in self.known:
                self.frontier.append((square, 0, 0))

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
        
        # the frontier is updated here and then we check
        # what the best move should be based on the new
        # frontier
        self.updateFrontier()
        
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
