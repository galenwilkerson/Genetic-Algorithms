# -*- coding: utf-8 -*-
#!/usr/bin/python3

"""
Created on Mon Apr 11 15:30:20 2016

@author: galen
@email: gjwilkerson@gmail.com

Represent a ball-maze / labyrinth in very simple format.

Useful for various kinds of testing.
"""

import math
import matplotlib.pyplot as plt
import numpy as np

import distance

class Labyrinth:

    '''
    Data members:
    
    The maze:
    0's for open space.
    1's for walls,
    
    Instruction alphabet:
    L = Left
    R = Right
    U = Up
    D = Down
    '''

    def __init__(self):
        '''
        initialize the maze, ball location, and goal location to pre-set values
        
        here, the maze is just hard-coded as you see below
        
        also initialize the move instructions
        
        keep track of the number of moves
        '''
    
        # the default ball maze
        self.ballMaze = [[0, 0, 0, 1, 0],
                         [1, 1, 0, 1, 0],
                         [0, 0, 0, 1, 0],
                         [0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1]]            
    
        self.numRows = 5
        self.numCols = 5
    
        # goal location
        self.goalLocation = (0, 4)
    
        # the current ball location
        self.ballLocation = (0, 0)
    
        # the alphabet of instructions
        self.moveAlphabet = set(["L", "R", "U", "D"])

        self.winningSequence = "RRDDLLDDRRRURUUU"


        # count total number of moves and move attempts (into walls etc)
        self.totalNumberOfMoves = 0
    
    def createMaze(self,inputMaze):
        '''
        create a maze from an input list of 0's 1's and 2's
        '''
        
        self.ballMaze = inputMaze    
        
        
    def createRandomMaze(self, rows, cols):
        '''
        create a random maze of size rows, cols
        '''        
        pass
     
    
    def resetBall(self):
        '''
        reset ball to (0,0) and total move count to 0
        '''
        self.ballLocation = (0, 0)
        self.totalNumberOfMoves = 0
    
    def setBallLocation(self, row, col):
        '''
        place the ball in the maze
        '''        
    
        if(self.ballMaze[row][col] == 1):               
            return
            #throw("Invalid ball location")
        else:
            self.ballLocation = (row, col)

    def getMoveAlphabet(self):
        '''
        just return the move instruction set
        '''
        return self.moveAlphabet

    def getDimensions(self):
        '''
        return the dimensions of the entire maze
        '''
        return (len(self.ballMaze), len(self.ballMaze[0]))
        
    def getMaxNumberOfMoves(self):
        '''
        return the number of 0's (open space)
        '''
        (M, N) = self.getDimensions()
        
        totalCount = 0
        for i in range(M):
           totalCount += (self.ballMaze[i]).count(0)
           
        return(totalCount)

        
    def printState(self):
        '''
        print the state of the maze, B for ball and G for goal
        '''
        
        for i in range(len(self.ballMaze)):
            for j in range(len(self.ballMaze[0])):
                if ((i,j) == self.ballLocation):
                    print("B", end = ' ')
                    continue
                elif ((i,j) == self.goalLocation):
                    print("G", end = ' ')
                    continue
                else:
                    print(self.ballMaze[i][j], end = ' ')
            print(end = '\n')
        print(end = '\n')


    def drawState(self, title = ""):
        '''
        Draw the state of the maze!        
        
        walls as black squares
        ball as a black circle
        goal as a red circle
        '''
        plt.figure()
        mazeImage = np.array(self.ballMaze)
        mazeImage[self.ballLocation] = 2        
        mazeImage[self.goalLocation] = 3
        plt.title(title)
        plt.imshow(mazeImage, interpolation = 'none')
        plt.show()
                       
                    
    # move the ball one step in the direction of motion from the alphabet
    # return the new location
    def runMove(self, moveDirection):
        '''
        move the ball one step in the direction of motion from the alphabet
        return the new location
        '''
    
        # get the ball location row and column
        (row, col) = self.ballLocation
        newRow = row
        newCol = col
    
        # check for edge of maze and walls in the move direction        

        # check left edge and no wall to left
        if(moveDirection == "L"):
            if(col > 0 and self.ballMaze[row][col - 1] != 1):
                newCol -= 1                
                
        # check right edge and no wall to right        
        elif(moveDirection == "R"):
#            if(col < len(self.ballMaze[:][0]) and self.ballMaze[row][col + 1] != 1):
#                newCol += 1
            if(col < (self.numCols - 1) and self.ballMaze[row][col + 1] != 1):
                newCol += 1
        
        # check top edge and no wall above        
        elif(moveDirection == "U"):
            if(row != 0 and self.ballMaze[row - 1][col] != 1):
                newRow -= 1
        
        # check bottom edge and no wall below        
        elif(moveDirection == "D"):
#            if(row < len(self.ballMaze[0][:]) and self.ballMaze[row + 1][col] != 1):
            if(row < (self.numRows - 1) and self.ballMaze[row + 1][col] != 1):
                newRow += 1                
    
        # make sure the old location is still zero
        self.ballMaze[row][col] = 0
        
        # set the new ball location
        self.ballLocation = (newRow, newCol)

        # count all moves and attempted moves
        self.totalNumberOfMoves += 1

    
    def runMoveSequence(self, movesList):
        '''
        move the ball according to the move instructions
        input:  a list of moves from the move alphabet
        '''
        
        for move in movesList:
            self.runMove(move)        
    
    def reachedGoal(self):
        '''
        check if we've reached the goal
        '''
        
        if (self.ballLocation == self.goalLocation):
            return True
        else:
            return False

    def euclideanDistance(self):
        '''
        return the Euclidean distance between the ball and goal
        '''
        
        (ballRow, ballCol) = self.ballLocation
        (goalRow, goalCol) = self.goalLocation

        distance = float(math.sqrt((goalRow - ballRow)**2 + (goalCol - ballCol)**2))
        return distance


    def levenshteinDistance(self, moveSequence):    
        '''
        return string distance
        '''
        
        return(distance.levenshtein(self.winningSequence, moveSequence))        
        
    
    def createMaze(self, rows, cols):
        '''
        create a maze of a certain size.  
        start in one corner
        make the goal in the opposite corner
 
        - move forward
        - go left
        - go right
        always draw wall(s) on each side of motion
        if hit a wall, backtrack        
        
        smaller chance:
        - split
        '''
    
    
def main():
    '''
    tester:
    create a maze
    try moving the ball
    print and draw the maze
    '''

    maze = Labyrinth()

    # print the state, including ball (3) and goal (2) locations
    print("Maze State: ")
    maze.printState()
    print("Distance: " + str(maze.euclideanDistance()), end = '\n\n')
    
    maze.drawState()    
    
    # try a move that hits a wall
    move = "D"
    print("Move: " + move)
    maze.runMove(move)
    print("State: ")
    maze.printState()

    # try a move that goes off the edge
    move = "U"
    print("Move: " + move)
    maze.runMove(move)
    print("State: ")
    maze.printState()
    
    # try a move that moves one right
    move = "R"
    print("Move: " + move)
    maze.runMove(move)
    print("State: ")
    maze.printState()
    
    print("Distance: " + str(maze.euclideanDistance()), end = '\n\n')

    print(maze.getDimensions())
    print(maze.getMaxNumberOfMoves())
    
     # try a move that moves one right again
    move = "R"
    print("Move: " + move)
    maze.runMove(move)
    print("State: ")
    maze.printState()
    maze.drawState()
     # try a move that moves one right
    move = "R"
    print("Move: " + move)
    maze.runMove(move)
    print("State: ")
    maze.printState()
    maze.drawState()
    # try a move sequence
    moveSequence = ["D","D","L"]
    print("Sequence: ", "".join(moveSequence))
    maze.runMoveSequence(moveSequence)
    print("State: ")
    maze.printState()
    maze.drawState()
    print("Distance: " + str(maze.euclideanDistance()), end = '\n')
    print("Reached goal? " + str(maze.reachedGoal()), end = '\n')
    print
    
    # reset the maze (we'll need this later)
    print("Resetting maze")
    maze.resetBall()
    print("State: ")
    maze.printState()
    maze.drawState()
    
    # try another sequence
    moveString = "RRDDLLDDRR"
    print("another try: ", moveString)
    moveSequence = list(moveString)
    maze.runMoveSequence(moveSequence)
    print("State: ")
    maze.printState()
    print("Distance: " + str(maze.euclideanDistance()), end = '\n')
    print("Reached goal? " + str(maze.reachedGoal()))
    print("Ball location: ", maze.ballLocation)
    print("Goal location: ", maze.goalLocation)

    maze.drawState()


    # try another sequence
    moveString = "RRDDLLDDRRRU"
    print("another try: ", moveString)
    moveSequence = list(moveString)
    maze.runMoveSequence(moveSequence)
    print("State: ")
    maze.printState()
    print("Distance: " + str(maze.euclideanDistance()), end = '\n')
    print("Reached goal? " + str(maze.reachedGoal()))
    print("Ball location: ", maze.ballLocation)
    print("Goal location: ", maze.goalLocation)

    maze.drawState()
    

    # try the winning sequence
    moveString = "RRDDLLDDRRRURUUU"
    print("one last try: ", moveString)
    moveSequence = list(moveString)
    maze.runMoveSequence(moveSequence)
    print("State: ")
    maze.printState()
    print("Distance: " + str(maze.euclideanDistance()), end = '\n')
    print("Reached goal? " + str(maze.reachedGoal()))
    print("Ball location: ", maze.ballLocation)
    print("Goal location: ", maze.goalLocation)

    maze.drawState()

if __name__ == '__main__':
    main()

