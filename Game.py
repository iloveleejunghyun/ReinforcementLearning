import numpy as np
import pandas as pd
import random
class Game:
    rewards = None
    positionCol = None
    positionRow = None
    
    def __init__(self, startCol=1, startRow=1):
        self.distance = pd.DataFrame({1:[8,7,6,5,4], 2:[7,6,5,4,3], 3:[6,5,4,3,2], 4:[5,4,3,2,1], 5:[4,3,2,1,0]}, index={1,2,3,4,5})
        self.rewards = pd.DataFrame({1:[0,1,2,3,4], 2:[1,2,3,4,5], 3:[2,3,4,5,6], 4:[3,4,5,6,7], 5:[4,5,6,7,8]}, index={1,2,3,4,5})
        self.positionCol = startCol
        self.positionRow = startRow
        
    def move(self, direction):
        reward = 0
        end = False
        distance_before = self.distance[self.positionCol][self.positionRow]
        if direction=='Up':
            self.positionRow -= 1    
        elif direction=='Down':
            self.positionRow += 1
        elif direction=='Left':
            self.positionCol -= 1  
        else:
            self.positionCol += 1
        
        #check if we lost
        if self.positionRow < 1 or self.positionRow > 5 or self.positionCol < 1 or self.positionCol > 5:
            end = True
            reward = -1000   
        #check if we have reached the end
        elif self.positionCol == 5 and self.positionRow == 5:
            end = True
            reward = self.rewards[self.positionCol][self.positionRow]
        else:
            end = False
            if distance_before < self.distance[self.positionCol][self.positionRow]:
                reward = -1000
            else:
                reward = self.rewards[self.positionCol][self.positionRow]
        
        #return reward and end of game indicator
        return (reward, end)