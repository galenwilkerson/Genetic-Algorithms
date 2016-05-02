#!/usr/bin/python3

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:39:22 2016

@author: galen
"""

import random as rnd
import numpy as np

# the maze
import labyrinth

# for sorting
import operator

import pandas as pd


class GenomePopulation:

    #    The population of genomes (a list of lists)
    
    # the genome population, just a list of genomes
 
    #                 # the genome population, just a list of genomes
    #        self.genomePopulation = []
    #        self.populationSize = populationSize
    #        
    #        self.genomeLength = genomeLength
    #
    #        self.codons = set(alphabet)
 
    # we could make a Genome class, 
    # but since this is just a string (for now) 
    # let's not make things too complicated)
 
        
    def __init__(self, populationSize = 10, alphabet = set(["L", "R", "U", "D"]), 
                                                           genomeLength = 16):

        # a dictionary of genome strings and fitness values
        self.genomePopAndFitness = {}
        
        self.populationSize = populationSize        
        self.genomeLength = genomeLength
        
        # here I just mean one instruction, not 3 as in biological codons
        self.codons = set(alphabet)
    
        for i in range(self.populationSize):
            g1 = self.createRandomGenome(self.genomeLength)
            self.genomePopAndFitness["".join(g1)] = -1
        

    # create a random genome of length 
    def createRandomGenome(self, length = 15):
        
        # turn the set of possible codons to a list for random choice
        codonList = list(self.codons)
        geneData = []
        
        for i in range(length):
            geneData.append(rnd.choice(codonList))

        return geneData
        

   
    def mutateGenome(self, genome, p = 0.5):
        '''
        randomly (uniform, P < p) mutate each codon in a genome to a random value
        '''
        
        # turn the set of possible codons to a list for random choice
        codonList = list(self.codons)

        # prepare the output list for no changes
        outGenome =  list(genome)
            
        for i in range(len(genome)):            
            
            P = rnd.random()            
            
            if(P < p):
                outGenome[i] = rnd.choice(codonList)
                
        return("".join(outGenome))
        
        
        
    def mutateAllGenomes(self, p = 0.5):
        '''
        mutate each genome to produce a new genome
        NOTE: MAY RESULT IN A POPULATION LARGER THAN DESIRED MAXIMUM POPULATION
        '''
        
        genomes = list(self.genomePopAndFitness.keys())

        for genome in genomes:

            # mutate and add to genome dictionary, with fitness of -1
            mutatedGenome = self.mutateGenome(genome, p = 0.5)
            self.genomePopAndFitness[mutatedGenome] = -1

                             
                             
    def frameShiftMutateGenome(self, genome):
        '''
        input genome as string or list
        randomly swap values of two codons in a genome 
        (here I just mean one instruction, not 3 as in biological codons)
        return mutated genome as string
        '''
        genomeAsList = list(genome)

        i = rnd.randint(0, len(genomeAsList) - 1)
        j = rnd.randint(0, len(genomeAsList) - 1)
        temp = genomeAsList[i]
        genomeAsList[i] = genomeAsList[j]
        genomeAsList[j] = temp
        
        return("".join(genomeAsList))    

    

    def frameShiftMutateAllGenomes(self):
        '''
        frame shift mutate each genome to produce a new genome
        NOTE: MAY RESULT IN A POPULATION LARGER THAN DESIRED MAXIMUM POPULATION
        '''
        
        genomes = list(self.genomePopAndFitness.keys())

        for genome in genomes:

            # mutate and add to genome dictionary, with fitness of -1
            mutatedGenome = self.frameShiftMutateGenome(genome)
            self.genomePopAndFitness[mutatedGenome] = -1    
    
    
    
    def recombineGenomePair(self, inGenome1, inGenome2, p = 0.5):
        '''
        iterate down inGenome1 and inGenome2 (must have same length and same codon alphabet)
        at each step, randomly (uniform, 50%) choose one of the two codons
        construct a new genome from the choices
        '''
        
        outGenome = []
        
        gene1 = list(inGenome1)
        gene2 = list(inGenome2)
        
        for i in range(len(gene1)):
            
            P = rnd.random()            
            if(P < p):
                outGenome.append(gene1[i])
            else:
                outGenome.append(gene2[i])
                
        return("".join(outGenome))


    def recombineAllGenomes(self, p = 0.5):
        '''
        assumes an even number of genomes
        pair off all genomes, recombine them to create "offspring", add to gene pool
        NOTE: MAY RESULT IN A POPULATION LARGER THAN DESIRED MAXIMUM POPULATION
        '''
                
        # make a local copy of genomes
        genomePopAndFitness = self.genomePopAndFitness.copy()
        genomeList = list(genomePopAndFitness.keys())
        
        randList = list(np.random.permutation(genomeList))

        # mate all pairs, if length odd, mate the last with itself (eww)
        for i in range(0, len(randList), 2):
            geneA = genomeList[i]
            geneB = genomeList[i+1]
            outGenome = self.recombineGenomePair(geneA, geneB, p)
            #print("recombining: ", geneA, geneB, "result: ", outGenome)
            genomePopAndFitness[outGenome] = -1
     #   print("new genomes")
     #   print(genomePopAndFitness)
        self.genomePopAndFitness = genomePopAndFitness.copy()
     #   print()

    def getFitness(self, maze, moveSequence):
        '''
        Given a maze and a moveSequence string containing valid moves, return a fitness.
        Lower fitness is better, with zero being best.
        '''

        moveSequenceList = list(moveSequence)
        
        # start the ball from the beginning
        maze.resetBall()
        maze.runMoveSequence(moveSequenceList)

        # hmm, perhaps as users we shouldn't know what distance function is being used...            

#        fitness = maze.euclideanDistance()
        fitness = maze.levenshteinDistance(moveSequence)
        
        return fitness
        
                
                
    def drawStateThisGenome(self, maze, moveSequence):
        '''
        Given a maze and a moveSequence string containing valid moves, 
        draw the state of the resulting maze (ball location and goal location)
        '''

        moveSequenceList = list(moveSequence)
        
        # start the ball from the beginning
        maze.resetBall()
        maze.runMoveSequence(moveSequenceList)
        maze.drawState(moveSequence)

        


    def setFitnessAllGenomes(self, maze):
        '''
        given a maze, 
        set the fitness for each genome in gene pool
        '''

        #- get the fitness of each genome
        for moveSequence in self.genomePopAndFitness.keys():
            
            fitness = self.getFitness(maze, moveSequence)

            self.genomePopAndFitness[moveSequence] = fitness
            
        
        
    # return a list sorted by fitness values    
    def sortByFitness(self):
        
        # sort by second element of the tuple (the fitness)
        sortedGenepoolbyFitness = sorted(self.genomePopAndFitness.items(), key=operator.itemgetter(1))    
        return sortedGenepoolbyFitness


    def fitnessHistogram(self):
        '''
        draw a histogram of the fitness levels
        '''
        
        df = pd.DataFrame.from_dict(self.genomePopAndFitness, orient='index')
        df.hist(bins = 30)


    def cullBestFraction(self, sortedGenepoolbyFitness, fraction):
        '''
        Keep only the top fraction of the total population.  
        Set the new genome pool and fitness.
        '''

        bestPopulation = sortedGenepoolbyFitness[0:int(len(sortedGenepoolbyFitness)*fraction)]

        genomePopAndFitness = {}
        
        # insert all new items into new dictionary
        for item in bestPopulation:
            genomePopAndFitness[item[0]] = item[1]

        self.genomePopAndFitness = genomePopAndFitness


    def cullBestSize(self, sortedGenepoolbyFitness, populationSize):
        '''
        Keep only the top of the total population, resetting to populationSize
        Set the new genome pool and fitness.
        '''

        bestPopulation = sortedGenepoolbyFitness[0:(populationSize)]

        genomePopAndFitness = {}
        
        # insert all new items into new dictionary
        for item in bestPopulation:
            genomePopAndFitness[item[0]] = item[1]

        self.genomePopAndFitness = genomePopAndFitness

    def removeDuplicates(self):
        '''
        remove duplicate keys from genomePopAndFitness
        assumes fitness is identical for identical genome strings
        '''
        # get the set of keys
        keySet = set(self.genomePopAndFitness.keys())

        uniqueDict = {}        
        for key in list(keySet):
            uniqueDict[key] = self.genomePopAndFitness[key]
        self.genomePopAndFitness = uniqueDict



    def mutationStep(self, mutationProbability, maze, populationSize):
        '''
        one mutation step:
        
        - mutate
        - find fitness
        - remove duplicates
        - sort by fitness
        - resize to populationsize        
        '''        
        
        self.mutateAllGenomes(mutationProbability)

        self.setFitnessAllGenomes(maze)
        
        self.removeDuplicates()

        sortedGenepoolbyFitness = self.sortByFitness()   
    
        self.cullBestSize(sortedGenepoolbyFitness, populationSize)



    def frameShiftMutationStep(self, maze, populationSize):
        '''
        one frame shift mutation step:
        
        - mutate
        - find fitness
        - remove duplicates
        - sort by fitness
        - resize to populationsize        
        '''        
        
        self.frameShiftMutateAllGenomes()

        self.setFitnessAllGenomes(maze)
        
        self.removeDuplicates()

        sortedGenepoolbyFitness = self.sortByFitness()   
    
        self.cullBestSize(sortedGenepoolbyFitness, populationSize)
        
        

    def recombinationStep(self, p, maze, populationSize):
        '''
        one recombination step:
        
        - recombine
        - find fitness
        - remove duplicates
        - sort by fitness
        - resize to populationsize  
        '''
        
        self.recombineAllGenomes(p)
 
        self.setFitnessAllGenomes(maze)
        
        self.removeDuplicates()

        sortedGenepoolbyFitness = self.sortByFitness()   
   
        self.cullBestSize(sortedGenepoolbyFitness, populationSize)



    # print out the genomes    
    def printGenes(self):
        print(len(self.genomePopAndFitness.keys()), "genomes of length", self.genomeLength)
        for key in self.genomePopAndFitness.keys():
            print(key)

   


def main_small():
    '''
    simple basic tester
    create a gene pool and maze
    '''
    
   
    populationSize = 10
    
    # create a population        
    genePool = GenomePopulation(populationSize)

    print("Starting genepool")    
    genePool.printGenes()
    
    numGenerations = 10

    # create a labyrinth
    maze = labyrinth.Labyrinth()
    maze.drawState()    
    
    
def main_fitness_only():
    '''
    create a pool of genes
    create maze
    
    sort the genes by fitness
    '''    
    
    populationSize = 10
    
    # create a population        
    genePool = GenomePopulation(populationSize)

    print("Starting genepool")    
    genePool.printGenes()
    

    # create a labyrinth
    maze = labyrinth.Labyrinth()
    maze.drawState()    

    # print fitnesses 
    print("Fitness:")
    print(genePool.genomePopAndFitness)
    print
    print("Setting fitness")
    #- set the fitness of each genome
    genePool.setFitnessAllGenomes(maze)
    
    # print fitnesses 
    print("Fitness:")
    print(genePool.genomePopAndFitness)
    print
    
    # sort genepool by fitness
    print("Sorted genepool (tuples): ")
    sortedGenepoolbyFitness = genePool.sortByFitness()   
    print(sortedGenepoolbyFitness)

    return(genePool, sortedGenepoolbyFitness)

def main_cull():
    '''
    same as all above, this time remove the less fit half of population
    
    first run above main, get genepool, sort by fitness
    '''

    (genePool, sortedGenepoolbyFitness) = main_fitness_only()
        
    print("main cull")
    print(sortedGenepoolbyFitness)

    print("culling")
    fraction = 0.5
    genePool.cullBestFraction(sortedGenepoolbyFitness, fraction)

    print(genePool.genomePopAndFitness)
      
    return(genePool)
    
    
def main_mutate():
    '''
    same as main_cull() above
    
    now mutate the remaining genomes to fill out the population
    
    find their fitness values
    
    print fitness
    '''    
    
    genePool = main_cull()
    
    populationSize = 10
    
    # probability of mutating each codon
    mutationProbability = 0.5
    
    print("mutating")
    
    # mutate each each codon with probability 1/2, fill out to populationSize
    genePool.mutateAllGenomes(mutationProbability)
    
    print(genePool.genomePopAndFitness)

    print("Setting fitness")
    
    # create a labyrinth
    maze = labyrinth.Labyrinth()
    
    #- set the fitness of each genome
    genePool.setFitnessAllGenomes(maze)
    
    # print fitnesses 
    print("Fitness:")
    print(genePool.genomePopAndFitness)
    print
    
    # sort genepool by fitness
    print("Sorted genepool (tuples): ")
    sortedGenepoolbyFitness = genePool.sortByFitness()   
    print(sortedGenepoolbyFitness)
    
    genePool.cullBestSize(sortedGenepoolbyFitness, populationSize)
    
    # return new genepool
    return(genePool, sortedGenepoolbyFitness)
    
    
def main_recombine():
    '''
        same as main_mutate() above
        
        now recombine the genomes to fill out the population
        
        find their fitness values
        
        print fitness
    '''    
    
    (genePool, sortedGenepoolbyFitness) = main_mutate()
    print(genePool.genomePopAndFitness)

    populationSize = 10
    
    print("Recombining")
    # the mutation bias, toward one genome or the other
    p = 0.5
    
    genePool.recombineAllGenomes(p)

    print(genePool.genomePopAndFitness)

    # create a labyrinth
    maze = labyrinth.Labyrinth()
    
    #- set the fitness of each genome
    genePool.setFitnessAllGenomes(maze)

    # sort genepool by fitness
    print("Sorted genepool (tuples): ")
    sortedGenepoolbyFitness = genePool.sortByFitness()   
    print(sortedGenepoolbyFitness)
    
    # resize to populationSize
    genePool.cullBestSize(sortedGenepoolbyFitness, populationSize)
    print(genePool.genomePopAndFitness)


def main_mutate_generations():
    '''
    try a few generations of mutation
    
    create a population
    
A:  while(fitness > 0 AND num_generations < 10):  
    
    find the fitness
    
    sort by fitness

    remove 1/2, 
    
    mutate
    
    resize to populationSize
    
    find the fitness
    
    sort by fitness

    '''

    populationSize = 1000
    numGenerations = 12
    
    # create a population        
    genePool = GenomePopulation(populationSize)

    print("Starting genepool")    
#    genePool.printGenes()
    

    # create a labyrinth
    maze = labyrinth.Labyrinth()
    maze.drawState()    

    # print fitnesses 
    print("Fitness:")
#    print(genePool.genomePopAndFitness)
    print
    print("Setting fitness")
    #- set the fitness of each genome
    genePool.setFitnessAllGenomes(maze)
    
    # print fitnesses 
    print("Fitness:")
#    print(genePool.genomePopAndFitness)
    print
    
    # sort genepool by fitness
    print("Sorted genepool (tuples): ")
    sortedGenepoolbyFitness = genePool.sortByFitness()   
#    print(sortedGenepoolbyFitness)

    bestFitness = sortedGenepoolbyFitness[0][1]

    generationNum = 0
    
    while(bestFitness > 0 and generationNum < numGenerations):
        
        # probability of mutating each codon
        mutationProbability = 0.9
        
        print("mutating")
        
        # mutate each each codon with probability 1/2, fill out to populationSize
        genePool.mutationStep(mutationProbability, maze, populationSize)
     
        bestFitness = sortedGenepoolbyFitness[0][1]
        bestGenome = sortedGenepoolbyFitness[0][0]
        
        print("Generation: ", generationNum, "best genome", bestGenome, "best fitness: ", bestFitness)
        generationNum += 1
        
        genePool.fitnessHistogram()
        
        


def main_full():
    '''
    finally do everything:
    
    create a population
    
A:  while(fitness > 0):  
    
    find the fitness
    
    sort by fitness

    remove 1/2, 
    
    mutate (in place)
    
    resize to populationSize

    recombine

    resize to populationSize
    '''

    ######  GLOBAL PARAMETERS

    populationSize = 1000
    numGenerations = 12
    
    # probability of mutating each codon
    mutationProbability = 0.9

    # the mutation bias, toward one genome or the other
    mutationBias = 0.5


    #######################

        
    # create a population        
    genePool = GenomePopulation(populationSize)

    print("Starting genepool")    
#    genePool.printGenes()
    

    # create a labyrinth
    maze = labyrinth.Labyrinth()
    maze.drawState()    

    # print fitnesses 
    print("Fitness:")
#    print(genePool.genomePopAndFitness)
    print
    print("Setting fitness")
    #- set the fitness of each genome
    genePool.setFitnessAllGenomes(maze)
    
    # print fitnesses 
    print("Fitness:")
#    print(genePool.genomePopAndFitness)
    print
    
    # sort genepool by fitness
    print("Sorted genepool (tuples): ")
    sortedGenepoolbyFitness = genePool.sortByFitness()   
#    print(sortedGenepoolbyFitness)

    bestFitness = sortedGenepoolbyFitness[0][1]

    generationNum = 0
    
    while(bestFitness > 0 and generationNum < numGenerations):
        
        print("mutating")
        
        # mutate each each codon with probability 1/2, fill out to populationSize
        genePool.mutationStep(mutationProbability, maze, populationSize)
     
        print("Recombining")
        
        
        
        genePool.recombinationStep(mutationBias, maze, populationSize)
        
        # sort genepool by fitness
#        print("Sorted genepool (tuples): ")
        
         
#        print(sortedGenepoolbyFitness)
    
        # print the best of the gene pool
        sortedGenepoolbyFitness = genePool.sortByFitness()  
        bestFitness = sortedGenepoolbyFitness[0][1]
        bestGenome = sortedGenepoolbyFitness[0][0]
        print("best 5 genomes:")
        print(sortedGenepoolbyFitness[:5])
        
        print("Generation: ", generationNum, "best genome", bestGenome, "best fitness: ", bestFitness)
        generationNum += 1

        genePool.fitnessHistogram()
        
        # create a labyrinth
        maze = labyrinth.Labyrinth()
        genePool.drawStateThisGenome(maze, "moves: " + bestGenome)
        
        
        # draw the state of the maze using the best genome



def main_full_frame_shift_mutate():
    '''
    finally do everything:
    
    create a population
    
A:  while(fitness > 0):  
    
    find the fitness
    
    sort by fitness

    remove 1/2, 
    
    mutate (frame shift)
    
    resize to populationSize

    recombine

    resize to populationSize
    '''

    ######  GLOBAL PARAMETERS

    populationSize = 1000
    numGenerations = 12
    
    # probability of mutating each codon
    mutationProbability = 0.9

    # the mutation bias, toward one genome or the other
    mutationBias = 0.5


    #######################

        
    # create a population        
    genePool = GenomePopulation(populationSize)

    print("Starting genepool")    
#    genePool.printGenes()
    

    # create a labyrinth
    maze = labyrinth.Labyrinth()
    maze.drawState()    

    # print fitnesses 
    print("Fitness:")
#    print(genePool.genomePopAndFitness)
    print
    print("Setting fitness")
    #- set the fitness of each genome
    genePool.setFitnessAllGenomes(maze)
    
    # print fitnesses 
    print("Fitness:")
#    print(genePool.genomePopAndFitness)
    print
    
    # sort genepool by fitness
    print("Sorted genepool (tuples): ")
    sortedGenepoolbyFitness = genePool.sortByFitness()   
#    print(sortedGenepoolbyFitness)

    bestFitness = sortedGenepoolbyFitness[0][1]

    generationNum = 0
    
    while(bestFitness > 0 and generationNum < numGenerations):
        
        print("mutating")
        
        # mutate each each codon with probability 1/2, fill out to populationSize
        genePool.frameShiftMutationStep(maze, populationSize)
     
        print("Recombining")
        
        
        
        genePool.recombinationStep(mutationBias, maze, populationSize)
        
        # sort genepool by fitness
#        print("Sorted genepool (tuples): ")
        
         
#        print(sortedGenepoolbyFitness)
    
        # print the best of the gene pool
        sortedGenepoolbyFitness = genePool.sortByFitness()  
        bestFitness = sortedGenepoolbyFitness[0][1]
        bestGenome = sortedGenepoolbyFitness[0][0]
        print("best 5 genomes:")
        print(sortedGenepoolbyFitness[:5])
        
        print("Generation: ", generationNum, "best genome", bestGenome, "best fitness: ", bestFitness)
        generationNum += 1

        genePool.fitnessHistogram()
        
        # create a labyrinth
        maze = labyrinth.Labyrinth()
        genePool.drawStateThisGenome(maze, "moves: " + bestGenome)
        
        
        # draw the state of the maze using the best genome

def main_both_kinds_mutate_and_recomb():
    '''
    finally do everything:
    
    create a population
    
A:  while(fitness > 0):  
    
    find the fitness
    
    sort by fitness

    remove 1/2, 
    
    mutate (frame shift)
    
    resize to populationSize

    mutate (in place)
    
    resize to populationSize
    
    recombine

    resize to populationSize
    '''

    ######  GLOBAL PARAMETERS

    populationSize = 1000
    numGenerations = 15
    
    
    # probability of mutating each codon
    mutationProbability = 0.9

    # the mutation bias, toward one genome or the other
    mutationBias = 0.5


    #######################

        
    # create a population        
    genePool = GenomePopulation(populationSize)

    print("Starting genepool")    
#    genePool.printGenes()
    

    # create a labyrinth
    maze = labyrinth.Labyrinth()
    maze.drawState()    

    # print fitnesses 
    print("Fitness:")
#    print(genePool.genomePopAndFitness)
    print
    print("Setting fitness")
    #- set the fitness of each genome
    genePool.setFitnessAllGenomes(maze)
    
    # print fitnesses 
    print("Fitness:")
#    print(genePool.genomePopAndFitness)
    print
    
    # sort genepool by fitness
    print("Sorted genepool (tuples): ")
    sortedGenepoolbyFitness = genePool.sortByFitness()   
#    print(sortedGenepoolbyFitness)

    bestFitness = sortedGenepoolbyFitness[0][1]

    generationNum = 0
    
    while(bestFitness > 0 and generationNum < numGenerations):
        
        print("frame shift mutating")
        
        # mutate each each codon with probability 1/2, fill out to populationSize
        genePool.frameShiftMutationStep(maze, populationSize)
     
     
        print("in place mutating")
        
        # mutate each each codon with probability 1/2, fill out to populationSize
        genePool.mutationStep(mutationProbability, maze, populationSize)
      
     
        print("Recombining")
        
        
        
        genePool.recombinationStep(mutationBias, maze, populationSize)
        
        # sort genepool by fitness
#        print("Sorted genepool (tuples): ")
        
         
#        print(sortedGenepoolbyFitness)
    
        # print the best of the gene pool
        sortedGenepoolbyFitness = genePool.sortByFitness()  
        bestFitness = sortedGenepoolbyFitness[0][1]
        bestGenome = sortedGenepoolbyFitness[0][0]
        print("best 5 genomes:")
        print(sortedGenepoolbyFitness[:5])
        
        print("Generation: ", generationNum, "best genome", bestGenome, "best fitness: ", bestFitness)
        generationNum += 1

        genePool.fitnessHistogram()
        
        # create a labyrinth
        maze = labyrinth.Labyrinth()
        genePool.drawStateThisGenome(maze, "moves: " + bestGenome)
        
        
        # draw the state of the maze using the best genome



if __name__ == '__main__':
    #main_small()
    #main_fitness_only()
    #main_cull()
    #main_mutate()
    #main_recombine()
    #main_mutate_generations()
    #main_full_in_place_mutate()
    #main_full_frame_shift_mutate()
    main_both_kinds_mutate_and_recomb()
    