# Genetic-Algorithms
An exploration of genetic algorithms using Python and a labyrinth

The "labyrinth" is a simple ball-maze that has 4 directions:  up, down, left, right

These directions can be seen as codons (U, D, L, R) for a 'genome' that contains a sequence of moves,
for example:
RRDDLLDDRRRU

We can then run them in a constant (same walls, size, and goal location) maze to see how close the ball get to the goal.

We then find the "fitness" by the distance to the goal location.
(Note here that, 0 = "best" and 15 -- the number of empty spots in the maze -- is "worst".)

Now, we can create a population of "Genomes" and mutate, recombine them over many generations to see how they do.



