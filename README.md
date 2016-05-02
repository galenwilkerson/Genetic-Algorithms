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

Note that there are two fitness functions available:

- Euclidean Distance
- String Distance

The Euclidean distance could work fine, but is highly non-monotonic.  That is, it does not reflect how flose the ball gets to the objective along the maze.  The second compares the move sequence to the solution move sequence, using a string difference comparison algorithm (the Levenshtein distance, which tells the number of simple string edits between two strings.), using the library: https://pypi.python.org/pypi/Distance

GenomePopulation.main_full()  creates a population and runs through generations of mutation and recombination, also printing the best genome strings and histograms of the genome population.  

At the end it displays the resulting maze state when running the best genome.
