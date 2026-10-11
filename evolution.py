"""
evolution.py - the genetic algorithm (no gradients, no labeled data).

Each network is flattened into a vector of 50 numbers: its "DNA"
(network.to_vector()). One generation of evolution:

  1. SELECTION : sort by fitness; the top 25% become parents.
  2. ELITISM   : the best 3 are copied unchanged, so the best driver
                 found so far can never be lost to an unlucky mutation.
  3. CROSSOVER : a child takes each of its 50 numbers from parent A or
                 parent B at random (a coin flip per number: "uniform").
  4. MUTATION  : each number has a 10% chance of a random nudge
                 (normal noise, standard deviation 0.4). Mutation is
                 what creates genuinely new behaviour.
"""
import numpy as np

ELITE = 3
PARENT_FRACTION = 0.25
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.4


def mutate(dna, rng, rate=MUTATION_RATE, strength=MUTATION_STRENGTH):
    """Return a NEW vector; each number nudged with probability `rate`."""
    mask = rng.random(dna.shape) < rate
    return dna + mask * rng.normal(0, strength, dna.shape)


def crossover(a, b, rng):
    """Uniform crossover: for each position, take the number from a or b."""
    pick_a = rng.random(a.shape) < 0.5
    return np.where(pick_a, a, b)


def next_generation(dnas, fitnesses, rng, elite=ELITE,
                    parent_fraction=PARENT_FRACTION,
                    rate=MUTATION_RATE, strength=MUTATION_STRENGTH):
    """dnas: list of vectors. fitnesses: one score per vector (higher = better).
    Returns a new list of the same size."""
    order = np.argsort(fitnesses)[::-1]               # best first
    ranked = [dnas[i] for i in order]
    n_parents = max(2, int(len(dnas) * parent_fraction))
    parents = ranked[:n_parents]

    children = [d.copy() for d in ranked[:elite]]     # elitism
    while len(children) < len(dnas):
        i, j = rng.choice(n_parents, size=2, replace=False)
        child = crossover(parents[i], parents[j], rng)
        children.append(mutate(child, rng, rate, strength))
    return children