import random
from .genome_registry import GenomeRegistry

class GenomeSampler:
    def __init__(self, registry, seed=None):
        self.registry = registry
        if seed is not None:
            random.seed(seed)

    def sample_sequences(self, num_samples, fragment_length):
        genomes = self.registry.genomes
        samples = []
        for _ in range(num_samples):
            genome = random.choice(genomes)
            start_idx = random.randint(0, len(genome) - fragment_length)
            fragment = genome[start_idx:start_idx + fragment_length]
            samples.append(fragment)
        return samples

    def sample_by_metadata(self, num_samples, criteria):
        genomes = [g for g in self.registry.genomes if self._matches_criteria(g, criteria)]
        samples = random.sample(genomes, num_samples)
        return samples

    def _matches_criteria(self, genome, criteria):
        # Реализуйте проверку соответствия генома заданным критериям
        pass