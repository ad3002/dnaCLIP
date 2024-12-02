import json
from pathlib import Path

class GenomeRegistry:
    def __init__(self, config_path):
        self.config_path = config_path
        self.genomes = {}
        self.load_registry()
    
    def load_registry(self):
        with open(self.config_path, 'r') as file:
            data = json.load(file)
            self.genomes = data['genomes']
    
    def get_sequences_by_taxonomy(self, taxonomy):
        """Get all sequences for a specific taxonomy"""
        return self.genomes[taxonomy]['sequences']
    
    def get_metadata(self, taxonomy):
        """Get metadata for a specific taxonomy"""
        return self.genomes[taxonomy]['metadata']
    
    def create_classification_dataset(self, sample_size=1000, fragment_length=512):
        """Create a balanced dataset for taxonomy classification"""
        dataset = []
        for taxonomy in ['prokaryotes', 'archaea']:
            sequences = self.get_sequences_by_taxonomy(taxonomy)
            metadata = self.get_metadata(taxonomy)
            
            for seq in sequences[:sample_size]:
                dataset.append({
                    'sequence': seq[:fragment_length],
                    'taxonomy': metadata['taxonomy']
                })
        return dataset