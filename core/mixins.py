from typing import Dict, List, Optional
import torch
import numpy as np

class NucleotideFeaturesMixin:
    """Mixin for nucleotide-based calculations"""
    def count_nucleotides(self, sequence: str) -> Dict[str, int]:
        sequence = sequence.upper()
        return {
            'A': sequence.count('A'),
            'T': sequence.count('T'),
            'G': sequence.count('G'),
            'C': sequence.count('C')
        }
    
    def calculate_gc_content(self, sequence: str) -> float:
        counts = self.count_nucleotides(sequence)
        total = sum(counts.values())
        return (counts['G'] + counts['C']) / total if total > 0 else 0.0

class TriNucleotideFeaturesMixin:
    """Mixin for trinucleotide-based calculations"""
    def get_trinucleotides(self, sequence: str) -> List[str]:
        sequence = sequence.upper()
        return [sequence[i:i+3] for i in range(len(sequence)-2)]
    
    def count_trinucleotides(self, sequence: str) -> Dict[str, int]:
        trinucs = self.get_trinucleotides(sequence)
        return {trinuc: trinucs.count(trinuc) for trinuc in set(trinucs)}

class MetricsCalculationMixin:
    """Mixin for common metric calculations"""
    @staticmethod
    def calculate_regression_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        correlation = np.corrcoef(predictions.flatten(), labels.flatten())[0,1]
        return {'mse': mse, 'mae': mae, 'correlation': correlation}
    
    @staticmethod
    def calculate_classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        accuracy = np.mean(predictions == labels)
        true_pos = np.sum((predictions == 1) & (labels == 1))
        false_pos = np.sum((predictions == 1) & (labels == 0))
        false_neg = np.sum((predictions == 0) & (labels == 1))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

class ThermalFeaturesMixin:
    """Mixin for thermal calculations like melting temperature"""
    def __init__(self):
        # Nearest-neighbor parameters (SantaLucia & Hicks, 2004)
        self.nn_params = {
            'AA/TT': (-7.9, -22.2),
            'AT/TA': (-7.2, -20.4),
            'TA/AT': (-7.2, -21.3),
            'CA/GT': (-8.5, -22.7),
            'GT/CA': (-8.4, -22.4),
            'CT/GA': (-7.8, -21.0),
            'GA/CT': (-8.2, -22.2),
            'CG/GC': (-10.6, -27.2),
            'GC/CG': (-9.8, -24.4),
            'GG/CC': (-8.0, -19.9),
        }
        self.Na = 50  # mM Na+
        self.R = 1.987  # Gas constant in cal/K·mol

    def calculate_tm(self, sequence):
        """Calculate normalized melting temperature using nearest-neighbor method"""
        sequence = sequence.upper()
        if len(sequence) < 2:
            return 0.0
            
        dH = 0  # Enthalpy
        dS = 0  # Entropy
        
        for i in range(len(sequence) - 1):
            pair = sequence[i:i+2]
            complement = ''.join(['T' if b == 'A' else 'A' if b == 'T' else 
                                'G' if b == 'C' else 'C' for b in pair[::-1]])
            key = f"{pair}/{complement}"
            
            if key in self.nn_params:
                h, s = self.nn_params[key]
                dH += h
                dS += s
            else:
                dH += -8.0
                dS += -22.0
        
        if sequence[0] in 'AT':
            dH += 2.3
            dS += 4.1
        if sequence[-1] in 'AT':
            dH += 2.3
            dS += 4.1
        
        salt_correction = 0.368 * (len(sequence) - 1) * np.log(self.Na/1000)
        dS += salt_correction
        
        tm = (1000 * dH) / (dS + self.R * np.log(1/4)) - 273.15
        normalized_tm = (tm - 0) / (120 - 0)
        return max(0.0, min(1.0, normalized_tm))

class BendabilityFeaturesMixin:
    """Mixin for DNA bendability calculations"""
    def __init__(self):
        # DNAse I-based bendability parameters
        self.bendability_params = {
            'AAA': 0.14, 'AAT': 0.16, 'AAG': 0.18, 'AAC': 0.17,
            'ATA': 0.15, 'ATT': 0.14, 'ATG': 0.16, 'ATC': 0.15,
            'AGA': 0.17, 'AGT': 0.16, 'AGG': 0.18, 'AGC': 0.17,
            'ACA': 0.16, 'ACT': 0.15, 'ACG': 0.17, 'ACC': 0.16,
            'TAA': 0.15, 'TAT': 0.14, 'TAG': 0.16, 'TAC': 0.15,
            'TTA': 0.14, 'TTT': 0.14, 'TTG': 0.16, 'TTC': 0.15,
            'TGA': 0.16, 'TGT': 0.15, 'TGG': 0.17, 'TGC': 0.16,
            'TCA': 0.15, 'TCT': 0.14, 'TCG': 0.16, 'TCC': 0.15,
            'GAA': 0.18, 'GAT': 0.16, 'GAG': 0.19, 'GAC': 0.17,
            'GTA': 0.16, 'GTT': 0.15, 'GTG': 0.17, 'GTC': 0.16,
            'GGA': 0.18, 'GGT': 0.17, 'GGG': 0.19, 'GGC': 0.18,
            'GCA': 0.17, 'GCT': 0.16, 'GCG': 0.18, 'GCC': 0.17,
            'CAA': 0.17, 'CAT': 0.15, 'CAG': 0.17, 'CAC': 0.16,
            'CTA': 0.15, 'CTT': 0.14, 'CTG': 0.16, 'CTC': 0.15,
            'CGA': 0.17, 'CGT': 0.16, 'CGG': 0.18, 'CGC': 0.17,
            'CCA': 0.16, 'CCT': 0.15, 'CCG': 0.17, 'CCC': 0.16
        }

    def calculate_bendability(self, sequence: str) -> float:
        """Calculate average bendability score"""
        sequence = sequence.upper()
        if len(sequence) < 3:
            return 0.0
            
        bendability_sum = 0.0
        count = 0
        
        for i in range(len(sequence) - 2):
            trinuc = sequence[i:i+3]
            if trinuc in self.bendability_params:
                bendability_sum += self.bendability_params[trinuc]
                count += 1
        
        return bendability_sum / count if count > 0 else 0.0

class FlexibilityFeaturesMixin:
    """Mixin for DNA flexibility calculations"""
    def __init__(self):
        # DNA flexibility parameters (trinucleotide scale)
        self.flexibility_params = {
            'AAA': (0.26, 0.14), 'AAT': (0.22, 0.16), 'AAG': (0.20, 0.18),
            'AAC': (0.19, 0.17), 'ATA': (0.20, 0.15), 'ATT': (0.17, 0.14),
            'ATG': (0.18, 0.16), 'ATC': (0.16, 0.15), 'AGA': (0.19, 0.17),
            # ...existing flexibility parameters...
            'CCA': (0.14, 0.16), 'CCT': (0.12, 0.15), 'CCG': (0.13, 0.17),
            'CCC': (0.11, 0.16)
        }
    
    def calculate_flexibility(self, sequence: str) -> List[float]:
        """Calculate average propeller twist and bendability"""
        sequence = sequence.upper()
        if len(sequence) < 3:
            return [0.0, 0.0]
        
        propeller_sum = 0.0
        bendability_sum = 0.0
        count = 0
        
        for i in range(len(sequence) - 2):
            trinuc = sequence[i:i+3]
            if trinuc in self.flexibility_params:
                prop, bend = self.flexibility_params[trinuc]
                propeller_sum += prop
                bendability_sum += bend
                count += 1
        
        if count == 0:
            return [0.0, 0.0]
        
        return [
            propeller_sum / count,
            bendability_sum / count
        ]