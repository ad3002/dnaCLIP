import os
import random
import argparse
from transformers import AutoTokenizer

def complement(seq):
    """Return complement of DNA sequence"""
    comp_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(comp_dict.get(base.upper(), base) for base in seq)

def reverse_complement(seq):
    """Return reverse complement of DNA sequence"""
    return complement(seq)[::-1]

def find_valid_sequence(sequence, tokenizer, target_tokens, avg_bases_per_token=5):
    """Find a subsequence that tokenizes to exactly target_tokens tokens using sliding window"""
    if len(sequence) < target_tokens * avg_bases_per_token:
        return None
    # Initial window size with some padding
    window_size = int(target_tokens * avg_bases_per_token * 1.23)
    
    # Try random point and extract window around it
    if window_size//2 >= len(sequence) - window_size//2:
        return None
    center_point = random.randint(window_size//2, len(sequence) - window_size//2)
    start = center_point - window_size//2
    end = start + window_size
    if start < 0:
        start = 0
        end = min(window_size, len(sequence))
    
    window = sequence[start:end]
    tokens = tokenizer.encode(window)
    
    if len(tokens) == target_tokens:
        return window
    # If we have more tokens, just take the first target_tokens
    elif len(tokens) > target_tokens:
        # Take subset of tokens and decode back to sequence
        subset_tokens = tokens[:target_tokens]
        return tokenizer.decode(subset_tokens, skip_special_tokens=True)
    
    print(f"Failed to find valid sequence in window: {len(tokens)} tokens, expected: {target_tokens}")
    return None

def collect_genome_files(input_dirs):
    """Collect paths to all genome files"""
    genome_files = []
    for input_dir in input_dirs:
        for filename in os.listdir(input_dir):
            if filename.endswith('.seq'):
                filepath = os.path.join(input_dir, filename)
                genome_files.append(filepath)
    return genome_files

def get_random_sequence_from_file(filepath):
    """Get a random sequence from a genome file"""
    with open(filepath) as f:
        sequences = f.readlines()
        return random.choice(sequences).strip()

def generate_dataset(genome_files, tokenizer, target_tokens, total_sequences, label):
    """Generate dataset by sampling from genome files"""
    result_sequences = []
    attempts = 0
    max_attempts = total_sequences * 3
    
    print(f"Generating {total_sequences} sequences with label '{label}'...")
    
    while len(result_sequences) < total_sequences and attempts < max_attempts:

        genome_file = random.choice(genome_files)    
    
        sequence = get_random_sequence_from_file(genome_file)
        valid_seq = find_valid_sequence(sequence, tokenizer, target_tokens)

        if valid_seq:
            result_sequences.append((valid_seq, label))
            continue

        attempts += 1
        if attempts % 1000 == 0:
            print(f"Attempts: {attempts}, Success rate: {len(result_sequences)/attempts:.2%}")
    
    print(f"Warning: Only generated {len(result_sequences)} sequences after {attempts} attempts")
    print(f"Target: {len([x for x in result_sequences if x])} valid sequences")
    
    return result_sequences

def main():
    parser = argparse.ArgumentParser(description='Generate random sequences from genome datasets')
    parser.add_argument('input_dirs', nargs='+', help='Input directories containing .seq files')
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer')
    parser.add_argument('--tokens', type=int, default=50, help='Target number of tokens per sequence')
    parser.add_argument('--total', type=int, required=True, help='Total number of sequences to generate')
    parser.add_argument('--label', required=True, help='Label for the sequences')
    parser.add_argument('--output', default='dataset.txt', help='Output file')
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    genome_files = collect_genome_files(args.input_dirs)
    
    if not genome_files:
        print("No genome files found!")
        return
        
    print(f"Found {len(genome_files)} genome files")
    result_sequences = generate_dataset(genome_files, tokenizer, args.tokens, args.total, args.label)
    
    with open(args.output, 'w') as f:
        for seq, label in result_sequences:
            f.write(f"{seq}\t{label}\n")
    
    print(f"\nGenerated {len(result_sequences)} sequences with label '{args.label}'")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
