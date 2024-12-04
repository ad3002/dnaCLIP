import csv
import argparse
from ast import literal_eval
import random
import os
import shutil
import subprocess

def parse_fasta(fasta_file):
    """Simple FASTA parser that yields sequences"""
    current_sequence = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                if current_sequence:
                    yield ''.join(current_sequence)
                    current_sequence = []
            else:
                current_sequence.append(line.strip())
    if current_sequence:
        yield ''.join(current_sequence)

def find_assemblies_by_taxonomy(csv_file, rank, value):
    matching_assemblies = []
    total_size = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            assembly_id = row[0]
            genome_size = int(row[4])
            taxonomy = literal_eval(row[5])  # Convert string representation of list to actual list
            
            # Check if the given rank and value match in the taxonomy
            for tax_rank, tax_value in taxonomy:
                if tax_rank == rank and tax_value == value:
                    matching_assemblies.append((assembly_id, genome_size))
                    total_size += genome_size
                    break

    return matching_assemblies, total_size

def sample_assemblies(assemblies, required_size_gb):
    required_size = required_size_gb * 1e9  # Convert to bases
    total_available = sum(size for _, size in assemblies)
    
    if total_available < required_size:
        return assemblies, total_available
    
    selected = []
    current_size = 0
    # Create copy and shuffle to randomly sample
    assemblies_copy = assemblies.copy()
    random.shuffle(assemblies_copy)
    
    for assembly in assemblies_copy:
        if current_size + assembly[1] <= required_size:
            selected.append(assembly)
            current_size += assembly[1]
    
    return selected, current_size

def download_and_process_genomes(assemblies, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for assembly_id, _ in assemblies:
        seq_path = os.path.join(output_dir, f"{assembly_id}.seq")
        
        if os.path.exists(seq_path):
            print(f"Skipping {assembly_id} - already processed")
            continue
            
        print(f"Processing {assembly_id}...")
        temp_dir = os.path.join(output_dir, f"temp_{assembly_id}")
        zip_file = os.path.join(output_dir, f"{assembly_id}.zip")
        
        try:
            # Download dehydrated archive
            subprocess.run([
                "datasets", "download", "genome", "accession",
                assembly_id, "--dehydrated", "--filename", zip_file
            ], check=True)
            
            # Unzip archive
            os.makedirs(temp_dir, exist_ok=True)
            subprocess.run([
                "unzip", zip_file, "-d", temp_dir
            ], check=True)
            
            # Rehydrate data
            subprocess.run([
                "datasets", "rehydrate", "--directory", temp_dir
            ], check=True)
            
            # Find and process the FASTA file
            ncbi_data_dir = os.path.join(temp_dir, "ncbi_dataset", "data", assembly_id)
            for file in os.listdir(ncbi_data_dir):
                if file.endswith(".fna"):
                    fna_path = os.path.join(ncbi_data_dir, file)
                    # Convert to one-line sequence format
                    with open(seq_path, 'w') as f_out:
                        for sequence in parse_fasta(fna_path):
                            f_out.write(sequence + '\n')
                    break
            
            print(f"Successfully processed {assembly_id}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing {assembly_id}: {e}")
            if os.path.exists(seq_path):
                os.remove(seq_path)
        finally:
            # Clean up temporary files
            if os.path.exists(zip_file):
                os.remove(zip_file)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description='Find assemblies by taxonomic rank and value')
    parser.add_argument('rank', help='Taxonomic rank (e.g., "phylum", "genus")')
    parser.add_argument('value', help='Value for the taxonomic rank')
    parser.add_argument('size', type=float, help='Required dataset size in Gb')
    args = parser.parse_args()

    csv_file = 'genomes_metadata_with_taxonomy.csv'
    assemblies, total_size = find_assemblies_by_taxonomy(csv_file, args.rank, args.value)
    
    print(f"\nResults for {args.rank}: {args.value}")
    print(f"Total available assemblies: {len(assemblies)}")
    print(f"Total available size: {total_size/1e9:.2f} Gb")
    
    if total_size < args.size * 1e9:
        print(f"\nWARNING: Available size ({total_size/1e9:.2f} Gb) is less than requested size ({args.size} Gb)")
        selected_assemblies = assemblies
        selected_size = total_size
    else:
        selected_assemblies, selected_size = sample_assemblies(assemblies, args.size)
        print(f"\nRandomly sampled dataset:")
        print(f"Selected assemblies: {len(selected_assemblies)}")
        print(f"Selected size: {selected_size/1e9:.2f} Gb")

        print("\nSelected assemblies:")
        for assembly_id, size in selected_assemblies:
            print(f"{assembly_id}: {size/1e6:.2f} Mb")

    download = input("\nDo you want to download these genomes? (yes/no): ").lower().strip()
    if download == 'yes':
        output_dir = input("Enter output directory path: ").strip()
        download_and_process_genomes(selected_assemblies, output_dir)
        print(f"\nAll genomes have been downloaded and processed in: {output_dir}")

if __name__ == "__main__":
    main()