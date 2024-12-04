import os
import requests
import pandas as pd

# URLs for the required files
ASSEMBLY_SUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_refseq.txt"
TAXONOMY_DUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz"

# Output file names
ASSEMBLY_FILE = "assembly_summary_refseq.txt"
TAXONOMY_DIR = "taxonomy"
METADATA_OUTPUT = "genomes_metadata_with_taxonomy.csv"

def download_file(url, output_path):
    """Download a file from a URL."""
    print(f"Downloading: {url}")
    response = requests.get(url, stream=True)
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    print(f"Downloaded: {output_path}")

def download_taxonomy():
    """Download and extract the NCBI taxonomy dump."""
    if not os.path.exists(TAXONOMY_DIR):
        os.makedirs(TAXONOMY_DIR)
    tar_path = os.path.join(TAXONOMY_DIR, "new_taxdump.tar.gz")
    if not os.path.exists(tar_path):
        download_file(TAXONOMY_DUMP_URL, tar_path)
        print("Extracting taxonomy dump...")
        os.system(f"tar -xzf {tar_path} -C {TAXONOMY_DIR}")
        print("Taxonomy dump extracted.")
    else:
        print("Taxonomy dump already exists.")

def build_taxonomy_lookup():
    """Parse the taxonomy files and build a lookup table."""
    nodes_file = os.path.join(TAXONOMY_DIR, "nodes.dmp")
    names_file = os.path.join(TAXONOMY_DIR, "names.dmp")

    parent_taxids = {}
    taxid2rank = {}
    with open(nodes_file, "r") as f:
        nodes = [l.strip().split("\t|\t") for l in f]
        for node in nodes:
            taxid = node[0]
            parent_taxid = node[1]
            rank = node[2]
            parent_taxids[taxid] = parent_taxid
            parent_taxids.setdefault(parent_taxid, None)
            taxid2rank[taxid] = rank
    
    # Parse nodes.dmp to build the hierarchy
    taxonomy = {}
    # Parse names.dmp to add scientific names
    with open(names_file, "r") as f:
        for line in f:
            fields = line.strip().split("\t|\t")
            taxid = fields[0].strip()
            name = fields[1].strip().split("\t")[0]
            name_type = fields[3].strip().split("\t")[0]

            taxonomy.setdefault(taxid, {})
            taxonomy[taxid].setdefault("names", {})

            taxonomy[taxid]["parent_taxid"] = parent_taxids[taxid]
            taxonomy[taxid]["rank"] = taxid2rank[taxid]
                
            # Store name by type
            taxonomy[taxid]["names"][name_type] = name
            
            # If this is a scientific name, set it as primary name
            if name_type == "scientific name":
                taxonomy[taxid]["name"] = name

    # Set fallback name for entries without scientific name
    for taxid in taxonomy:
        if "name" not in taxonomy[taxid]:
            name_types = taxonomy[taxid].get("names", {})
            if name_types:
                # Use first available name as fallback
                taxonomy[taxid]["name"] = next(iter(name_types.values()))
            else:
                taxonomy[taxid]["name"] = "Unknown"

    return taxonomy

def get_full_taxonomy(taxonomy, taxid):
    """Recursively build the full taxonomy path."""
    path = []
    while taxid in taxonomy and taxid != "1":  # Stop at the root
        node = taxonomy[taxid]
        path.append((node["rank"], node.get("name", "unknown")))
        taxid = node["parent_taxid"]
    return path[::-1]

def process_genomes():
    """Download and process genome metadata."""
    if not os.path.exists(ASSEMBLY_FILE):
        if not os.path.exists(ASSEMBLY_FILE):
            download_file(ASSEMBLY_SUMMARY_URL, ASSEMBLY_FILE)
    else:
        print("Assembly summary already exists.")
    
    print("Reading assembly summary...")
    df = pd.read_csv(ASSEMBLY_FILE, sep="\t", skiprows=1)
    df = df[["#assembly_accession", "species_taxid", "organism_name", "assembly_level", "genome_size"]]
    
    print("Building taxonomy lookup...")
    taxonomy = build_taxonomy_lookup()
    
    print("Mapping genomes to full taxonomy...")
    df["taxonomy"] = df["species_taxid"].apply(
        lambda taxid: get_full_taxonomy(taxonomy, str(taxid))
    )
    
    print("Saving metadata with taxonomy...")
    df.to_csv(METADATA_OUTPUT, index=False)
    print(f"Metadata saved to {METADATA_OUTPUT}")

def get_kingdom_from_taxonomy(tax_str):
    """Extract kingdom from taxonomy string."""
    try:
        taxonomy = eval(tax_str)
        for rank, name in taxonomy:
            if rank == 'kingdom':
                return name
        # If no kingdom rank found, try to use superkingdom
        for rank, name in taxonomy:
            if rank == 'superkingdom':
                return name
        return "Unknown"
    except:
        return "Unknown"

def get_superkingdom_from_taxonomy(tax_str):
    """Extract superkingdom from taxonomy string."""
    try:
        taxonomy = eval(tax_str)
        for rank, name in taxonomy:
            if rank == 'superkingdom':
                return name
        return "Unknown"
    except:
        return "Unknown"

if __name__ == "__main__":

    if not os.path.exists(METADATA_OUTPUT):
        
        print("Step 1: Downloading and processing taxonomy...")
        download_taxonomy()
        print("Step 2: Processing genome metadata...")
        process_genomes()

    import pandas as pd
    import numpy as np

    # Read the metadata file
    df = pd.read_csv("genomes_metadata_with_taxonomy.csv")

    # Get total stats
    total_genomes = len(df)
    total_size_gb = df['genome_size'].sum() / (1024**3)  # Convert to GB

    import pandas as pd
    import numpy as np

    # Read the metadata file
    df = pd.read_csv("genomes_metadata_with_taxonomy.csv")

    # Get total stats
    total_genomes = len(df)
    total_size_gb = df['genome_size'].sum() / (1024**3)  # Convert to GB

    # Group by kingdom (first level in taxonomy)
    kingdom_stats = df.copy()
    kingdom_stats['kingdom'] = kingdom_stats['taxonomy'].apply(get_kingdom_from_taxonomy)

    kingdom_summary = kingdom_stats.groupby('kingdom').agg({
        '#assembly_accession': 'count',  # Number of genomes
        'genome_size': ['sum', 'mean']   # Total and average size
    }).round(2)

    # Convert genome sizes to GB
    kingdom_summary[('genome_size', 'sum')] = kingdom_summary[('genome_size', 'sum')] / (1024**3)
    kingdom_summary[('genome_size', 'mean')] = kingdom_summary[('genome_size', 'mean')] / (1024**3)

    # Group by superkingdom
    superkingdom_stats = df.copy()
    superkingdom_stats['superkingdom'] = superkingdom_stats['taxonomy'].apply(get_superkingdom_from_taxonomy)

    superkingdom_summary = superkingdom_stats.groupby('superkingdom').agg({
        '#assembly_accession': 'count',  # Number of genomes
        'genome_size': ['sum', 'mean']   # Total and average size
    }).round(2)

    # Convert genome sizes to GB
    superkingdom_summary[('genome_size', 'sum')] = superkingdom_summary[('genome_size', 'sum')] / (1024**3)
    superkingdom_summary[('genome_size', 'mean')] = superkingdom_summary[('genome_size', 'mean')] / (1024**3)

    print(f"Total RefSeq Stats:")
    print(f"Number of genomes: {total_genomes:,}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print("\nStats by Kingdom:")
    print(kingdom_summary)
    print("\nStats by Superkingdom:")
    print(superkingdom_summary)