import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)

@pytest.fixture
def sample_peaks_df():
    """Create a sample peaks DataFrame for testing."""
    return pd.DataFrame({
        'chrom': ['chr1', 'chr1', 'chr2'],
        'start': [100, 200, 300],
        'end': [150, 250, 350],
        'name': ['peak1', 'peak2', 'peak3'],
        'score': [1.0, 2.0, 3.0],
        'strand': ['+', '-', '+']
    })

@pytest.fixture
def sample_sequence():
    """Create a sample DNA sequence for testing."""
    return "ATCGATCGATCGATATCGATCGATCGATCG"

@pytest.fixture
def sample_motif():
    """Create a sample motif pattern for testing."""
    return "GATC"

@pytest.fixture
def sample_counts_matrix():
    """Create a sample counts matrix for testing."""
    return np.array([[1, 0, 2], 
                     [0, 3, 1], 
                     [2, 1, 0]])
