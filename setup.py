
from setuptools import setup, find_packages

setup(
    name="dnaCLIP",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "datasets>=2.12.0",
        "scikit-learn>=1.2.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "evaluate>=0.4.0",
    ],
)