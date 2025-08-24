#!/usr/bin/env python3
"""
Setup script for NEAT Implementation Package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
    return long_description

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="neat-implementation",
    version="1.0.0",
    author="Crystal Hu",
    author_email="crystal.hu@example.com",
    description="A complete implementation of NEAT (NeuroEvolution of Augmenting Topologies) with GPU acceleration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/CrystalHu940106/evojax",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "jax[cuda]>=0.4.13",
            "jaxlib[cuda]>=0.4.13",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

