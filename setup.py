#!/usr/bin/env python3
"""Setup script for SignXAI2 - Comprehensive explainable AI library with TensorFlow and PyTorch support."""

from setuptools import setup, find_packages
import os

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from the existing requirements structure
def read_requirements(filename):
    """Read requirements from requirements/{filename}."""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements", filename)
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Get common requirements
common_requirements = read_requirements("common.txt")

setup(
    name="signxai2",
    version="0.4.0",
    author="TIME XAI Group",
    author_email="nils.gumpfer@kite.thm.de",
    description="A comprehensive explainable AI library supporting both TensorFlow and PyTorch with unified API and advanced XAI methods including SIGN, LRP, and Grad-CAM. Authored by Nils Gumpfer, Jana Fischer and Alexander Paul.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TimeXAI-group/signxai2",
    project_urls={
        "Bug Tracker": "https://github.com/TimeXAI-group/signxai2/issues",
        "Documentation": "https://timexai-group.github.io/signxai2/index.html",
        "Source": "https://github.com/TimeXAI-group/signxai2",
    },
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
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=common_requirements,
    extras_require={
        "tensorflow": read_requirements("tensorflow.txt"),
        "pytorch": read_requirements("pytorch.txt"), 
        "dev": read_requirements("dev.txt"),
        "all": read_requirements("tensorflow.txt") + read_requirements("pytorch.txt"),
    },
    entry_points={
        "console_scripts": [
            # Add command-line tools if needed
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "explainable-ai", "xai", "pytorch", "tensorflow", "lrp", 
        "layer-wise-relevance-propagation", "gradient", "attribution",
        "interpretability", "machine-learning", "deep-learning",
        "computer-vision", "neural-networks", "innvestigate-compatible"
    ],
)