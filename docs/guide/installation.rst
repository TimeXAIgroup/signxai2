=============
Installation
=============

This guide provides detailed instructions for installing SignXAI2 with proper dependencies to avoid compatibility issues.

Python Version Requirements
--------------------------

SignXAI2 requires Python 3.9, 3.10, 3.11, or 3.12. Python 3.13 and newer versions are not supported due to TensorFlow compatibility limitations.

.. warning::
   Using Python 3.13+ will lead to TensorFlow installation failures. Always check your Python version before installation:
   
   .. code-block:: bash
   
       python --version

Recommended Installation Method
------------------------------

The most reliable way to install SignXAI2 is using Conda with a step-by-step approach to manage dependencies properly.

Step 1: Create a Fresh Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Create a new conda environment with Python 3.10 (recommended)
    conda create -n signxai2 python=3.10 -y
    conda activate signxai2

Step 2: Install Jupyter Support (for Notebooks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Install Jupyter and register the kernel
    conda install -y jupyter ipykernel
    python -m ipykernel install --user --name=signxai2 --display-name="Python (signxai2)"

Step 3: Install Dependencies in the Correct Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Navigate to the SignXAI2 package directory (if you're installing from source)
    cd /path/to/signxai2/

    # Install common dependencies first
    pip install -r requirements/common.txt

    # Install PyTorch dependencies
    pip install -r requirements/pytorch.txt

    # Install TensorFlow dependencies last
    pip install -r requirements/tensorflow.txt

.. note::
   The order of installation is important to avoid dependency conflicts.

Step 4: Install SignXAI
~~~~~~~~~~~~~~~~~~~~~

For development mode (from source):

.. code-block:: bash

    # Install in development mode
    pip install -e .

For regular installation from PyPI:

.. code-block:: bash

    # With TensorFlow support
    pip install signxai[tensorflow]

    # With PyTorch support
    pip install signxai[pytorch]

    # With both frameworks
    pip install signxai[all]

Step 5: Verify Installation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -c "import numpy, matplotlib, torch, tensorflow, signxai; print(f'NumPy: {numpy.__version__}, PyTorch: {torch.__version__}, TensorFlow: {tensorflow.__version__}, SignXAI: {signxai.__version__}')"

This should output the version numbers for all the key packages.

Framework-Specific Installations
-------------------------------

If you only need one framework, you can install SignXAI with specific dependencies:

TensorFlow Only
~~~~~~~~~~~~~

.. code-block:: bash
    
    # Create conda environment with Python 3.10
    conda create -n signxai-tf python=3.10 -y
    conda activate signxai-tf
    
    # Install dependencies
    pip install -r requirements/common.txt
    pip install -r requirements/tensorflow.txt
    
    # Install SignXAI
    pip install -e .

PyTorch Only
~~~~~~~~~~

.. code-block:: bash
    
    # Create conda environment with Python 3.10
    conda create -n signxai-pt python=3.10 -y
    conda activate signxai-pt
    
    # Install dependencies
    pip install -r requirements/common.txt
    pip install -r requirements/pytorch.txt
    
    # Install SignXAI
    pip install -e .

Running Jupyter Notebooks
------------------------

After installation, you can run the example notebooks:

.. code-block:: bash

    # Activate your environment
    conda activate signxai
    
    # Start Jupyter notebook
    jupyter notebook

When opening a notebook, make sure to select the correct kernel:

1. Click on the "Kernel" menu
2. Select "Change kernel"
3. Choose "Python (signxai)" from the dropdown

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: TensorFlow installation fails**

.. code-block:: bash

    ERROR: No matching distribution found for tensorflow<=2.12.1,>=2.8.0

**Solution**: Check your Python version. TensorFlow 2.12.1 requires Python 3.9-3.12:

.. code-block:: bash

    python --version
    # If using 3.13+, create a new environment with 3.10
    conda create -n signxai python=3.10 -y

**Issue: Package dependency conflicts**

**Solution**: Install dependencies in the correct order as specified above.

**Issue: Cannot import signxai module**

**Solution**: Check that you've activated the correct conda environment:

.. code-block:: bash

    # Check which environment is active
    conda info --envs
    
    # Activate the correct environment
    conda activate signxai
    
    # Verify installation
    python -c "import signxai; print(signxai.__version__)"

Dependencies
-----------

Core Dependencies
~~~~~~~~~~~~~~~

* Python (>=3.9, <3.13)
* NumPy (>=1.19.0)
* Matplotlib (>=3.7.0)
* SciPy (>=1.10.0)
* Pillow (>=8.0.0)
* Requests (>=2.25.0)

Framework Dependencies
~~~~~~~~~~~~~~~~~~~

TensorFlow:
    * TensorFlow (>=2.8.0, <=2.12.1)

PyTorch:
    * PyTorch (>=1.10.0)

Optional Dependencies
~~~~~~~~~~~~~~~~~~

* Scikit-image (for visualization and comparison)
* Jupyter/IPython (for running example notebooks)

CUDA and GPU Support
~~~~~~~~~~~~~~~~~~~

SignXAI does not directly specify CUDA dependencies. For GPU support, ensure you have installed the GPU-compatible versions of TensorFlow and/or PyTorch according to their official documentation:

- `TensorFlow GPU Support <https://www.tensorflow.org/install/gpu>`_
- `PyTorch GPU Support <https://pytorch.org/get-started/locally/>`_