=============
Installation
=============

This guide provides instructions for installing SignXAI2.

Requirements
------------

- Python 3.9 or 3.10 (Python 3.11+ is not supported)
- TensorFlow >=2.8.0,<=2.12.1
- PyTorch >=1.10.0
- NumPy, Matplotlib, SciPy

.. warning::
   SignXAI2 requires Python 3.9 or 3.10. Using Python 3.11+ will lead to installation failures.
   Always check your Python version before installation:
   
   .. code-block:: bash
   
       python --version

Install from PyPI
-----------------

SignXAI2 requires you to explicitly choose which deep learning framework(s) to install:

**For TensorFlow users:**

.. code-block:: bash

    pip install signxai2[tensorflow]

**For PyTorch users:**

.. code-block:: bash

    pip install signxai2[pytorch]

**For both frameworks:**

.. code-block:: bash

    pip install signxai2[all]

**For development (includes all frameworks + dev tools):**

.. code-block:: bash

    pip install signxai2[dev]

**Note:** Installing ``pip install signxai2`` alone is not supported. You must specify at least one framework.

Install from Source
-------------------

.. code-block:: bash

    git clone https://github.com/IRISlaboratory/signxai2.git
    cd signxai2
    
    # Choose your installation:
    pip install -e .[tensorflow]    # TensorFlow only
    pip install -e .[pytorch]       # PyTorch only  
    pip install -e .[all]           # Both frameworks
    pip install -e .[dev]           # Development (all frameworks + tools)

Setup Git LFS
-------------

Before you get started, please set up `Git LFS <https://git-lfs.github.com/>`_ to download the large files in this repository. This is required to access the pre-trained models and example data.

.. code-block:: bash

    git lfs install

Load Data and Documentation
---------------------------

After installation, run the setup script to download documentation, examples, and sample data:

.. code-block:: bash

    bash ./prepare.sh

This will download:

- 📚 Full documentation (viewable at ``docs/index.html``)
- 📝 Example scripts and notebooks (``examples/``)  
- 📊 Sample ECG data and images (``examples/data/``)

Verify Installation
-------------------

To verify that SignXAI2 is installed correctly:

.. code-block:: python

    import signxai
    print(signxai.__version__)
    
    # Check available backends
    from signxai import list_methods
    print(f"Available methods: {len(list_methods())}")
    
    # Check specific backends if installed
    try:
        import signxai.tf_signxai
        print("TensorFlow backend available")
    except ImportError:
        print("TensorFlow backend not installed")
        
    try:
        import signxai.torch_signxai
        print("PyTorch backend available")
    except ImportError:
        print("PyTorch backend not installed")

Creating a Virtual Environment (Optional)
-----------------------------------------

If you prefer using a virtual environment instead of conda:

.. code-block:: bash

    # Create virtual environment
    python3.10 -m venv signxai_env
    
    # Activate (Linux/Mac)
    source signxai_env/bin/activate
    
    # Activate (Windows)
    signxai_env\Scripts\activate
    
    # Install SignXAI2 with your preferred framework
    pip install signxai2[tensorflow]  # or signxai2[pytorch] or signxai2[all]

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: Python version error**

.. code-block:: text

    ERROR: Package 'signxai2' requires a different Python: 3.11.0 not in '>=3.9,<3.11'

**Solution:** Install Python 3.9 or 3.10:

.. code-block:: bash

    # Using conda
    conda create -n signxai2 python=3.10
    conda activate signxai2
    
    # Or using pyenv
    pyenv install 3.10.11
    pyenv local 3.10.11

**Issue: TensorFlow installation fails**

.. code-block:: text

    ERROR: No matching distribution found for tensorflow>=2.8.0,<=2.12.1

**Solution:** Ensure you're using Python 3.9 or 3.10, as TensorFlow 2.12.1 doesn't support newer Python versions.

**Issue: Import errors after installation**

**Solution:** Ensure all dependencies are installed:

.. code-block:: bash

    # Reinstall with all dependencies
    pip uninstall signxai2
    pip install --upgrade pip
    pip install signxai2[all]  # or your preferred framework

**Issue: Missing large files (models/data)**

**Solution:** Ensure Git LFS is installed and run:

.. code-block:: bash

    git lfs install
    git lfs pull

GPU Support
-----------

SignXAI2 will automatically use GPU if available. For GPU support:

**TensorFlow GPU:**
Follow the `TensorFlow GPU installation guide <https://www.tensorflow.org/install/gpu>`_

**PyTorch GPU:**
Follow the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ and select the appropriate CUDA version.

Next Steps
----------

After installation:

1. Check out the :doc:`quickstart` guide
2. Explore the example notebooks in ``examples/tutorials/``
3. Read about :doc:`basic_usage` for detailed API information