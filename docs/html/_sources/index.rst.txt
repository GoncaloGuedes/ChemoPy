.. ChemoPy documentation master file, created by
   sphinx-quickstart on Fri Mar 15 19:37:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/chemopy_logo.png
   :align: center
   :width: 200px

Welcome to ChemoPy's Documentation!
===================================

The ChemoPy Chemometric Toolbox seamlessly incorporates scikit-learn, streamlining model training and facilitating the creation of robust pipelines. By leveraging scikit-learn's extensive functionality, users can easily train models, perform data preprocessing, and construct complex pipelines for streamlined analysis workflows. These pipelines can be shared effortlessly, allowing researchers and practitioners to collaborate and reproduce analyses with ease.

Getting Started
===============

To get started with ChemoPy, follow these steps:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/GoncaloGuedes/chemopy.git
      cd chemopy

2. Install the requirements:

   .. code-block:: bash

      pip install -r requirements.txt

This will clone the ChemoPy repository to your local machine and install all necessary dependencies.

.. note:: ChemoPy will soon be available on PyPI for easy installation using pip.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Preprocessing <chemopy.preprocessing>
   Dataloader <chemopy.dataloader>
