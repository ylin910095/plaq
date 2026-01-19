.. plaq documentation master file

Welcome to plaq's documentation!
================================

**plaq** is a lattice gauge theory toolkit for Python, built on PyTorch.

.. note::

   This project is under active development.

Getting Started
---------------

Install plaq using pip:

.. code-block:: bash

   pip install plaq

Or with uv:

.. code-block:: bash

   uv add plaq

Quick Example
-------------

.. code-block:: python

   import plaq as pq

   # Check default configuration
   print(pq.config.DEFAULT_DTYPE)  # torch.complex128

Mathematical Background
-----------------------

The Wilson action for lattice gauge theory is defined as:

.. math::

   S_W = \beta \sum_{x, \mu < \nu} \left(1 - \frac{1}{N_c} \text{Re} \, \text{Tr} \, U_{\mu\nu}(x)\right)

where :math:`U_{\mu\nu}(x)` is the plaquette (elementary Wilson loop) at site :math:`x`
in the :math:`\mu`-:math:`\nu` plane, and :math:`\beta = \frac{2N_c}{g^2}` is the
inverse coupling.

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
