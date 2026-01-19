plaq package
============

The main plaq package re-exports all key types and functions for convenience.

Core Types
----------

.. autoclass:: plaq.Lattice
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: plaq.BoundaryCondition
   :members:
   :undoc-members:
   :show-inheritance:

Fields
------

.. autoclass:: plaq.SpinorField
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: plaq.GaugeField
   :members:
   :undoc-members:
   :show-inheritance:

Gamma Matrices
--------------

.. autodata:: plaq.gamma
   :annotation: = GammaMatrices instance

.. autodata:: plaq.gamma5
   :annotation: = torch.Tensor

.. autofunction:: plaq.P_plus

.. autofunction:: plaq.P_minus

Layout System
-------------

.. autofunction:: plaq.pack_eo

.. autofunction:: plaq.unpack_eo

Wilson Operator
---------------

.. autoclass:: plaq.WilsonParams
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: plaq.apply_M

.. autofunction:: plaq.apply_Mdag

.. autofunction:: plaq.apply_MdagM

Configuration
-------------

.. autoclass:: plaq.PlaqConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autodata:: plaq.config
   :annotation: = PlaqConfig instance
