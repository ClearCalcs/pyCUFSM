Usage
=====

.. _installation:

Installation
------------

To use pyCUFSM, first install it using pip:

.. code-block:: console

   $ pip install pycufsm

Quick Start
-----------

There are two main entry points into running a pyCUFSM analysis, one of which is intended to be as easy to use as possible for common analyses, and the other is intended to maintain near-full compatibility with the original Matlab CUFSM syntax. 

Easy Usage
~~~~~~~~~~

Use the ``strip_new()`` function::
   import pycufsm.strip_new
   import pycufsm.cutwp.prop2_new
   
   nodes = [[5, 1], [5, 0], [2.5, 0], [0, 0], [0, 3], [0, 6], [0, 9], [2.5, 9], [5, 9], [5, 8]]
   elements = [{"nodes": "all", "t": 0.1, "mat": "CFS"}]
   props = {"CFS": {"E": 29500, "nu": 0.3}}
   yield_force = {"force": "Mxx", "direction": "+", "f_y": 50}
   sect_props = cutwp.prop2_new(nodes, elements)
   
   signature, curve, shapes, nodes_stressed, lengths = strip_new(nodes=nodes, elements=elements, props=props, yield_force=yield_force, sect_props=sect_props)

The ``signature`` output is probably the one you're most interested in. If you plot it against ``lengths`` with your favourite plotting library, then you'll see the signature curve for your section with the given 'Mxx' moment loading about the X-axis. 

Matlab-Compatibility Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're used to using CUFSM's original Matlab interface, then you're in luck - we've still maintained the original syntax that you're used to. You can even load a .MAT file directly into the pyCUFSM! To do so::
   import pycufsm.strip
   import pycufsm.cutwp.prop2
   import pycufsm.helpers.load_cufsm_mat
   import numpy as np

   inputs = helpers.load_cufsm_mat(mat_file="path/to/data_file.mat")
   coords = inputs["nodes"][:, 1:3]
   ends = inputs["elements"][:, 1:4]
   sect_props = cutwp.prop2(coords, ends)
   m_all_ones = np.ones((len(inputs["lengths"]), 1))

   signature, curve, shapes = strip(
      props=inputs["props"], 
      nodes=inputs["nodes"], 
      elements=inputs["elements"], 
      lengths=inputs["lengths"],
      springs=inputs["springs"], 
      constraints=inputs["constraints"], 
      gbt_con=inputs["GBTcon"], 
      b_c='S-S', 
      m_all=m_all_ones, 
      n_eigs=10, 
      sect_props=sect_props
   )

