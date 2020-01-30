# This example presents a very simple Cee section,
# solved for pure compression,
# in the Imperial unit system

import numpy as np
from pyCUFSM.fsm import strip
from pyCUFSM.preprocess import stress_gen


def __main__():
    # Define an isotropic material with E = 29,500 ksi and nu = 0.3
    props = [[0, 29500, 29500, 0.3, 0.3, 29500 / (2 * (1 + 0.3))]]

    # Define a lightly-meshed Cee shape
    # (1 element per lip, 2 elements per flange, 3 elements on the web)
    # Nodal location units are inches
    nodes = [[0, 5, 1, 1, 1, 1, 1, 0], [1, 5, 0, 1, 1, 1, 1, 0],
             [2, 2.5, 0, 1, 1, 1, 1, 0], [3, 0, 0, 1, 1, 1, 1, 0],
             [4, 0, 3, 1, 1, 1, 1, 0], [5, 0, 6, 1, 1, 1, 1, 0],
             [6, 0, 9, 1, 1, 1, 1, 0], [7, 2.5, 9, 1, 1, 1, 1, 0],
             [8, 5, 9, 1, 1, 1, 1, 0], [9, 5, 8, 1, 1, 1, 1, 0]]
    elements = [[0, 0, 1, 0.1, 0], [1, 1, 2, 0.1, 0], [2, 2, 3, 0.1, 0],
                [3, 3, 4, 0.1, 0], [4, 4, 5, 0.1, 0], [5, 5, 6, 0.1, 0],
                [6, 6, 7, 0.1, 0], [7, 7, 8, 0.1, 0], [8, 8, 9, 0.1, 0]]

    # These lengths will generally provide sufficient accuracy for
    # local, distortional, and global buckling modes
    # Length units are inches
    lengths = [
        0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75,
        4, 4.25, 4.5, 4.75, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52,
        54, 56, 58, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 132, 144,
        156, 168, 180, 204, 228, 252, 276, 300
    ]

    # No special springs or constraints
    springs = []
    constraints = []

    # Values here correspond to signature curve basis and orthogonal based upon geometry
    gbt_con = {
        'glob': [0],
        'dist': [0],
        'local': [0],
        'other': [0],
        'o_space': 1,
        'couple': 1,
        'orth': 2,
        'norm': 0,
    }

    # Simply-supported boundary conditions
    b_c = 'S-S'

    # For signature curve analysis, only a single array of ones makes sense here
    m_all = np.ones((len(lengths), 1))

    # Solve for 10 eigenvalues
    n_eigs = 10

    # Set the section properties for this simple section
    # Normally, these might be calculated by an external package
    sect_props = {
        'cx': 1.67,
        'cy': 4.5,
        'x0': -2.27,
        'y0': 4.5,
        'phi': 0,
        'A': 2.06,
        'Ixx': 28.303,
        'Ixy': 0,
        'Iyy': 7.019,
        'I11': 28.303,
        'I22': 7.019
    }

    # Generate the stress points assuming 50 ksi yield and pure compression
    nodes_p = stress_gen(nodes=nodes,
                         forces={
                             'P': sect_props['A'] * 50,
                             'Mxx': 0,
                             'Myy': 0,
                             'M11': 0,
                             'M22': 0
                         },
                         sect_props=sect_props)

    # Perform the Finite Strip Method analysis
    signature, curve, shapes = strip(props=props,
                                     nodes=nodes_p,
                                     elements=elements,
                                     lengths=lengths,
                                     springs=springs,
                                     constraints=constraints,
                                     gbt_con=gbt_con,
                                     b_c=b_c,
                                     m_all=m_all,
                                     n_eigs=n_eigs,
                                     sect_props=sect_props)

    # Return the important example results
    # The signature curve is simply a matter of plotting the
    # 'signature' values against the lengths
    # (usually on a logarithmic axis)
    return {
        'X_values': lengths,
        'Y_values': signature,
        'Y_values_allmodes': curve,
        'Orig_coords': nodes,
        'Deformations': shapes
    }
