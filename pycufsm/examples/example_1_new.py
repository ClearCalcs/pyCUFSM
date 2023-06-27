from typing import Dict, List

import numpy as np

from pycufsm.fsm import strip_new
from pycufsm.types import Analysis_Config, New_Element, Sect_Props

# This example presents a very simple Cee section,
# solved for pure compression,
# in the Imperial unit system


def __main__() -> Dict[str, np.ndarray]:
    # Define an isotropic material with E = 29,500 ksi and nu = 0.3
    props = {"CFS": {'E': 29500, 'nu': 0.3}}

    # Define a lightly-meshed Cee shape
    # (1 element per lip, 2 elements per flange, 3 elements on the web)
    # Nodal location units are inches
    nodes = [[5, 1], [5, 0], [2.5, 0], [0, 0], [0, 3], [0, 6], [0, 9], [2.5, 9], [5, 9], [5, 8]]
    elements: List[New_Element] = [{"nodes": "all", "t": 0.1, "mat": "CFS"}]

    # Values here correspond to signature curve basis and orthogonal based upon geometry
    analysis_config: Analysis_Config = {'b_c': "S-S", 'n_eigs': 10}

    # Set the section properties for this simple section
    # Normally, these might be calculated by an external package
    sect_props: Sect_Props = {
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
        'I22': 7.019,
        'Cw': 0,
        'J': 0,
        'B1': 0,
        'B2': 0,
        'wn': np.array([])
    }

    # Perform the Finite Strip Method analysis
    signature, curve, shapes, nodes_stressed, lengths = strip_new(
        props=props,
        nodes=nodes,
        elements=elements,
        forces={
            'P': sect_props['A'] * 50,
            'Mxx': 0,
            'Myy': 0,
            'M11': 0,
            'M22': 0,
            'restrain': False,
            'offset': [-elements[0]["t"] / 2, -elements[0]["t"] / 2]
        },
        analysis_config=analysis_config,
        sect_props=sect_props
    )

    # Return the important example results
    # The signature curve is simply a matter of plotting the
    # 'signature' values against the lengths
    # (usually on a logarithmic axis)
    return {
        'X_values': lengths,
        'Y_values': signature,
        'Y_values_allmodes': curve,
        'Orig_coords': nodes_stressed,
        'Deformations': shapes
    }
