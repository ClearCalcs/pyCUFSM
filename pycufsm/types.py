from typing import Any, List, Literal, Union

import numpy as np
from typing_extensions import TypedDict

B_C = Literal["S-S", "C-C", "S-C", "C-F", "C-G"]  # pylint: disable=invalid-name

GBT_Con = TypedDict(
    'GBT_Con', {
        'glob': List[Literal[0, 1]],
        'dist': List[Literal[0, 1]],
        'local': List[Literal[0, 1]],
        'other': List[Literal[0, 1]],
        'o_space': int,
        'couple': int,
        'orth': int,
        'norm': int
    }
)

Forces = TypedDict(
    'Forces', {
        'P': float,
        'Mxx': float,
        'Myy': float,
        'M11': float,
        'M22': float,
    }
)

Sect_Props = TypedDict(
    'Sect_Props', {
        "A": float,
        "cx": float,
        "cy": float,
        "Ixx": float,
        "Iyy": float,
        "Ixy": float,
        "phi": float,
        "I11": float,
        "I22": float,
        "J": float,
        "x0": float,
        "y0": float,
        "Cw": float,
        "B1": float,
        "B2": float,
        "wn": np.ndarray
    }
)

Sect_Geom = TypedDict(
    'Sect_Geom', {
        'n_d': int,
        'n_b1': int,
        'n_b2': int,
        'n_l1': int,
        'n_l2': int,
        'n_r': int,
        'type': str,
        'b_1': float,
        'b_2': float,
        't': float,
        'r_out': float,
        'd': float,
        'l_1': float,
        'l_2': float
    }
)

Cufsm_MAT_File = TypedDict(
    'Cufsm_MAT_File',
    {
        'node': list,
        'elem': list,
        'lengths': list,
        'prop': list,
        'constraints': list,
        'springs': list,
        'curve': list,
        'GBTcon':
            Any,  # this is some special structure with string dictionaries but a dtype attribute
        'shapes': list,
        'clas': str,
    }
)

PyCufsm_Input = TypedDict(
    'PyCufsm_Input', {
        'nodes': np.ndarray,
        'elements': np.ndarray,
        'lengths': np.ndarray,
        'props': np.ndarray,
        'constraints': np.ndarray,
        'springs': np.ndarray,
        'curve': np.ndarray,
        'shapes': np.ndarray,
        'clas': str,
        'GBTcon': GBT_Con
    }
)

New_Props = TypedDict(
    'New_Props', {
        'E_x': float,
        'E_y': float,
        'nu_x': float,
        'nu_y': float,
        'bulk': float
    }
)

New_Element = TypedDict(
    'New_Element',
    {
        'nodes': Union[str, List[int]],  # "all" or [node1, node2, node3, ...]
        't': float,  # thickness
        'mat': str  # "mat_name"
    }
)

New_Spring = TypedDict(
    'New_Spring',
    {
        'node': int,  # node # 
        'k_x': float,  # x stiffness
        'k_y': float,  # y stiffness
        'k_z': float,  # z stiffness
        'k_q': float,  # q stiffness
        'k_type': Literal["foundation", "total",
                          "node_pair"],  # "foundation"|"total"|"node_pair" - stiffness type,
        'node_pair': int,  # node number to which to pair (if relevant)
        'discrete': bool,
        'y': float,  # location of discrete spring
    }
)

New_Constraint = TypedDict(
    'New_Constraint',
    {
        'elim_node': int,  # node #
        'elim_dof': Literal["x", "y", "z", "q"],  # "q" is the twist dof 
        'coeff': float,  # elim_dof = coeff * keep_dof
        'keep_node': int,  # node #
        'keep_dof': Literal["x", "y", "z", "q"],  # "q" is the twist dof
    }
)

New_Node_Props = TypedDict(
    'New_Node_Props',
    {
        'dof_x': bool,  # defaults to True
        'dof_y': bool,  # defaults to True
        'dof_z': bool,  # defaults to True
        'dof_q': bool,  # defaults to True
        'stress': float,  # defaults to 0.0
    }
)

Analysis_Config = TypedDict(
    'Analysis_Config',
    {
        'b_c': B_C,  # boundary condition type
        'n_eigs': int,  # number of eigenvalues to consider
    }
)

Cfsm_Config = TypedDict(
    'Cfsm_Config',
    {
        'glob_modes': List[Literal[0, 1]],  # list of 1's (inclusion) and 0's (exclusion)
        'dist_modes': List[Literal[0, 1]],  # list of 1's (inclusion) and 0's (exclusion)
        'local_modes': List[Literal[0, 1]],  # list of 1's (inclusion) and 0's (exclusion)
        'other_modes': List[Literal[0, 1]],  # list of 1's (inclusion) and 0's (exclusion)
        'null_space': Literal["ST", "k_global", "kg_global", "vector"],
        'normalization': Literal["none", "vector", "strain_energy", "work"],
        'coupled': bool,  # coupled basis vs uncoupled basis for general B.C.
        'orthogonality': Literal["natural", "modal_axial", "modal_load"],  # natural or modal basis
    }
)