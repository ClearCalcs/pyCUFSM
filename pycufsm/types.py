from typing import Any, List

import numpy as np
from typing_extensions import TypedDict

GBT_Con = TypedDict(
    'GBT_Con', {
        'glob': List[int],
        'dist': List[int],
        'local': List[int],
        'other': List[int],
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

MAT_File = TypedDict(
    'MAT_File',
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

Cufsm_Input = TypedDict(
    'Cufsm_Input', {
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
