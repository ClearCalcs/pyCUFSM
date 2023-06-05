from typing import Optional, Tuple

import numpy as np

import pycufsm.cfsm
import pycufsm.fsm
from pycufsm.types import Cufsm_MAT_File, GBT_Con, PyCufsm_Input, Sect_Props

# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE, CPEng
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


def gammait2(phi: float, disp_local: np.ndarray) -> np.ndarray:
    """transform local displacements into global displacements

    Args:
        phi (float): angle
        disp_local (np.ndarray): local displacements

    Returns:
        np.ndarray: global displacements

    BWS, 1998
    """
    gamma = np.array([[np.cos(phi), 0, -np.sin(phi)], [0, 1, 0], [np.sin(phi), 0, np.cos(phi)]])
    return np.dot(np.linalg.inv(gamma), disp_local)  # type: ignore


def shapef(links: int, disp: np.ndarray, length: float) -> np.ndarray:
    """Apply displacements using shape function

    Args:
        links (int): the number of additional line segments used to show the disp shape
        disp (np.ndarray): the vector of nodal displacements
        length (float): the actual length of the element

    Returns:
        np.ndarray: applied displacements

    BWS, 1998
    """
    inc = 1 / (links)
    x_disps = np.linspace(inc, 1 - inc, links - 1)
    disp_local = np.zeros((3, len(x_disps)))
    for i, x_d in enumerate(x_disps):
        n_1 = 1 - 3*x_d*x_d + 2*x_d*x_d*x_d
        n_2 = x_d * length * (1 - 2*x_d + x_d**2)
        n_3 = 3 * x_d**2 - 2 * x_d**3
        n_4 = x_d * length * (x_d**2 - x_d)
        n_matrix = np.array([[(1 - x_d), 0, x_d, 0, 0, 0, 0, 0], [0, (1 - x_d), 0, x_d, 0, 0, 0, 0],
                             [0, 0, 0, 0, n_1, n_2, n_3, n_4]])
        disp_local[:, i] = np.dot(n_matrix, disp).reshape(3)
    return disp_local


def lengths_recommend(
    nodes: np.ndarray,
    elements: np.ndarray,
    length_append: Optional[float] = None,
    n_lengths: int = 50
) -> np.ndarray:
    """generate the signature curve solution, part 1: find recommended lengths

    Args:
        nodes (np.ndarray): standard parameter
        elements (np.ndarray): standard parameter
        length_append (Optional[float], optional): Any additional specific length to include 
            in the half-wavelengths. Defaults to None.
        n_lengths (int, optional): number of half-wavelengths to include. Defaults to 50.

    Returns:
        np.ndarray: recommended lengths

    Z. Li, July 2010 (last modified)
    Function split by B Smith; this part only finds recommended lengths
    """
    min_el_length = 1000000  #Minimum element length
    max_el_length = 0  #Maximum element length
    min_el_thick = elements[0][3]  #Minimum element thickness
    for elem in elements:
        hh1 = abs(
            np.sqrt((nodes[int(elem[1]), 1] - nodes[int(elem[2]), 1])**2
                    + (nodes[int(elem[1]), 2] - nodes[int(elem[2]), 2])**2)
        )
        min_el_length = min(hh1, min_el_length)
        max_el_length = max(hh1, max_el_length)
        min_el_thick = min(elem[3], min_el_thick)

    lengths = np.logspace(
        np.log10(max(min_el_length, min_el_thick)), np.log10(1000 * max_el_length), num=n_lengths
    )

    if length_append is not None:
        lengths = np.sort(np.concatenate((lengths, np.array([length_append]))))

    return lengths


def signature_ss(
    props: np.ndarray, nodes: np.ndarray, elements: np.ndarray, i_gbt_con: GBT_Con,
    sect_props: Sect_Props, lengths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """generate the signature curve solution, part 2: actually solve the signature curve

    Args:
        props (np.ndarray): standard parameter
        nodes (np.ndarray): standard parameter
        elements (np.ndarray): standard parameter
        i_gbt_con (GBT_Con): cFSM configuration options
        sect_props (Sect_Props): section properties
        lengths (np.ndarray): half-wavelengths

    Returns:
        signature: signature curve,
        curve: all the curve results,
        shapes: deformed shapes at each point
    
    Z. Li, July 2010 (last modified)
    """
    i_springs = np.array([])
    i_constraints = np.array([])
    i_b_c = 'S-S'
    i_m_all = np.ones((len(lengths), 1)).tolist()

    isignature, icurve, ishapes = pycufsm.fsm.strip(
        props=props,
        nodes=nodes,
        elements=elements,
        lengths=lengths,
        springs=i_springs,
        constraints=i_constraints,
        gbt_con=i_gbt_con,
        b_c=i_b_c,
        m_all=i_m_all,
        n_eigs=10,
        sect_props=sect_props,
    )

    return isignature, icurve, ishapes


def m_recommend(
    props: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    sect_props: Sect_Props,
    length_append: Optional[float] = None,
    n_lengths: int = 50,
    lengths: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Suggested longitudinal terms are calculated based on the characteristic
    half-wave lengths of local, distortional, and global buckling from the
    signature curve.

    Args:
        props (np.ndarray): standard parameter
        nodes (np.ndarray): standard parameter
        elements (np.ndarray): standard parameter
        sect_props (Sect_Props): section properties
        length_append (Optional[float], optional): any additional half-wavelength to include. 
            Defaults to None.
        n_lengths (int, optional): number of half-wavelengths. Defaults to 50.
        lengths (Optional[np.ndarray], optional): specific half-wavelengths to use. 
            Defaults to None.

    Returns:
        _type_: _description_
    
    Z. Li, Oct. 2010
    """
    i_gbt_con: GBT_Con = {
        "glob": [0],
        "dist": [0],
        "local": [0],
        "other": [0],
        "o_space": 1,
        "couple": 1,
        "orth": 1,
        "norm": 1,
    }
    if lengths is None:
        lengths = lengths_recommend(
            nodes=nodes, elements=elements, length_append=length_append, n_lengths=n_lengths
        )

    print("Running initial pyCUFSM signature curve")
    isignature, icurve, ishapes = signature_ss(
        props=props,
        nodes=nodes,
        elements=elements,
        i_gbt_con=i_gbt_con,
        sect_props=sect_props,
        lengths=lengths
    )

    curve_signature = np.zeros((len(lengths), 2))
    curve_signature[:, 0] = lengths.T
    curve_signature[:, 1] = isignature

    local_minima = []
    for i, c_sign in enumerate(curve_signature[:-2]):
        load1 = c_sign[1]
        load2 = curve_signature[i + 1, 1]
        load3 = curve_signature[i + 2, 1]
        if load2 < load1 and load2 <= load3:
            local_minima.append(curve_signature[i + 1, 0])

    _, _, _, _, _, _, n_dist_modes, n_local_modes, _ = pycufsm.cfsm.base_properties(
        nodes=nodes, elements=elements
    )

    n_global_modes = 4
    n_other_modes = 2 * (len(nodes) - 1)

    i_gbt_con["local"] = np.ones((n_local_modes, 1)).tolist()
    i_gbt_con["dist"] = np.zeros((n_dist_modes, 1)).tolist()
    i_gbt_con["glob"] = np.zeros((n_global_modes, 1)).tolist()
    i_gbt_con["other"] = np.zeros((n_other_modes, 1)).tolist()

    print("Running pyCUFSM local modes curve")
    isignature_local, icurve_local, ishapes_local = signature_ss(
        props=props,
        nodes=nodes,
        elements=elements,
        i_gbt_con=i_gbt_con,
        sect_props=sect_props,
        lengths=lengths
    )

    print("Running pyCUFSM distortional modes curve")
    i_gbt_con["local"] = np.zeros((n_local_modes, 1)).tolist()
    i_gbt_con["dist"] = np.ones((n_dist_modes, 1)).tolist()
    i_gbt_con["glob"] = np.zeros((n_global_modes, 1)).tolist()
    i_gbt_con["other"] = np.zeros((n_other_modes, 1)).tolist()
    isignature_dist, icurve_dist, ishapes_dist = signature_ss(
        props=props,
        nodes=nodes,
        elements=elements,
        i_gbt_con=i_gbt_con,
        sect_props=sect_props,
        lengths=lengths
    )

    curve_signature_local = np.zeros((len(lengths), 2))
    curve_signature_local[:, 0] = lengths
    curve_signature_local[:, 1] = isignature_local
    curve_signature_dist = np.zeros((len(lengths), 2))
    curve_signature_dist[:, 0] = lengths
    curve_signature_dist[:, 1] = isignature_dist

    #cFSM local half-wavelength
    local_minima_local = []
    for i, c_sign in enumerate(curve_signature_local[:-2]):
        load1 = c_sign[1]
        load2 = curve_signature_local[i + 1, 1]
        load3 = curve_signature_local[i + 2, 1]
        if load2 < load1 and load2 <= load3:
            local_minima_local.append(curve_signature_local[i + 1, 0])
    # If there were no local minima, then take the absolute minimum
    if len(local_minima_local) == 0:
        ind = np.argmin([val[1] for val in curve_signature_local])
        local_minima_local.append(curve_signature_local[ind, 0])

    #cFSM dist half-wavelength
    local_minima_dist = []
    for i, c_sign in enumerate(curve_signature_dist[:-2]):
        load1 = c_sign[1]
        load2 = curve_signature_dist[i + 1, 1]
        load3 = curve_signature_dist[i + 2, 1]
        if load2 < load1 and load2 <= load3:
            local_minima_dist.append(curve_signature_dist[i + 1, 0])
    # If there were no local minima, then take the absolute minimum
    if len(local_minima_dist) == 0:
        ind = np.argmin([val[1] for val in curve_signature_dist])
        local_minima_dist.append(curve_signature_dist[ind, 0])

    if len(local_minima) == 2:
        length_crl = local_minima[0]
        length_crd = local_minima[1]

    else:
        #half-wavelength of local and distortional buckling
        length_crl = local_minima_local[0]
        length_crd = local_minima_dist[0]

    #recommend longitudinal terms m
    im_pm_all = []
    for im_p_len in lengths:

        if np.ceil(im_p_len / length_crl) > 4:
            im_pm_all_temp = [
                np.ceil(im_p_len / length_crl) - 3,
                np.ceil(im_p_len / length_crl) - 2,
                np.ceil(im_p_len / length_crl) - 1,
                np.ceil(im_p_len / length_crl),
                np.ceil(im_p_len / length_crl) + 1,
                np.ceil(im_p_len / length_crl) + 2,
                np.ceil(im_p_len / length_crl) + 3,
            ]
        else:
            im_pm_all_temp = [1, 2, 3, 4, 5, 6, 7]

        if np.ceil(im_p_len / length_crd) > 4:
            im_pm_all_temp.extend([
                np.ceil(im_p_len / length_crd) - 3,
                np.ceil(im_p_len / length_crd) - 2,
                np.ceil(im_p_len / length_crd) - 1,
                np.ceil(im_p_len / length_crd),
                np.ceil(im_p_len / length_crd) + 1,
                np.ceil(im_p_len / length_crd) + 2,
                np.ceil(im_p_len / length_crl) + 3,
            ])
        else:
            im_pm_all_temp.extend([1, 2, 3, 4, 5, 6, 7])

        im_pm_all_temp.extend([1, 2, 3])

        im_pm_all.append(im_pm_all_temp)

    #m_a_recommend = analysis.m_sort(im_pm_all)
    m_a_recommend = np.array(im_pm_all)

    return (
        m_a_recommend,
        lengths,
        isignature,
        icurve,
        ishapes,
        length_crl,
        length_crd,
        isignature_local,
        icurve_local,
        ishapes_local,
        isignature_dist,
        icurve_dist,
        ishapes_dist,
    )


def load_mat(mat: Cufsm_MAT_File) -> PyCufsm_Input:
    """load Matlab CUFSM data from a MAT file

    Args:
        mat (Cufsm_MAT_File): dicionary of data read from a MAT file (as generated by 
            scipy.io.loadmat())

    Returns:
        PyCufsm_Input: cleaned data ready for pyCUFSM input
    """
    cufsm_input: PyCufsm_Input = {
        'nodes': np.array([]),
        'elements': np.array([]),
        'lengths': np.array([]),
        'props': np.array([]),
        'constraints': np.array([]),
        'springs': np.array([]),
        'curve': np.array([]),
        'shapes': np.array([]),
        'clas': '',
        'GBTcon': {
            'glob': [],
            'dist': [],
            'local': [],
            'other': [],
            'o_space': 0,
            'norm': 0,
            'couple': 0,
            'orth': 0
        }
    }
    if 'node' in mat:
        nodes = np.array(mat['node'], dtype=np.dtype(np.double))
        for i in range(len(nodes)):
            nodes[i, 0] = int(np.double(nodes[i, 0])) - 1
            nodes[i, 3] = int(np.double(nodes[i, 3]))
            nodes[i, 4] = int(np.double(nodes[i, 4]))
            nodes[i, 5] = int(np.double(nodes[i, 5]))
            nodes[i, 6] = int(np.double(nodes[i, 6]))
            #nodes[i, 7] = 0
            cufsm_input['nodes'] = np.array(nodes)
    if 'elem' in mat:
        elements = np.array(mat['elem'])
        for i in range(len(elements)):
            elements[i, 0] = int(np.double(elements[i, 0])) - 1
            elements[i, 1] = int(np.double(elements[i, 1])) - 1
            elements[i, 2] = int(np.double(elements[i, 2])) - 1
            cufsm_input['elements'] = np.array(elements)
    if 'lengths' in mat:
        cufsm_input['lengths'] = np.array(mat['lengths']).T
    if 'prop' in mat:
        cufsm_input['props'] = np.array(mat['prop'])
    if 'constraints' in mat:
        constraints = np.array(mat['constraints'])
        if len(constraints[0]) > 5:
            for i, constraint_row in enumerate(constraints):
                for j, constraint in enumerate(constraint_row):
                    if j < 5 and j != 2:
                        constraints[i, j] = int(constraint) - 1
        if len(constraints[0]) < 5:
            constraints = np.array([])
        cufsm_input['constraints'] = constraints
    if 'springs' in mat:
        springs = np.array(mat['springs'])
        if len(springs[0]) < 4:
            springs = np.array([])
        cufsm_input['springs'] = springs
    if 'curve' in mat:
        cufsm_input['curve'] = np.array(mat['curve'])
    if 'GBTcon' in mat:
        gbt_con: GBT_Con = {
            "glob":
                mat["GBTcon"]["glob"].flatten()[0].flatten()
                if "glob" in mat["GBTcon"].dtype.names else [0],
            "dist":
                mat["GBTcon"]["dist"].flatten()[0].flatten()
                if "dist" in mat["GBTcon"].dtype.names else [0],
            "local":
                mat["GBTcon"]["local"].flatten()[0].flatten()
                if "local" in mat["GBTcon"].dtype.names else [0],
            "other":
                mat["GBTcon"]["other"].flatten()[0].flatten()
                if "other" in mat["GBTcon"].dtype.names else [0],
            "o_space":
                mat["GBTcon"]["o_space"] if "o_space" in mat["GBTcon"].dtype.names else 1,
            "norm":
                mat["GBTcon"]["norm"] if "norm" in mat["GBTcon"].dtype.names else 1,
            "couple":
                mat["GBTcon"]["couple"] if "couple" in mat["GBTcon"].dtype.names else 1,
            "orth":
                mat["GBTcon"]["orth"] if "orth" in mat["GBTcon"].dtype.names else 1
        }
        cufsm_input['GBTcon'] = gbt_con
    if 'shapes' in mat:
        cufsm_input['shapes'] = np.array(mat['shapes'])
    if 'clas' in mat:
        cufsm_input['clas'] = mat['clas']
    return cufsm_input
