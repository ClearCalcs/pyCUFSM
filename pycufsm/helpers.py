from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

import pycufsm.cfsm
import pycufsm.fsm
from pycufsm.types import (
    B_C, Analysis_Config, ArrayLike, Cfsm_Config, Cufsm_MAT_File, GBT_Con, New_Constraint,
    New_Element, New_Node_Props, New_Spring, PyCufsm_Input, Sect_Props
)

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
    i_b_c: B_C = 'S-S'
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


def inputs_new_to_old(
    props: Dict[str, Dict[str, float]],
    nodes: ArrayLike,
    elements: Sequence[New_Element],
    lengths: Union[ArrayLike, set, Dict[float, ArrayLike]],
    node_props: Optional[Dict[Union[Literal["all"], int], New_Node_Props]] = None,
    springs: Optional[Sequence[New_Spring]] = None,
    constraints: Optional[Sequence[New_Constraint]] = None,
    analysis_config: Optional[Analysis_Config] = None,
    cfsm_config: Optional[Cfsm_Config] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, GBT_Con, B_C,
           np.ndarray, int]:
    """Converts new format of inputs to old (original CUFSM) format

    Args:
        props (Dict[str, Dict[str, float]]): Dictionary of named materials and their properties
            {"mat_name": {E_x: float, E_y: float, nu_x: float, nu_y: float, bulk: float}}
        nodes (ArrayLike): 2D array of nodal coordinates
            [[x, y]]
            Note that any node numbers used in later inputs refer to the index of the node in
            this 'nodes' array, with the first node being node 0. 
        elements (Sequence[New_Element]): Element connectivity and properties 
            [{
                nodes: "all"|List[int], # "all" or [node1, node2, node3, ...] in sequence
                t: float,               # thickness
                mat: str                # "mat_name"
            )].
            nodes: "all" is a special indicator that all nodes should be connected sequentially
            If nodes is given as an array, any number of nodal indices may be listed in order
        lengths (Union[ArrayLike, set, Dict[float, ArrayLike]): Half-wavelengths for analysis
            [length1, length2, ...]     # lengths only (assumes [m_a] = [1] for each length) 
            {length: list[int]}         # length: [m_a]
        node_props (Optional[Dict[Union[Literal["all", int], New_Node_Props]]): DOF restrictions 
            and stresses on nodes.
            {
                node_#|"all": {
                    dof_x: bool,    # defaults to True (included / not constrained)
                    dof_y: bool,    # defaults to True (included / not constrained)
                    dof_z: bool,    # defaults to True (included / not constrained)
                    dof_q: bool,    # defaults to True (included / not constrained)
                    stress: float   # defaults to 0.0 (no stress)
                }
            }
            Defaults to None, and taken as all DOFs included and all 0.0 stresses if so
        springs (Optional[Sequence[New_Spring]]): Definition of any springs in cross-section
            [{
                node: int,      # node # 
                k_x: float,     # x stiffness
                k_y: float,     # y stiffness
                k_z: float,     # z stiffness
                k_q: float,     # q stiffness
                k_type: str,    # "foundation"|"node_pair"|"total" - stiffness type
                node_pair: int, # node # to which to pair (if relevant)
                discrete: bool, # whether spring is at a discrete location
                y: float,       # location of discrete spring
            }]
            Defaults to None.
        constraints (Optional[Sequence[New_Constraint]]): Definition of any constraints in section
            [{
                elim_node: int,     # node #
                elim_dof: str,      # "x"|"y"|"z"|"q" - "q" is the twist dof 
                coeff: float,       # elim_dof = coeff * keep_dof
                keep_node: int,     # node #
                keep_dof: str       # "x"|"y"|"z"|"q" - "q" is the twist dof
            }]
            Defaults to None.
        analysis_config (Optional[Analysis_Config]): Configuration options for any analysis
            {
                b_c: str,           # "S-S"|"C-C"|"S-C"|"C-F"|"C-G" - boundary condition type
                n_eigs: int         # number of eigenvalues to consider
            }
            Defaults to None, and taken as {b_c: "S-S", n_eigs: 10} if so.
            Boundary condition types (at loaded edges):
                'S-S' simple-simple
                'C-C' clamped-clamped
                'S-C' simple-clamped
                'C-F' clamped-free
                'C-G' clamped-guided
        cfsm_config (Optional[Cfsm_Config]): Configuration options for cFSM (constrained modes) 
            {
                glob_modes: list(int),      # list of 1's (inclusion) and 0's (exclusion) for 
                    each mode from the analysis
                dist_modes: list(int),      # list of 1's (inclusion) and 0's (exclusion) for 
                    each mode from the analysis
                local_modes: list(int),     # list of 1's (inclusion) and 0's (exclusion) for 
                    each mode from the analysis
                other_modes: list(int),     # list of 1's (inclusion) and 0's (exclusion) for 
                    each mode from the analysis
                null_space: str,            # "ST"|"k_global"|"kg_global"|"vector"
                normalization: str,         # "none"|"vector"|"strain_energy"|"work"
                coupled: bool,              # coupled basis vs uncoupled basis for general B.C.
                orthogonality: str          # "natural"|"modal_axial"|"modal_load" - natural or 
                    modal basis
            }
            Defaults to None, in which case no cFSM analysis is performed
            null_space:
                "ST": ST basis
                "k_global": null space of GDL with respect to k_global
                "kg_global": null space of GDL with respect to kg_global
                "vector": null space of GDL in vector sense
            coupled:        basis for general B.C. especially for non-simply supported B.C.s
                uncoupled basis = the basis will be block diagonal
                coupled basis = the basis is fully spanned
            orthogonality:      natural basis vs modal basis
                "natural": natural basis
                "modal_axial": modal basis, axial orthogonality
                "modal_load": modal basis, load dependent orthogonality

    Returns:
        props (np.ndarray): [mat_num stiff_x stiff_y nu_x nu_y bulk] 6 x n_mats
        nodes (np.ndarray): [node# x y dof_x dof_y dof_z dof_r stress] n_nodes x 8
        elements (np.ndarray): [elem# node_i node_j thick mat_num] n_elements x 5
        lengths (np.ndarray): [L1 L2 L3...] 1 x n_lengths lengths to be analyzed; 
            could be half-wavelengths for signature curve or physical lengths for general b.c.
        springs (np.ndarray): [node# d.o.f. k_spring k_flag] where 1=x dir 2= y dir 3 = z dir 
            4 = q dir (twist) flag says if k_stiff is a foundation stiffness or a total stiffness
        constraints (np.ndarray): [node# e dof_e coeff node# k dof_k] e=dof to be eliminated
            k=kept dof dof_e_node = coeff*dof_k_node_k
        gbt_con (GBT_Con): gbt_con.glob,gbt_con.dist, gbt_con.local, gbt_con.other vectors of 1's
            and 0's referring to the inclusion (1) or exclusion of a given mode from the analysis,
            gbt_con.o_space - choices of ST/O mode
                    1: ST basis
                    2: O space (null space of GDL) with respect to k_global
                    3: O space (null space of GDL) with respect to kg_global
                    4: O space (null space of GDL) in vector sense
            gbt_con.norm - code for normalization (if normalization is done at all)
                    0: no normalization,
                    1: vector norm
                    2: strain energy norm
                    3: work norm
            gbt_con.couple - coupled basis vs uncoupled basis for general
                        B.C. especially for non-simply supported B.C.
                    1: uncoupled basis, the basis will be block diagonal
                    2: coupled basis, the basis is fully spanned
            gbt_con.orth - natural basis vs modal basis
                    1: natural basis
                    2: modal basis, axial orthogonality
                    3: modal basis, load dependent orthogonality
        b_c (str): ['S-S'] a string specifying boundary conditions to be analyzed:
            'S-S' simply-pimply supported boundary condition at loaded edges
            'C-C' clamped-clamped boundary condition at loaded edges
            'S-C' simply-clamped supported boundary condition at loaded edges
            'C-F' clamped-free supported boundary condition at loaded edges
            'C-G' clamped-guided supported boundary condition at loaded edges
        m_all (np.ndarray): m_all{length#}=[longitudinal_num# ... longitudinal_num#],
            longitudinal terms m for all the lengths in cell notation
            each cell has a vector including the longitudinal terms for this length
        n_eigs (int): the number of eigenvalues to be determined at length (default=10)
        sect_props (Sect_Props): _description_
    """
    # Convert props
    props_old: list = []
    mat_index: Dict[str, int] = {}
    i: int = 0
    for mat_name, mat in props.items():
        mat_index[mat_name] = i
        if 'E' in mat and 'nu' in mat:
            props_old.append([
                i, mat["E"], mat["E"], mat["nu"], mat["nu"], mat["E"] / (2 * (1 + mat["nu"]))
            ])
        elif 'E_x' in mat and 'E_y' in mat and 'nu_x' in mat and 'nu_y' in mat and 'bulk' in mat:
            props_old.append([i, mat["E_x"], mat["E_y"], mat["nu_x"], mat["nu_y"], mat["bulk"]])
        else:
            raise TypeError(
                f"'props' must EITHER be a dictionary with 'E' and 'nu' keys, or "
                f"with 'E_x', 'E_y', 'nu_x', 'nu_y', and 'bulk'. The dictionary "
                f"passed for material {mat_name} includes {mat.keys()}"
            )

    # Convert nodes
    nodes_old: list = []
    if node_props is None:
        node_props = {}
    if "all" in node_props:
        dof_x = int(node_props["all"]["dof_x"]) if "dof_x" in node_props["all"] else 1
        dof_y = int(node_props["all"]["dof_y"]) if "dof_y" in node_props["all"] else 1
        dof_z = int(node_props["all"]["dof_z"]) if "dof_z" in node_props["all"] else 1
        dof_q = int(node_props["all"]["dof_q"]) if "dof_q" in node_props["all"] else 1
        stress = node_props["all"]["stress"] if "stress" in node_props["all"] else 0.0
    for i, node in enumerate(nodes):
        if "all" in node_props:
            nodes_old.append([i, node[0], node[1], dof_x, dof_y, dof_z, dof_q, stress])
        elif i in node_props:
            dof_x = int(node_props[i]["dof_x"]) if "dof_x" in node_props[i] else 1
            dof_y = int(node_props[i]["dof_y"]) if "dof_y" in node_props[i] else 1
            dof_z = int(node_props[i]["dof_z"]) if "dof_z" in node_props[i] else 1
            dof_q = int(node_props[i]["dof_q"]) if "dof_q" in node_props[i] else 1
            stress = node_props[i]["stress"] if "stress" in node_props[i] else 0.0
            nodes_old.append([i, node[0], node[1], dof_x, dof_y, dof_z, dof_q, stress])
        else:
            nodes_old.append([i, node[0], node[1], 1, 1, 1, 1, 0.0])

    # Convert elements
    elements_old: list = []
    for i, elem in enumerate(elements):
        if isinstance(elem["nodes"], str) and elem["nodes"] == "all":
            elem["nodes"] = list(range(len(nodes)))
        for node1, node2 in zip(elem["nodes"][0:], elem["nodes"][1:]):
            elements_old.append([i, node1, node2, elem["t"], mat_index[elem["mat"]]])

    # Convert lengths and m_all
    lengths_old: List[float] = []
    m_all_old: List[List[int]] = []
    if isinstance(lengths, set) or isinstance(lengths, list) or isinstance(
            lengths, np.ndarray) or isinstance(lengths, tuple):
        for length in lengths:
            lengths_old.append(length)
            m_all_old.append([1])
    elif isinstance(lengths, dict):
        for length, m_a in lengths.items():
            lengths_old.append(length)
            m_all_old.append(list(m_a))

    # Convert springs
    springs_old: list = []
    if springs is None:
        springs = []
    for spring in springs:
        springs_old.append([
            spring["node"], spring["node_pair"] if spring["k_type"] == "node_pair" else -1,
            spring["k_x"], spring["k_y"], spring["k_z"], spring["k_q"],
            0 if spring["k_type"] == "foundation" else 1,
            int(spring["discrete"]), spring["y"]
        ])

    # Convert constraints
    dof_conv = {"x": 1, "y": 2, "z": 3, "q": 4}
    constraints_old: list = []
    if constraints is None:
        constraints = []
    for constraint in constraints:
        constraints_old.append([
            constraint["elim_node"], dof_conv[constraint["elim_dof"]], constraint["coeff"],
            constraint["keep_node"], dof_conv[constraint["keep_dof"]]
        ])

    # Convert configurations
    if analysis_config is not None:
        b_c_old = analysis_config["b_c"]
        n_eigs_old = analysis_config["n_eigs"]
    else:
        b_c_old = "S-S"
        n_eigs_old = 10
    if cfsm_config is not None:
        o_space_conv = {"ST": 1, "k_global": 2, "kg_global": 3, "vector": 4}
        norm_conv = {"none": 0, "vector": 1, "strain_energy": 2, "work": 3}
        orth_conv = {"natural": 1, "modal_axial": 2, "modal_load": 3}
        gbt_con_old: GBT_Con = {
            "glob": cfsm_config["glob_modes"],
            "dist": cfsm_config["dist_modes"],
            "local": cfsm_config["local_modes"],
            "other": cfsm_config["other_modes"],
            "o_space": o_space_conv[cfsm_config["null_space"]],
            "norm": norm_conv[cfsm_config["normalization"]],
            "couple": 2 if cfsm_config["coupled"] else 1,
            "orth": orth_conv[cfsm_config["orthogonality"]]
        }
    else:
        gbt_con_old = {
            "glob": [0],
            "dist": [0],
            "local": [0],
            "other": [0],
            "o_space": 1,
            "norm": 0,
            "couple": 1,
            "orth": 2,
        }

    return np.array(props_old), np.array(nodes_old), np.array(elements_old), np.array(
        lengths_old
    ), np.array(springs_old), np.array(constraints_old
                                       ), gbt_con_old, b_c_old, np.array(m_all_old), n_eigs_old
