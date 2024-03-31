from operator import itemgetter
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.io import loadmat  # type: ignore

from pycufsm.types import (
    BC,
    Analysis_Config,
    ArrayLike,
    Cfsm_Config,
    Cufsm_MAT_File,
    GBT_Con,
    New_Constraint,
    New_Element,
    New_Node_Props,
    New_Props,
    New_Spring,
    PyCufsm_Input,
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
        n_1 = 1 - 3 * x_d * x_d + 2 * x_d * x_d * x_d
        n_2 = x_d * length * (1 - 2 * x_d + x_d**2)
        n_3 = 3 * x_d**2 - 2 * x_d**3
        n_4 = x_d * length * (x_d**2 - x_d)
        n_matrix = np.array(
            [[(1 - x_d), 0, x_d, 0, 0, 0, 0, 0], [0, (1 - x_d), 0, x_d, 0, 0, 0, 0], [0, 0, 0, 0, n_1, n_2, n_3, n_4]]
        )
        disp_local[:, i] = np.dot(n_matrix, disp).reshape(3)
    return disp_local


def lengths_recommend(
    nodes: np.ndarray, elements: np.ndarray, length_append: Optional[float] = None, n_lengths: int = 50
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
    min_el_length = 1000000  # Minimum element length
    max_el_length = 0  # Maximum element length
    min_el_thick = elements[0][3]  # Minimum element thickness
    for elem in elements:
        hh1 = abs(
            np.sqrt(
                (nodes[int(elem[1]), 1] - nodes[int(elem[2]), 1]) ** 2
                + (nodes[int(elem[1]), 2] - nodes[int(elem[2]), 2]) ** 2
            )
        )
        min_el_length = min(hh1, min_el_length)
        max_el_length = max(hh1, max_el_length)
        min_el_thick = min(elem[3], min_el_thick)

    lengths = np.logspace(np.log10(max(min_el_length, min_el_thick)), np.log10(1000 * max_el_length), num=n_lengths)

    if length_append is not None:
        lengths = np.sort(np.concatenate((lengths, np.array([length_append]))))

    return lengths


def load_cufsm_mat(mat_file: Optional[str] = None, mat_data: Optional[Cufsm_MAT_File] = None) -> PyCufsm_Input:
    """load Matlab CUFSM data from a MAT file

    Args:
        mat (Cufsm_MAT_File): dicionary of data read from a MAT file (as generated by
            scipy.io.loadmat())

    Returns:
        PyCufsm_Input: cleaned data ready for pyCUFSM input
    """
    if mat_file is not None and mat_data is None:
        mat_data = loadmat(file_name=mat_file)
    elif mat_data is not None and mat_file is not None:
        raise ValueError("Either 'mat_file' or 'mat_data' may be passed, but not both")
    assert mat_data is not None

    cufsm_input: PyCufsm_Input = {
        "nodes": np.array([]),
        "elements": np.array([]),
        "lengths": np.array([]),
        "props": np.array([]),
        "constraints": np.array([]),
        "springs": np.array([]),
        "curve": np.array([]),
        "shapes": np.array([]),
        "clas": "",
        "GBTcon": {"glob": [], "dist": [], "local": [], "other": [], "o_space": 0, "norm": 0, "couple": 0, "orth": 0},
    }
    if "node" in mat_data:
        nodes = np.array(mat_data["node"], dtype=np.dtype(np.double))
        for i in range(len(nodes)):
            nodes[i, 0] = int(np.double(nodes[i, 0])) - 1
            nodes[i, 3] = int(np.double(nodes[i, 3]))
            nodes[i, 4] = int(np.double(nodes[i, 4]))
            nodes[i, 5] = int(np.double(nodes[i, 5]))
            nodes[i, 6] = int(np.double(nodes[i, 6]))
            # nodes[i, 7] = 0
            cufsm_input["nodes"] = np.array(nodes)
    if "elem" in mat_data:
        elements = np.array(mat_data["elem"])
        for i in range(len(elements)):
            elements[i, 0] = int(np.double(elements[i, 0])) - 1
            elements[i, 1] = int(np.double(elements[i, 1])) - 1
            elements[i, 2] = int(np.double(elements[i, 2])) - 1
            cufsm_input["elements"] = np.array(elements)
    if "lengths" in mat_data:
        cufsm_input["lengths"] = np.array(mat_data["lengths"]).T
    if "prop" in mat_data:
        cufsm_input["props"] = np.array(mat_data["prop"])
    if "constraints" in mat_data:
        constraints = np.array(mat_data["constraints"])
        if len(constraints[0]) > 5:
            for i, constraint_row in enumerate(constraints):
                for j, constraint in enumerate(constraint_row):
                    if j < 5 and j != 2:
                        constraints[i, j] = int(constraint) - 1
        if len(constraints[0]) < 5:
            constraints = np.array([])
        cufsm_input["constraints"] = constraints
    if "springs" in mat_data:
        springs = np.array(mat_data["springs"])
        if len(springs[0]) < 4:
            springs = np.array([])
        cufsm_input["springs"] = springs
    if "curve" in mat_data:
        cufsm_input["curve"] = np.array(mat_data["curve"])
    if "GBTcon" in mat_data:
        gbt_con: GBT_Con = {
            "glob": (
                mat_data["GBTcon"]["glob"].flatten()[0].flatten() if "glob" in mat_data["GBTcon"].dtype.names else [0]
            ),
            "dist": (
                mat_data["GBTcon"]["dist"].flatten()[0].flatten() if "dist" in mat_data["GBTcon"].dtype.names else [0]
            ),
            "local": (
                mat_data["GBTcon"]["local"].flatten()[0].flatten() if "local" in mat_data["GBTcon"].dtype.names else [0]
            ),
            "other": (
                mat_data["GBTcon"]["other"].flatten()[0].flatten() if "other" in mat_data["GBTcon"].dtype.names else [0]
            ),
            "o_space": mat_data["GBTcon"]["o_space"] if "o_space" in mat_data["GBTcon"].dtype.names else 1,
            "norm": mat_data["GBTcon"]["norm"] if "norm" in mat_data["GBTcon"].dtype.names else 1,
            "couple": mat_data["GBTcon"]["couple"] if "couple" in mat_data["GBTcon"].dtype.names else 1,
            "orth": mat_data["GBTcon"]["orth"] if "orth" in mat_data["GBTcon"].dtype.names else 1,
        }
        cufsm_input["GBTcon"] = gbt_con
    if "shapes" in mat_data:
        cufsm_input["shapes"] = np.array(mat_data["shapes"])
    if "clas" in mat_data:
        cufsm_input["clas"] = mat_data["clas"]
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
    cfsm_config: Optional[Cfsm_Config] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, GBT_Con, BC, np.ndarray, int]:
    """Converts new format of inputs to old (original CUFSM) format

    Args:
        props (Dict[str, Dict[str, float]]): Dictionary of named materials and their properties
            {"mat_name": {E_x: float, E_y: float, nu_x: float, nu_y: float, bulk: float}}
        nodes (ArrayLike): 2D array of nodal coordinates
            [[x, y]] or [[x, y, stress]]
            Note that any node numbers used in later inputs refer to the index of the node in
            this 'nodes' array, with the first node being node 0.
            If array only has 2 columns, then stress is assumed to be 0.0 initially.
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
        if "E" in mat and "nu" in mat:
            props_old.append([i, mat["E"], mat["E"], mat["nu"], mat["nu"], mat["E"] / (2 * (1 + mat["nu"]))])
        elif "E_x" in mat and "E_y" in mat and "nu_x" in mat and "nu_y" in mat and "bulk" in mat:
            props_old.append([i, mat["E_x"], mat["E_y"], mat["nu_x"], mat["nu_y"], mat["bulk"]])
        else:
            raise TypeError(
                f"'props' must EITHER be a dictionary with 'E' and 'nu' keys, or "
                f"with 'E_x', 'E_y', 'nu_x', 'nu_y', and 'bulk'. The dictionary "
                f"passed for material {mat_name} includes {mat.keys()}"
            )

    # Convert nodes
    nodes_old: list = []
    nodes = np.array(nodes)
    if node_props is None:
        node_props = {}
    # Add a column of zeros, for zero stress, if no stress was set
    if np.shape(nodes)[1] == 2:
        nodes = np.c_[nodes, np.zeros(len(nodes))]
    if "all" in node_props:
        dof_x = int(node_props["all"]["dof_x"]) if "dof_x" in node_props["all"] else 1
        dof_y = int(node_props["all"]["dof_y"]) if "dof_y" in node_props["all"] else 1
        dof_z = int(node_props["all"]["dof_z"]) if "dof_z" in node_props["all"] else 1
        dof_q = int(node_props["all"]["dof_q"]) if "dof_q" in node_props["all"] else 1
    for i, node in enumerate(nodes):
        if "all" in node_props:
            nodes_old.append([i, node[0], node[1], dof_x, dof_y, dof_z, dof_q, node[2]])
        elif i in node_props:
            dof_x = int(node_props[i]["dof_x"]) if "dof_x" in node_props[i] else 1
            dof_y = int(node_props[i]["dof_y"]) if "dof_y" in node_props[i] else 1
            dof_z = int(node_props[i]["dof_z"]) if "dof_z" in node_props[i] else 1
            dof_q = int(node_props[i]["dof_q"]) if "dof_q" in node_props[i] else 1
            nodes_old.append([i, node[0], node[1], dof_x, dof_y, dof_z, dof_q, node[2]])
        else:
            nodes_old.append([i, node[0], node[1], 1, 1, 1, 1, node[2]])

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
    if isinstance(lengths, (set, list, np.ndarray, tuple)):
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
        springs_old.append(
            [
                spring["node"],
                spring["node_pair"] if spring["k_type"] == "node_pair" else -1,
                spring["k_x"],
                spring["k_y"],
                spring["k_z"],
                spring["k_q"],
                0 if spring["k_type"] == "foundation" else 1,
                int(spring["discrete"]),
                spring["y"],
            ]
        )

    # Convert constraints
    dof_conv = {"x": 1, "y": 2, "z": 3, "q": 4}
    constraints_old: list = []
    if constraints is None:
        constraints = []
    for constraint in constraints:
        constraints_old.append(
            [
                constraint["elim_node"],
                dof_conv[constraint["elim_dof"]],
                constraint["coeff"],
                constraint["keep_node"],
                dof_conv[constraint["keep_dof"]],
            ]
        )

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
            "orth": orth_conv[cfsm_config["orthogonality"]],
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

    return (
        np.array(props_old),
        np.array(nodes_old),
        np.array(elements_old),
        np.array(lengths_old),
        np.array(springs_old),
        np.array(constraints_old),
        gbt_con_old,
        b_c_old,
        np.array(m_all_old),
        n_eigs_old,
    )


def inputs_old_to_new(
    props: ArrayLike,
    nodes: ArrayLike,
    elements: ArrayLike,
    lengths: ArrayLike,
    springs: ArrayLike,
    constraints: ArrayLike,
    gbt_con: GBT_Con,
    b_c: BC,
    m_all: ArrayLike,
    n_eigs: int,
) -> Tuple[
    Dict[str, New_Props],
    np.ndarray,
    List[New_Element],
    Dict[float, np.ndarray],
    Dict[Union[Literal["all"], int], New_Node_Props],
    List[New_Spring],
    List[New_Constraint],
    Analysis_Config,
    Cfsm_Config,
]:
    """Converts old (original CUFSM) format of inputs to new format

    Args:
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

    Returns:
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
    """
    # Convert props
    props_new: Dict[str, New_Props] = {}
    for mat in props:
        props_new[f"mat #{mat[0]}"] = {"E_x": mat[1], "E_y": mat[2], "nu_x": mat[3], "nu_y": mat[4], "bulk": mat[5]}

    # Convert nodes
    nodes = np.array(sorted(nodes, key=itemgetter(0)))  # sort by node #
    nodes_new: np.ndarray = np.c_[nodes[:, 1:2], nodes[:, 7]]
    node_props_new: Dict[Union[Literal["all"], int], New_Node_Props] = {}
    node_index: Dict[int, int] = {}
    for i, node in enumerate(nodes):
        node_index[node[0]] = i
        if any(node[3:6] == 0):
            node_props_new[i] = {
                "dof_x": node[3],
                "dof_y": node[4],
                "dof_z": node[5],
                "dof_q": node[6],
            }

    # Convert elements
    elements = np.array(sorted(elements, key=itemgetter(0)))  # sort by elem #
    elements_new: List[New_Element] = [
        {
            "nodes": [node_index[elements[0, 1]], node_index[elements[0, 2]]],
            "t": elements[0, 3],
            "mat": f"mat #{elements[0,4]}",
        }
    ]
    i = 1
    while i < len(elements):
        while elements[i - 1, 2] == elements[i, 1] and i < len(elements):
            elements_new[-1]["nodes"].append(elements[i, 2])  # type: ignore
            i += 1
        if i != len(elements) - 1:
            elements_new.append(
                {
                    "nodes": [node_index[elements[i, 1]], node_index[elements[i, 2]]],
                    "t": elements[i, 3],
                    "mat": f"mat #{elements[i,4]}",
                }
            )
        i += 1
    if len(elements_new) == 1:
        elements_new[0]["nodes"] = "all"

    # Convert lengths and m_all
    lengths_new: Dict[float, np.ndarray] = {}
    for i, length in enumerate(lengths):
        lengths_new[length] = np.array(m_all[i])

    # Convert springs
    springs_new: List[New_Spring] = []
    k_type: Literal["foundation", "total", "node_pair"]
    for spring in springs:
        if spring[1] != -1:
            k_type = "node_pair"
        elif spring[6] == 1:
            k_type = "total"
        elif spring[6] == 0:
            k_type = "foundation"
        else:
            raise ValueError(
                "Unknown spring type; either a node pair must be set, or " + "spring must be set as a foundation spring"
            )
        springs_new.append(
            {
                "node": node_index[spring[0]],
                "node_pair": node_index[spring[1]],
                "k_x": spring[2],
                "k_y": spring[3],
                "k_z": spring[4],
                "k_q": spring[5],
                "k_type": k_type,
                "discrete": bool(spring[7]),
                "y": spring[8],
            }
        )

    # Convert constraints
    dof_conv: Dict[int, Literal["x", "y", "z", "q"]] = {1: "x", 2: "y", 3: "z", 4: "q"}
    constraints_new: List[New_Constraint] = []
    for constraint in constraints:
        constraints_new.append(
            {
                "elim_node": node_index[constraint[0]],
                "elim_dof": dof_conv[constraint[1]],
                "coeff": constraint[2],
                "keep_node": node_index[constraint[3]],
                "keep_dof": dof_conv[constraint[4]],
            }
        )

    # Convert configurations
    analysis_config: Analysis_Config = {
        "b_c": b_c,
        "n_eigs": n_eigs,
    }

    o_space_conv: Dict[int, Literal["ST", "k_global", "kg_global", "vector"]] = {
        1: "ST",
        2: "k_global",
        3: "kg_global",
        4: "vector",
    }
    norm_conv: Dict[int, Literal["none", "vector", "strain_energy", "work"]] = {
        0: "none",
        1: "vector",
        2: "strain_energy",
        3: "work",
    }
    orth_conv: Dict[int, Literal["natural", "modal_axial", "modal_load"]] = {
        1: "natural",
        2: "modal_axial",
        3: "modal_load",
    }
    cfsm_config: Cfsm_Config = {
        "glob_modes": gbt_con["glob"],
        "dist_modes": gbt_con["dist"],
        "local_modes": gbt_con["local"],
        "other_modes": gbt_con["other"],
        "null_space": o_space_conv[gbt_con["o_space"]],
        "normalization": norm_conv[gbt_con["norm"]],
        "coupled": gbt_con["couple"] == 2,
        "orthogonality": orth_conv[gbt_con["orth"]],
    }

    return (
        props_new,
        nodes_new,
        elements_new,
        lengths_new,
        node_props_new,
        springs_new,
        constraints_new,
        analysis_config,
        cfsm_config,
    )
