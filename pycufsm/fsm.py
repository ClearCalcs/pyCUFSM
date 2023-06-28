from copy import deepcopy
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg as spla  # type: ignore

import pycufsm.cfsm
from pycufsm.analysis import analysis
from pycufsm.helpers import inputs_new_to_old, m_recommend
from pycufsm.preprocess import stress_gen, yield_mp
from pycufsm.types import (
    B_C, Analysis_Config, ArrayLike, Cfsm_Config, Forces, GBT_Con, New_Constraint, New_Element,
    New_Node_Props, New_Spring, Sect_Props, Yield_Force
)

# from scipy.sparse.linalg import eigs
# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE, CPEng
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


def strip(
    props: np.ndarray, nodes: np.ndarray, elements: np.ndarray, lengths: np.ndarray,
    springs: np.ndarray, constraints: np.ndarray, gbt_con: GBT_Con, b_c: B_C, m_all: np.ndarray,
    n_eigs: int, sect_props: Sect_Props
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a finite strip analysis

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
        signature (np.ndarray): signature curve
        curve (np.ndarray): buckling curve (load factor) for each length
            curve[i] = [ length mode# 1
                        length mode# 2
                        ...    ...
                        length mode#]
        shapes (np.ndarray): mode shapes for each length
            shapes[i] = mode, mode is a matrix, each column corresponds to a mode.
    """

    n_nodes = len(nodes)
    curve = []
    shapes = []
    signature = np.zeros((len(lengths), 1))

    # CLEAN UP INPUT
    # clean u_j 0's, multiple terms. or out-of-order terms in m_all
    m_all = analysis.m_sort(m_all)

    # DETERMINE FLAGS FOR USER CONSTRAINTS AND INTERNAL (AT NODE) B.C.'s
    bc_flag = analysis.constr_bc_flag(nodes=nodes, constraints=constraints)

    # GENERATE STRIP WIDTH AND DIRECTION ANGLE
    el_props = analysis.elem_prop(nodes=nodes, elements=elements)

    # ENABLE cFSM ANALYSIS IF APPLICABLE, AND FIND BASE PROPERTIES
    if sum(gbt_con['glob']) + sum(gbt_con['dist']) \
                + sum(gbt_con['local']) + sum(gbt_con['other']) > 0:
        # turn on modal classification analysis
        cfsm_analysis = 1
        # set u_p stress to 1.0 for finding kg_global and k_global for axial modes
        nodes_base = deepcopy(nodes)
        nodes_base[:, 7] = np.ones_like(nodes[:, 7])

        # natural base first
        # properties all the longitudinal terms share
        [main_nodes, meta_elements, node_props, n_main_nodes, \
            n_corner_nodes, n_sub_nodes, n_dist_modes, n_local_modes, dof_perm] \
            = pycufsm.cfsm.base_properties(nodes=nodes_base, elements=elements)
        [r_x, r_z, r_yd, r_ys, r_ud] = pycufsm.cfsm.mode_constr(
            nodes=nodes_base,
            elements=elements,
            node_props=node_props,
            main_nodes=main_nodes,
            meta_elements=meta_elements
        )
        [d_y, n_global_modes] = pycufsm.cfsm.y_dofs(
            nodes=nodes_base,
            elements=elements,
            main_nodes=main_nodes,
            n_main_nodes=n_main_nodes,
            n_dist_modes=n_dist_modes,
            r_yd=r_yd,
            r_ud=r_ud,
            sect_props=sect_props,
            el_props=el_props
        )
    else:
        # no modal classification constraints are engaged
        cfsm_analysis = 0

    # LOOP OVER ALL THE LENGTHS TO BE INVESTIGATED
    for i, length in enumerate(lengths):
        # longitudinal terms to be included for this length
        m_a = m_all[i]

        total_m = len(m_a)  # Total number of longitudinal terms

        # SET SWITCH AND PREPARE BASE VECTORS (r_matrix) FOR cFSM ANALYSIS
        if cfsm_analysis == 1:
            # generate natural base vectors for axial compression loading
            b_v_l = pycufsm.cfsm.base_column(
                nodes_base=nodes_base,
                elements=elements,
                props=props,
                length=length,
                b_c=b_c,
                m_a=m_a,
                el_props=el_props,
                node_props=node_props,
                n_main_nodes=n_main_nodes,
                n_corner_nodes=n_corner_nodes,
                n_sub_nodes=n_sub_nodes,
                n_global_modes=n_global_modes,
                n_dist_modes=n_dist_modes,
                n_local_modes=n_local_modes,
                dof_perm=dof_perm,
                r_x=r_x,
                r_z=r_z,
                r_ys=r_ys,
                d_y=d_y
            )

        k_global, kg_global = analysis.k_kg_global(
            nodes=nodes,
            elements=elements,
            el_props=el_props,
            props=props,
            length=length,
            b_c=b_c,
            m_a=m_a
        )

        # ADD SPRING CONTRIBUTIONS TO STIFFNESS
        # Prior to version 4.3 the springs format was [node# dof k_stiffness k_type]
        #   where k_type indicated either a foundation or total stiffness
        # Now from version 4.3 this is the new springs method
        if len(springs) != 0:
            # springs variable exists
            for spring in springs:
                # Generate spring stiffness matrix (k_s) in local coordinates
                k_u = spring[3]
                k_v = spring[4]
                k_w = spring[5]
                k_q = spring[6]
                discrete = spring[8]
                y_s = spring[9] * length
                ks_l = analysis.spring_klocal(
                    k_u=k_u,
                    k_v=k_v,
                    k_w=k_w,
                    k_q=k_q,
                    length=length,
                    b_c=b_c,
                    m_a=m_a,
                    discrete=discrete,
                    y_s=y_s
                )

                # Transform k_s into global coordinates
                node_i = spring[1]
                node_j = spring[2]
                if node_j == -1 or spring[7] == 0:  # spring is to ground
                    # handle the spring to ground during assembly
                    alpha: float = 0  # use global coordinates for spring
                    node_j = -1
                else:  # spring is between nodes
                    x_i = nodes[node_i, 1]
                    y_i = nodes[node_i, 2]
                    x_j = nodes[node_j, 1]
                    y_j = nodes[node_j, 2]
                    d_x = x_j - x_i
                    d_y = y_j - y_i
                    width = np.sqrt(d_x**2 + d_y**2)
                    if width < 1E-10:  # coincident nodes
                        alpha = 0  # use global coordinates for spring
                    else:
                        # local orientation for spring
                        # np.arctan2() function is mis-typed in numpy - given floats,
                        # it DOES return a float.It does not always return an array
                        alpha = np.arctan2(d_y, d_x)  # type: ignore

                gamma = analysis.trans(alpha=alpha, total_m=total_m)
                k_s = gamma @ ks_l @ gamma.conj().T

                # Add element contribution of k_s to full matrix k_global
                k_global = analysis.spring_assemble(
                    k_global=k_global,
                    k_local=k_s,
                    node_i=node_i,
                    node_j=node_j,
                    n_nodes=n_nodes,
                    m_a=m_a
                )

        # INTERNAL BOUNDARY CONDITIONS (ON THE NODES) AND USER DEFINED CONSTR.
        # Check for user defined constraints too
        if bc_flag == 1:
            # size boundary conditions and user constraints for use in r_matrix format
            # d_constrained=r_user*d_unconstrained, d=nodal DOF vector (note by
            # BWS June 5 2006)
            r_user = pycufsm.cfsm.constr_user(nodes=nodes, constraints=constraints, m_a=m_a)
            r_u0_matrix = spla.null_space(r_user.conj().T)
            # Number of boundary conditions and user defined constraints = nu0
            nu0 = len(r_u0_matrix[0])

        # GENERATION OF cFSM CONSTRAINT MATRIX
        if cfsm_analysis == 1:
            # PERFORM ORTHOGONALIZATION IF GBT-LIKE MODES ARE ENFORCED
            b_v = pycufsm.cfsm.base_update(
                gbt_con=gbt_con,
                b_v_l=b_v_l,
                length=length,
                m_a=m_a,
                nodes=nodes,
                elements=elements,
                props=props,
                n_global_modes=n_global_modes,
                n_dist_modes=n_dist_modes,
                n_local_modes=n_local_modes,
                b_c=b_c,
                el_props=el_props
            )
            # no normalization is enforced: 0:  m
            # assign base vectors to constraints
            b_v = pycufsm.cfsm.mode_select(
                b_v=b_v,
                n_global_modes=n_global_modes,
                n_dist_modes=n_dist_modes,
                n_local_modes=n_local_modes,
                gbt_con=gbt_con,
                n_dof_m=4 * n_nodes,
                m_a=m_a
            )  # m
            r_mode = b_v
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else:
            # no modal constraints are activated therefore
            r_mode = np.eye(4 * n_nodes * total_m)  # activate modal constraints

        # CREATE FINAL CONSTRAINT MATRIX
        # Determine the number of modal constraints, nm0
        if bc_flag == 0:
            # if no user defined constraints and fixities.
            r_matrix = r_mode
        else:
            # should performed uncoupled for block diagonal basis?
            if cfsm_analysis == 1:
                nm0 = 0
                r_m0_matrix = spla.null_space(r_mode.conj().T)
                nm0 = len(r_m0_matrix[0])
                r_0_matrix = r_m0_matrix
                if nu0 > 0:
                    r_0_matrix[:, nm0:(nm0 + nu0)] = r_u0_matrix
                r_matrix = spla.null_space(r_0_matrix.conj().T)
            else:
                r_matrix = spla.null_space(r_u0_matrix.conj().T)

        # INTRODUCE CONSTRAINTS AND REDUCE k_global MATRICES TO FREE PARTS ONLY
        k_global_ff = r_matrix.transpose() @ k_global @ r_matrix
        kg_global_ff = r_matrix.transpose() @ kg_global @ r_matrix
        # SOLVE THE EIGENVALUE PROBLEM
        # Determine which solver to use
        # small problems usually use eig (dense matrix),
        # and large problems use eigs (sparse matrix).
        # the eigs solver is not as stable as the full eig solver...
        # LAPACK reciprocal condition estimator
        # rcond_num = 1 / np.linalg.cond(np.linalg.pinv(kg_global_ff) @ k_global_ff)

        # Here, assume when rcond_num is bigger than half of the eps, eigs can provide
        # reliable solution. Otherwise, eig, the robust solver should be used.
        # if rcond_num >= np.spacing(1.0) / 2:
        #     eig_sparse = True
        #     # eigs
        # else:
        #     eig_sparse = False
        #     # eig

        # if eig_sparse:
        #     k_eigs = max(min(2*n_eigs, len(k_global_ff)), 1)
        #     if k_eigs == 1 or k_eigs == len(k_global_ff):
        #         [length_factors, modes] = spla.eig(
        #             a=k_global_ff,
        #             b=kg_global_ff
        #         )
        #     else:
        #         # pull out 10 eigenvalues
        #         [length_factors, modes] = eigs(
        #             A=k_global_ff,
        #             k=k_eigs,
        #             M=kg_global_ff,
        #             which='SM'
        #         )
        # else:
        [length_factors, modes] = spla.eig(a=k_global_ff, b=kg_global_ff)
        # CLEAN UP THE EIGEN SOLUTION
        # eigenvalues are along the diagonal of matrix length_factors
        #length_factors = np.diag(length_factors)
        # find all the positive eigenvalues and corresponding vectors, squeeze out the rest
        index = np.logical_and(length_factors > 0, abs(np.imag(length_factors)) < 0.00001)
        length_factors = length_factors[index]
        modes = modes[:, index]
        # sort from small to large
        index = np.argsort(length_factors)
        length_factors = length_factors[index]
        modes = modes[:, index]
        # only the real part is of interest
        # (eigensolver may give some small nonzero imaginary parts)
        length_factors = np.real(length_factors)
        modes = np.real(modes)

        # truncate down to reasonable number of modes to be kept
        num_pos_modes = len(length_factors)
        n_modes = min([n_eigs, num_pos_modes])
        length_factors = length_factors[:n_modes]
        modes = modes[:, :n_modes]

        # FORM THE FULL MODE SHAPE BY BRINGING BACK ELIMINATED DOF
        modes_full = r_matrix @ modes

        # CLEAN UP NORMALIZATION OF MODE SHAPE
        # eig and eigs solver use different normalization
        # set max entry (absolute) to +1.0 and scale the rest

        max_vals = np.amax(abs(modes_full), axis=0)
        for j in range(0, n_modes):
            modes_full[:, j] = modes_full[:, j] / max_vals[j]
        # GENERATE OUTPUT VALUES
        # curve and shapes are changed to cells!!
        # curve: buckling curve (load factor)
        # curve{i} = [mode# 1 ... mode#]
        # shapes = mode shapes
        # shapes{i} = mode, each column corresponds to a mode.
        if len(length_factors) > 0:
            signature[i] = length_factors[0]
        curve.append(length_factors)
        # shapes(:,i,1:min([n_modes,num_pos_modes]))=modes
        shapes.append(modes_full)

    return signature.flatten(), np.array(curve), np.array(shapes)


def strip_new(
    props: Dict[str, Dict[str, float]],
    nodes: ArrayLike,
    elements: Sequence[New_Element],
    sect_props: Sect_Props,
    lengths: Optional[Union[ArrayLike, set, Dict[float, ArrayLike]]] = None,
    node_props: Optional[Dict[Union[Literal["all"], int], New_Node_Props]] = None,
    springs: Optional[Sequence[New_Spring]] = None,
    constraints: Optional[Sequence[New_Constraint]] = None,
    analysis_config: Optional[Analysis_Config] = None,
    cfsm_config: Optional[Cfsm_Config] = None,
    forces: Optional[Forces] = None,
    yield_force: Optional[Yield_Force] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        sect_props (Sect_Props): Dictionary of section properties, such as A, Ixx, Iyy, J, etc.
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
        yield_force (Optional[Yield_Force]): Single yield force to apply to section.
            Either this or 'forces' must be set.
            {
                force: "Mxx"|"Myy"|"M11"|"M22"|"P", # which force should cause yield
                direction: "Pos"|"Neg"|"+"|"-",     # direction to apply that force
                f_y: float,                         # yield strength of material (stress)
                restrain: bool,                     # whether section is restrained
                offset: ArrayLike                   # [x_offset, y_offset] of centroid
            }
            Note that 'restrain' only affects "Mxx" or "Myy" forces, and then only for sections
            in which the principal axes are no aligned with the geometric axes (such as Z sections)
        forces (Optional[Forces]): Specific forces to apply to the section.
            Either this or 'yield_force' must be set.
            {
                'P': float,         # axial force
                'Mxx': float,       # moment about x-x axis
                'Myy': float,       # moment about y-y axis
                'M11': float,       # moment about 1-1 (primary principal) axis
                'M22': float,       # moment about 2-2 axis
                'restrain': bool,   # whether section is restrained
                'offset': ArrayLike # [x_offset, y_offset] of centroid
            }

    Returns:
        signature (np.ndarray): signature curve
        curve (np.ndarray): buckling curve (load factor) for each length
            curve[i] = [ length mode# 1
                        length mode# 2
                        ...    ...
                        length mode#]
        shapes (np.ndarray): mode shapes for each length
            shapes[i] = mode, mode is a matrix, each column corresponds to a mode.
    """
    if lengths is None or isinstance(lengths, int):
        n_lengths = lengths if isinstance(lengths, int) else 50
        lengths = []

    (
        props_old, nodes_old, elements_old, lengths_old, springs_old, constraints_old, gbt_con_old,
        b_c_old, m_all_old, n_eigs_old
    ) = inputs_new_to_old(
        props=props,
        nodes=nodes,
        lengths=lengths,
        elements=elements,
        springs=springs,
        constraints=constraints,
        node_props=node_props,
        analysis_config=analysis_config,
        cfsm_config=cfsm_config
    )

    if forces is None and yield_force is not None:
        restrained = yield_force['restrain'] if 'restrain' in yield_force else False
        offset = yield_force['offset'] if 'offset' in yield_force else [0, 0]
        all_yields = yield_mp(
            nodes=nodes_old, f_y=yield_force['f_y'], sect_props=sect_props, restrained=restrained
        )
        forces = {
            "Mxx": 0,
            "Myy": 0,
            "M11": 0,
            "M22": 0,
            "P": 0,
            "restrain": restrained,
            "offset": offset
        }
        if yield_force['direction'] == '-' or yield_force['direction'] == 'Neg':
            multiplier = -1
        else:
            multiplier = 1
        forces[yield_force['force']] = all_yields[yield_force['force']] * multiplier
    elif forces is not None and yield_force is not None:
        raise ValueError("Only one of 'forces' or 'yield_force' may be set - but not both")
    elif forces is None and yield_force is None:
        raise ValueError("Either 'forces' or 'yield_force' must be set")
    assert forces is not None

    nodes_stressed = stress_gen(nodes=nodes_old, forces=forces, sect_props=sect_props)

    if len(lengths) == 0:
        m_all_old, lengths_old, signature, curve, shapes, _, _, _, _, _, _, _, _ = m_recommend(
            props=props_old,
            nodes=nodes_old,
            elements=elements_old,
            sect_props=sect_props,
            n_lengths=n_lengths
        )
    else:
        signature, curve, shapes = strip(
            props=props_old,
            nodes=nodes_stressed,
            elements=elements_old,
            lengths=lengths_old,
            springs=springs_old,
            constraints=constraints_old,
            gbt_con=gbt_con_old,
            b_c=b_c_old,
            m_all=m_all_old,
            n_eigs=n_eigs_old,
            sect_props=sect_props
        )

    return signature, curve, shapes, nodes_stressed, lengths_old
