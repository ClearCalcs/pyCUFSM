from copy import deepcopy
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg as spla  # type: ignore

import pycufsm.cfsm
from pycufsm.analysis import analysis
from pycufsm.helpers import inputs_new_to_old, lengths_recommend
from pycufsm.preprocess import stress_gen, yield_mp
from pycufsm.types import (
    BC,
    Analysis_Config,
    ArrayLike,
    Cfsm_Config,
    Forces,
    GBT_Con,
    New_Constraint,
    New_Element,
    New_Node_Props,
    New_Spring,
    Sect_Props,
    Yield_Force,
)

# from scipy.sparse.linalg import eigs
# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE, CPEng
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


def strip(
    props: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    lengths: np.ndarray,
    springs: np.ndarray,
    constraints: np.ndarray,
    GBT_con: GBT_Con,
    B_C: BC,
    m_all: np.ndarray,
    n_eigs: int,
    sect_props: Sect_Props,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a finite strip analysis

    Args:
        props (np.ndarray): Material properties
            | `[[mat_num, stiff_x, E_y, nu_x, nu_y, G_bulk], ...]`
        nodes (np.ndarray): Nodal properties
            | `[[node#, x, y, dof_x, dof_y, dof_z, dof_r, stress], ...]`
        elements (np.ndarray): Element properties
            | `[[elem#, node_i, node_j, thick, mat_num], ...]`
        lengths (np.ndarray): Half-wavelengths to analyse
            | `[length1, length2, ...]`
            These could be half-wavelengths for signature curve or physical lengths for general b.c.
        springs (np.ndarray): Nodal springs (if any)
            | `[[node#, node_pair, k_x, k_y, k_z, k_q, k_type, discrete, y_s], ...]`
            where `k_flag` is 0 for a foundation spring, or 1 for a total spring. discrete is 1 for
            a discrete spring. k_* are the spring stiffnesses for each DOF, and y_s is the location
            of the discrete spring; it is ignored if the spring is not discrete.
        constraints (np.ndarray):
            | `[[node#_e dof_e coeff node#_k dof_k], ...]
            where k=kept dof, e=dof to be eliminated. Each DOF is set as an integer where
            1=x, 2=y, 3=z, 4=q.
            The resulting constraint will be `node_e_dof = coeff * node_k_dof`
        GBT_con (GBT_Con): GBT Configuration
            | {
            |   "glob": [0|1, 0|1, ...],
            |   "dist": [0|1, 0|1, ...],
            |   "local": [0|1, 0|1, ...],
            |   "other": [0|1, 0|1, ...],
            |   "o_space": 1|2|3|4,
            |   "norm": 0|1|2|3,
            |   "couple": 1|2,
            |   "orth": 1|2|3,
            | }
            GBT_con.glob,GBT_con.dist, GBT_con.local, GBT_con.other:
                vectors of 1's and 0's referring to the inclusion (1) or exclusion of a
                given mode from the analysis,
            GBT_con.o_space - choices of ST/O mode
                | 1: ST basis
                | 2: O space (null space of GDL) with respect to K_global
                | 3: O space (null space of GDL) with respect to Kg_global
                | 4: O space (null space of GDL) in vector sense
            GBT_con.norm - code for normalization (if normalization is done at all)
                | 0: no normalization,
                | 1: vector norm
                | 2: strain energy norm
                | 3: work norm
            GBT_con.couple - coupled basis vs uncoupled basis
                for general B.C. especially for non-simply supported B.C.
                | 1: uncoupled basis, the basis will be block diagonal
                | 2: coupled basis, the basis is fully spanned
            GBT_con.orth - natural basis vs modal basis
                | 1: natural basis
                | 2: modal basis, axial orthogonality
                | 3: modal basis, load dependent orthogonality
        B_C (str): Boundary condition to be analyzed
            | 'S-S' simply-pimply supported boundary condition at loaded edges
            | 'C-C' clamped-clamped boundary condition at loaded edges
            | 'S-C' simply-clamped supported boundary condition at loaded edges
            | 'C-F' clamped-free supported boundary condition at loaded edges
            | 'C-G' clamped-guided supported boundary condition at loaded edges
        m_all (np.ndarray): Longitudinal terms for each half-wavelength
            | m_all[length#] = [longitudinal_num, ...],
            Longitudinal terms m for all the lengths in cell notation
            each cell has a vector including the longitudinal terms for this length
        n_eigs (int): Number of eigenvalues
            The number of eigenvalues to be determined at length (default=10)
        sect_props (Sect_Props): Section properties
            | {
            |   "A": float,
            |   "cx": float,
            |   "cy": float,
            |   "Ixx": float,
            |   "Iyy": float,
            |   "Ixy": float,
            |   "phi": float,
            |   "I11": float,
            |   "I22": float,
            |   "J": float,
            |   "x0": float,
            |   "y0": float,
            |   "Cw": float,
            |   "B1": float,
            |   "B2": float,
            |   "wn": np.ndarray
            | }
            Dictionary of section properties of cross-section

    Returns:
        signature (np.ndarray): Signature curve
            | `signature[length#] = load_factor`
        curve (np.ndarray): buckling curve (load factor) for each length
            | `curve[length#] = [mode1_load_factor, mode2_load_factor, ...]`
        shapes (np.ndarray): mode shapes for each length
            | `shapes[length#] = [disp_mode1, disp_mode2, ...]`
            Each `disp_*` is an array of displacements for each node in the section
    """

    n_nodes = len(nodes)
    curve = []
    shapes = []
    signature = np.zeros((len(lengths), 1))

    # CLEAN UP INPUT
    # clean u_j 0's, multiple terms. or out-of-order terms in m_all
    m_all = analysis.m_sort(m_all)

    # DETERMINE FLAGS FOR USER CONSTRAINTS AND INTERNAL (AT NODE) B.C.'s
    BC_flag = analysis.constr_BC_flag(nodes=nodes, constraints=constraints)

    # GENERATE STRIP WIDTH AND DIRECTION ANGLE
    el_props = analysis.elem_prop(nodes=nodes, elements=elements)

    # ENABLE cFSM ANALYSIS IF APPLICABLE, AND FIND BASE PROPERTIES
    if sum(GBT_con["glob"]) + sum(GBT_con["dist"]) + sum(GBT_con["local"]) + sum(GBT_con["other"]) > 0:
        # turn on modal classification analysis
        cfsm_analysis = 1
    else:
        cfsm_analysis = 0

    if cfsm_analysis == 1:
        # set u_p stress to 1.0 for finding Kg_global and K_global for axial modes
        nodes_base = deepcopy(nodes)
        nodes_base[:, 7] = np.ones_like(nodes[:, 7])

        # natural base first
        # properties all the longitudinal terms share
        [
            main_nodes,
            meta_elements,
            node_props,
            n_main_nodes,
            n_corner_nodes,
            n_sub_nodes,
            n_dist_modes,
            n_local_modes,
            dof_perm,
        ] = pycufsm.cfsm.base_properties(nodes=nodes_base, elements=elements)
        [r_x, r_z, r_yd, r_ys, r_ud] = pycufsm.cfsm.mode_constr(
            nodes=nodes_base,
            elements=elements,
            node_props=node_props,
            main_nodes=main_nodes,
            meta_elements=meta_elements,
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
            el_props=el_props,
        )
    else:
        # no modal classification constraints are engaged
        cfsm_analysis = 0

    # LOOP OVER ALL THE LENGTHS TO BE INVESTIGATED
    for i, length in enumerate(lengths):
        # longitudinal terms to be included for this length
        m_a = m_all[i]

        total_m = len(m_a)  # Total number of longitudinal terms

        # SET SWITCH AND PREPARE BASE VECTORS (R_matrix) FOR cFSM ANALYSIS
        if cfsm_analysis == 1:
            # generate natural base vectors for axial compression loading
            b_v_l = pycufsm.cfsm.base_column(
                nodes_base=nodes_base,
                elements=elements,
                props=props,
                length=length,
                B_C=B_C,
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
                d_y=d_y,
            )

        K_global, Kg_global = analysis.k_kg_global(
            nodes=nodes, elements=elements, el_props=el_props, props=props, length=length, B_C=B_C, m_a=m_a
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
                    k_u=k_u, k_v=k_v, k_w=k_w, k_q=k_q, length=length, B_C=B_C, m_a=m_a, discrete=discrete, y_s=y_s
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
                    if width < 1e-10:  # coincident nodes
                        alpha = 0  # use global coordinates for spring
                    else:
                        # local orientation for spring
                        # np.arctan2() function is mis-typed in numpy - given floats,
                        # it DOES return a float.It does not always return an array
                        alpha = np.arctan2(d_y, d_x)  # type: ignore

                gamma = analysis.trans(alpha=alpha, total_m=total_m)
                k_s = gamma @ ks_l @ gamma.conj().T

                # Add element contribution of k_s to full matrix K_global
                K_global = analysis.spring_assemble(
                    K_global=K_global, k_local=k_s, node_i=node_i, node_j=node_j, n_nodes=n_nodes, m_a=m_a
                )

        # INTERNAL BOUNDARY CONDITIONS (ON THE NODES) AND USER DEFINED CONSTR.
        # Check for user defined constraints too
        if BC_flag == 1:
            # size boundary conditions and user constraints for use in R_matrix format
            # d_constrained=r_user*d_unconstrained, d=nodal DOF vector (note by
            # BWS June 5 2006)
            r_user = pycufsm.cfsm.constr_user(nodes=nodes, constraints=constraints, m_a=m_a)
            R_u0_matrix = spla.null_space(r_user.conj().T)
            # Number of boundary conditions and user defined constraints = nu0
            nu0 = len(R_u0_matrix[0])

        # GENERATION OF cFSM CONSTRAINT MATRIX
        if cfsm_analysis == 1:
            # PERFORM ORTHOGONALIZATION IF GBT-LIKE MODES ARE ENFORCED
            b_v = pycufsm.cfsm.base_update(
                GBT_con=GBT_con,
                b_v_l=b_v_l,
                length=length,
                m_a=m_a,
                nodes=nodes,
                elements=elements,
                props=props,
                n_global_modes=n_global_modes,
                n_dist_modes=n_dist_modes,
                n_local_modes=n_local_modes,
                B_C=B_C,
                el_props=el_props,
            )
            # no normalization is enforced: 0:  m
            # assign base vectors to constraints
            b_v = pycufsm.cfsm.mode_select(
                b_v=b_v,
                n_global_modes=n_global_modes,
                n_dist_modes=n_dist_modes,
                n_local_modes=n_local_modes,
                GBT_con=GBT_con,
                n_dof_m=4 * n_nodes,
                m_a=m_a,
            )  # m
            R_mode = b_v
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else:
            # no modal constraints are activated therefore
            R_mode = np.eye(4 * n_nodes * total_m)  # activate modal constraints

        # CREATE FINAL CONSTRAINT MATRIX
        # Determine the number of modal constraints, nm0
        if BC_flag == 1:
            # should performed uncoupled for block diagonal basis?
            if cfsm_analysis == 1:
                nm0 = 0
                R_m0_matrix = spla.null_space(R_mode.conj().T)
                nm0 = len(R_m0_matrix[0])
                R_0_matrix = R_m0_matrix
                if nu0 > 0:
                    R_0_matrix[:, nm0 : (nm0 + nu0)] = R_u0_matrix
                R_matrix = spla.null_space(R_0_matrix.conj().T)
            else:
                R_matrix = spla.null_space(R_u0_matrix.conj().T)
        else:
            # if no user defined constraints and fixities.
            R_matrix = R_mode

        # INTRODUCE CONSTRAINTS AND REDUCE K_global MATRICES TO FREE PARTS ONLY
        k_global_ff = R_matrix.transpose() @ K_global @ R_matrix
        kg_global_ff = R_matrix.transpose() @ Kg_global @ R_matrix
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
        # length_factors = np.diag(length_factors)
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
        modes_full = R_matrix @ modes

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
        props (Dict[str, Dict[str, float]]): Material properties
            | `{"mat_name": {E_x: float, E_y: float, nu_x: float, nu_y: float, G_bulk: float}, ...}`
            | or
            | `{"mat_name": {E: float, nu: float}}`
            The latter option assumes an isotropic material.
        nodes (ArrayLike): Nodal coordinates
            | `[[x, y, stress], ...]`
            | or
            | `[[x, y], ...]`
            The latter option assumes that the stress will be set using
            Note that any node numbers used in later inputs refer to the **index** of the node in
            this `nodes` array, with the first node being node number 0.
        elements (Sequence[New_Element]): Element connectivity and properties
            | [{
            |   `nodes: "all"|List[int],`
            |   `t: float,`
            |   `mat: str`
            | )].
            elements["nodes"]:
                `nodes: "all"` is a special indicator that all nodes should be connected
                sequentially. If `nodes` is given as an array, any number of nodal indices
                may be listed in order
            elements["t"]:
                Material thickness
            elements["mat"]:
                material name as a string
        sect_props (Sect_Props): Section properties
            | {
            |   "A": float,
            |   "cx": float,
            |   "cy": float,
            |   "Ixx": float,
            |   "Iyy": float,
            |   "Ixy": float,
            |   "phi": float,
            |   "I11": float,
            |   "I22": float,
            |   "J": float,
            |   "x0": float,
            |   "y0": float,
            |   "Cw": float,
            |   "B1": float,
            |   "B2": float,
            |   "wn": np.ndarray
            | }
            Dictionary of section properties of cross-section
        lengths (Union[ArrayLike, set, Dict[float, ArrayLike]): Half-wavelengths for analysis
            | `[length1, length2, ...]`
            | or
            | `{length1: List[int], length2: List[int], ...}
            If given as a simple array, then the longitudinal m term will be taken as `[1]` for each
            half-wavelength (which is normally what you want for a signature curve analysis). If
            given as a dictionary, then the longitudinal m terms must be set to an array with
            appropriate values
        node_props (Optional[Dict[Union[Literal["all", int], New_Node_Props]]): Nodal DOF inclusion
            | {
            |    node_#|"all": {
            |        dof_x: bool,
            |        dof_y: bool,
            |        dof_z: bool,
            |        dof_q: bool,
            |    }
            | }
            Defaults to None, and taken as all DOFs included if so. Any DOFs set to false will be
            taken as fully constrained to ground
        springs (Optional[Sequence[New_Spring]]): Definition of any springs in cross-section
            | [{
            |    node: int,
            |    k_x: float,
            |    k_y: float,
            |    k_z: float,
            |    k_q: float,
            |    k_type: "foundation"|"node_pair"|"total",
            |    node_pair: int,
            |    discrete: bool,
            |    y: float,
            | }, ...]
            `k_type` is the stiffness type. `node_pair` and `y` keys are only required if the
            `k_type` or `discrete` options are set to require them.
            Defaults to None.
        constraints (Optional[Sequence[New_Constraint]]): Definition of any constraints in section
            | [{
            |    elim_node: int,
            |    elim_dof: "x"|"y"|"z"|"q",
            |    coeff: float,
            |    keep_node: int,
            |    keep_dof: "x"|"y"|"z"|"q"
            | }, ...]
            `"q"` is the twist DOF. Each constraint takes the form of `elim_dof = coeff * keep_dof`.
            Defaults to None.
        analysis_config (Optional[Analysis_Config]): Configuration options for any analysis
            | {
            |    B_C: "S-S"|"C-C"|"S-C"|"C-F"|"C-G",
            |    n_eigs: int
            | }
            Defaults to None, and taken as `{B_C: "S-S", n_eigs: 10}` if so.
            analysis_config["B_C"]: Boundary condition types (at loaded edges):
                | 'S-S' simple-simple
                | 'C-C' clamped-clamped
                | 'S-C' simple-clamped
                | 'C-F' clamped-free
                | 'C-G' clamped-guided
        cfsm_config (Optional[Cfsm_Config]): Configuration options for cFSM (constrained modes)
            | {
            |    glob_modes: list(int),
            |    dist_modes: list(int),
            |    local_modes: list(int),
            |    other_modes: list(int),
            |    null_space: "ST"|"K_global"|"Kg_global"|"vector",
            |    normalization: "none"|"vector"|"strain_energy"|"work",
            |    coupled: bool,
            |    orthogonality: "natural"|"modal_axial"|"modal_load"
            }
            Defaults to None, in which case no cFSM analysis is performed
            analysis_config["*_modes"]:
                list of 1's (inclusion) and 0's (exclusion) for each mode from the analysis
            analysis_config["null_space"]:
                | "ST": ST basis
                | "K_global": null space of GDL with respect to K_global
                | "Kg_global": null space of GDL with respect to Kg_global
                | "vector": null space of GDL in vector sense
            analysis_config["normalization"]: Type of normalization
                If any is performed.
            analysis_config["coupled"]: basis for general boundary conditions
                | uncoupled basis = the basis will be block diagonal
                | coupled basis = the basis is fully spanned
            analysis_config["orthogonality"]: natural basis vs modal basis
                | "natural": natural basis
                | "modal_axial": modal basis, axial orthogonality
                | "modal_load": modal basis, load dependent orthogonality
        yield_force (Optional[Yield_Force]): Single yield force to apply to section.
            | {
            |    force: "Mxx"|"Myy"|"M11"|"M22"|"P",
            |    direction: "Pos"|"Neg"|"+"|"-",
            |    f_y: float,
            |    restrain: bool,
            |    offset: ArrayLike
            | }
            Either this or 'forces' must be set, or stresses must be set manually in `nodes`.
            yield_force["restrain"]:
                Note that 'restrain' only affects "Mxx" or "Myy" forces, and then only for sections
                in which the principal axes are no aligned with the geometric axes (such as Z
                sections)
            yield_force["offset"]:
                | `[x_offset, y_offset]`
                Offset from the (0,0) coordinate used in calculating section properties and the
                (0,0) coordinate used to define the nodal coordinates. This may commonly differ
                by thickness/2, for example, if an external section properties calculator is used
        forces (Optional[Forces]): Specific forces to apply to the section.
            | {
            |    'P': float,
            |    'Mxx': float,
            |    'Myy': float,
            |    'M11': float,
            |    'M22': float,
            |    'restrain': bool,
            |    'offset': ArrayLike
            | }
            forces["restrain"]:
                Note that 'restrain' only affects "Mxx" or "Myy" forces, and then only for sections
                in which the principal axes are no aligned with the geometric axes (such as Z
                sections)
            forces["offset"]:
                | `[x_offset, y_offset]`
                Offset from the (0,0) coordinate used in calculating section properties and the
                (0,0) coordinate used to define the nodal coordinates. This may commonly differ
                by thickness/2, for example, if an external section properties calculator is used
            Either this or 'yield_force' must be set, or stresses must be set manually in `nodes`.


    Returns:
        signature (np.ndarray): Signature curve
            | `signature[length#] = load_factor`
        curve (np.ndarray): buckling curve (load factor) for each length
            | `curve[length#] = [mode1_load_factor, mode2_load_factor, ...]`
        shapes (np.ndarray): mode shapes for each length
            | `shapes[length#] = [disp_mode1, disp_mode2, ...]`
            Each `disp_*` is an array of displacements for each node in the section
        nodes_stressed: Nodal coordinates with stresses
            | `[[x, y, stress], ...]`
        lengths: Half-wavelengths used in analysis
            | `[length1, length2, ...]`
    """
    if lengths is None or isinstance(lengths, int):
        n_lengths = lengths if isinstance(lengths, int) else 50
        lengths = []

    (
        props_old,
        nodes_old,
        elements_old,
        lengths_old,
        springs_old,
        constraints_old,
        GBT_con_old,
        B_C_old,
        m_all_old,
        n_eigs_old,
    ) = inputs_new_to_old(
        props=props,
        nodes=nodes,
        lengths=lengths,
        elements=elements,
        springs=springs,
        constraints=constraints,
        node_props=node_props,
        analysis_config=analysis_config,
        cfsm_config=cfsm_config,
    )

    if forces is None and yield_force is not None:
        restrained = yield_force["restrain"] if "restrain" in yield_force else False
        offset = yield_force["offset"] if "offset" in yield_force and yield_force["offset"] is not None else [0, 0]
        all_yields = yield_mp(nodes=nodes_old, f_y=yield_force["f_y"], sect_props=sect_props, restrained=restrained)
        forces = {"Mxx": 0, "Myy": 0, "M11": 0, "M22": 0, "P": 0, "restrain": restrained, "offset": offset}
        if yield_force["direction"] == "-" or yield_force["direction"] == "Neg":
            multiplier = -1
        else:
            multiplier = 1
        forces[yield_force["force"]] = all_yields[yield_force["force"]] * multiplier
    elif forces is not None and yield_force is not None:
        raise ValueError("Only one of 'forces' or 'yield_force' may be set - but not both")
    elif forces is None and yield_force is None and np.shape(nodes)[1] == 2:
        raise ValueError(
            "Either 'forces' or 'yield_force' must be set, " + "or stress must be set manually for each node"
        )
    if forces is not None:
        nodes_stressed = stress_gen(nodes=nodes_old, forces=forces, sect_props=sect_props)
    elif np.shape(nodes)[1] == 3:
        nodes_stressed = nodes_old
    else:
        raise ValueError("Either 'forces' or 'yield_force' must be set, or stress must be set manually for each node")

    if len(lengths) == 0:
        lengths_old = lengths_recommend(nodes=nodes_old, elements=elements_old, n_lengths=n_lengths)

    signature, curve, shapes = strip(
        props=props_old,
        nodes=nodes_stressed,
        elements=elements_old,
        lengths=lengths_old,
        springs=springs_old,
        constraints=constraints_old,
        GBT_con=GBT_con_old,
        B_C=B_C_old,
        m_all=m_all_old,
        n_eigs=n_eigs_old,
        sect_props=sect_props,
    )

    return signature, curve, shapes, nodes_stressed, lengths_old


def signature_ss(
    props: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    i_GBT_con: GBT_Con,
    sect_props: Sect_Props,
    lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """generate the signature curve solution, part 2: actually solve the signature curve

    Args:
        props (np.ndarray): standard parameter
        nodes (np.ndarray): standard parameter
        elements (np.ndarray): standard parameter
        i_GBT_con (GBT_Con): cFSM configuration options
        sect_props (Sect_Props): section properties
        lengths (np.ndarray): half-wavelengths

    Returns:
        signature: signature curve,
        curve: all the curve results,
        shapes: deformed shapes at each point

    (function originally in helpers; moved to fsm because it drives entire fsm analyses)
    Z. Li, July 2010 (last modified)
    """
    i_springs = np.array([])
    i_constraints = np.array([])
    i_B_C: BC = "S-S"
    i_m_all = np.ones((len(lengths), 1))

    isignature, icurve, ishapes = pycufsm.fsm.strip(
        props=props,
        nodes=nodes,
        elements=elements,
        lengths=lengths,
        springs=i_springs,
        constraints=i_constraints,
        GBT_con=i_GBT_con,
        B_C=i_B_C,
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
    lengths: Optional[np.ndarray] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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

    (function originally in helpers; moved to fsm because it drives entire fsm analyses)
    Z. Li, Oct. 2010
    """
    i_GBT_con: GBT_Con = {
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
        lengths = lengths_recommend(nodes=nodes, elements=elements, length_append=length_append, n_lengths=n_lengths)

    print("Running initial pyCUFSM signature curve")
    isignature, icurve, ishapes = signature_ss(
        props=props, nodes=nodes, elements=elements, i_GBT_con=i_GBT_con, sect_props=sect_props, lengths=lengths
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

    _, _, _, _, _, _, n_dist_modes, n_local_modes, _ = pycufsm.cfsm.base_properties(nodes=nodes, elements=elements)

    n_global_modes = 4
    n_other_modes = 2 * (len(nodes) - 1)

    i_GBT_con["local"] = np.ones((n_local_modes, 1)).tolist()
    i_GBT_con["dist"] = np.zeros((n_dist_modes, 1)).tolist()
    i_GBT_con["glob"] = np.zeros((n_global_modes, 1)).tolist()
    i_GBT_con["other"] = np.zeros((n_other_modes, 1)).tolist()

    print("Running pyCUFSM local modes curve")
    isignature_local, icurve_local, ishapes_local = signature_ss(
        props=props, nodes=nodes, elements=elements, i_GBT_con=i_GBT_con, sect_props=sect_props, lengths=lengths
    )

    print("Running pyCUFSM distortional modes curve")
    i_GBT_con["local"] = np.zeros((n_local_modes, 1)).tolist()
    i_GBT_con["dist"] = np.ones((n_dist_modes, 1)).tolist()
    i_GBT_con["glob"] = np.zeros((n_global_modes, 1)).tolist()
    i_GBT_con["other"] = np.zeros((n_other_modes, 1)).tolist()
    isignature_dist, icurve_dist, ishapes_dist = signature_ss(
        props=props, nodes=nodes, elements=elements, i_GBT_con=i_GBT_con, sect_props=sect_props, lengths=lengths
    )

    curve_signature_local = np.zeros((len(lengths), 2))
    curve_signature_local[:, 0] = lengths
    curve_signature_local[:, 1] = isignature_local
    curve_signature_dist = np.zeros((len(lengths), 2))
    curve_signature_dist[:, 0] = lengths
    curve_signature_dist[:, 1] = isignature_dist

    # cFSM local half-wavelength
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

    # cFSM dist half-wavelength
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
        # half-wavelength of local and distortional buckling
        length_crl = local_minima_local[0]
        length_crd = local_minima_dist[0]

    # recommend longitudinal terms m
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
            im_pm_all_temp.extend(
                [
                    np.ceil(im_p_len / length_crd) - 3,
                    np.ceil(im_p_len / length_crd) - 2,
                    np.ceil(im_p_len / length_crd) - 1,
                    np.ceil(im_p_len / length_crd),
                    np.ceil(im_p_len / length_crd) + 1,
                    np.ceil(im_p_len / length_crd) + 2,
                    np.ceil(im_p_len / length_crl) + 3,
                ]
            )
        else:
            im_pm_all_temp.extend([1, 2, 3, 4, 5, 6, 7])

        im_pm_all_temp.extend([1, 2, 3])

        im_pm_all.append(im_pm_all_temp)

    # m_a_recommend = analysis.m_sort(im_pm_all)
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
