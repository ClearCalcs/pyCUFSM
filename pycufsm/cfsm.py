from copy import deepcopy
from typing import List, Tuple

import numpy as np
from scipy import linalg as spla  # type: ignore

from pycufsm.analysis import analysis
from pycufsm.types import GBT_Con, Sect_Props

# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE, CPEng
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


def base_column(
    nodes_base: np.ndarray,
    elements: np.ndarray,
    props: np.ndarray,
    length: float,
    B_C: str,
    m_a: np.ndarray,
    el_props: np.ndarray,
    node_props: np.ndarray,
    n_main_nodes: int,
    n_corner_nodes: int,
    n_sub_nodes: int,
    n_global_modes: int,
    n_dist_modes: int,
    n_local_modes: int,
    dof_perm: np.ndarray,
    r_x: np.ndarray,
    r_z: np.ndarray,
    r_ys: np.ndarray,
    d_y: np.ndarray,
) -> np.ndarray:
    """this routine creates base vectors for a column with length length for all the
    specified longitudinal terms in m_a

    assumptions
        orthogonalization is not performed unless the user wants
        orthogonalization is done by solving the eigen-value problem within each sub-space
        normalization is not done

    Args:
        nodes_base (np.ndarray): standard nodes parameter, but with zero stresses
        elements (np.ndarray): standard parameter
        props (np.ndarray): standard parameter
        length (float): half-wavelength
        B_C (str): ['S-S'] a string specifying boundary conditions to be analyzed:
            'S-S' simply-pimply supported boundary condition at loaded edges
            'C-C' clamped-clamped boundary condition at loaded edges
            'S-C' simply-clamped supported boundary condition at loaded edges
            'C-F' clamped-free supported boundary condition at loaded edges
            'C-G_bulk' clamped-guided supported boundary condition at loaded edges
        m_a (np.ndarray): longitudinal terms (half-wave numbers)
        el_props (np.ndarray): standard parameter
        node_props (np.ndarray): _description_
        n_main_nodes (int): _description_
        n_corner_nodes (int): _description_
        n_sub_nodes (int): _description_
        n_global_modes (int): _description_
        n_dist_modes (int): _description_
        n_local_modes (int): _description_
        dof_perm (np.ndarray): _description_
        r_x (np.ndarray): _description_
        r_z (np.ndarray): _description_
        r_ys (np.ndarray): _description_
        d_y (np.ndarray): _description_

    Returns:
        b_v_l (np.ndarray): base vectors (each column corresponds to a certain mode)
            assemble for each half-wave number m_i on its diagonal
            b_v_l = diag(b_v_m)
            for each half-wave number m_i, b_v_m
                    columns 1..n_global_modes: global modes
                    columns (n_global_modes+1)..(n_global_modes+n_dist_modes): dist. modes
                    columns (n_global_modes+n_dist_modes+1)
                            ..(n_global_modes+n_dist_modes+n_local_modes): local modes
                    columns (n_global_modes+n_dist_modes+n_local_modes+1)..n_dof: other modes
            n_global_modes, n_dist_modes, n_local_modes - number of G_bulk, D, L modes, respectively

    S. Adany, Aug 28, 2006
    B. Schafer, Aug 29, 2006
    Z. Li, Dec 22, 2009
    Z. Li, June 2010
    """

    # construct the base for all the longitudinal terms
    n_nodes = len(nodes_base)
    n_dof_m = 4 * n_nodes
    total_m = len(m_a)
    b_v_l = np.zeros((n_dof_m * total_m, n_dof_m * total_m))
    for i, m_i in enumerate(m_a):
        # to create r_p constraint matrix for the rest of planar DOFs
        r_p = constr_planar_xz(nodes_base, elements, props, node_props, dof_perm, m_i, length, B_C, el_props)
        b_v_m = base_vectors(
            d_y=d_y,
            elements=elements,
            el_props=el_props,
            length=length,
            m_i=m_i,
            node_props=node_props,
            n_main_nodes=n_main_nodes,
            n_corner_nodes=n_corner_nodes,
            n_sub_nodes=n_sub_nodes,
            n_global_modes=n_global_modes,
            n_dist_modes=n_dist_modes,
            n_local_modes=n_local_modes,
            r_x=r_x,
            r_z=r_z,
            r_p=r_p,
            r_ys=r_ys,
            dof_perm=dof_perm,
        )
        b_v_l[(n_dof_m * i) : n_dof_m * (i + 1), (n_dof_m * i) : n_dof_m * (i + 1)] = b_v_m

    return b_v_l


def base_update(
    GBT_con: GBT_Con,
    b_v_l: np.ndarray,
    length: float,
    m_a: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    props: np.ndarray,
    n_global_modes: int,
    n_dist_modes: int,
    n_local_modes: int,
    B_C: str,
    el_props: np.ndarray,
) -> np.ndarray:
    """this routine optionally makes orthogonalization and normalization of base vectors

    assumptions
        orthogonalization is done by solving the EV problem for each sub-space
        three options for normalization is possible, set by 'GBT_con['norm']' parameter

    Args:
        GBT_con (GBT_Con):
            GBT_con['o_space'] - by GBT_con, choices of ST/O mode
                    1: ST basis
                    2: O space (null space of GDL) with respect to K_global
                    3: O space (null space of GDL) with respect to Kg_global
                    4: O space (null space of GDL) in vector sense
            GBT_con['norm'] - by GBT_con, code for normalization (if normalization is done at all)
                    0: no normalization,
                    1: vector norm
                    2: strain energy norm
                    3: work norm
            GBT_con['couple'] - by GBT_con, coupled basis vs uncoupled basis for general B.C.
                        especially for non-simply supported B.C.
                    1: uncoupled basis, the basis will be block diagonal
                    2: coupled basis, the basis is fully spanned
            GBT_con['orth'] - by GBT_con, natural basis vs modal basis
                    1: natural basis
                    2: modal basis, axial orthogonality
                    3: modal basis, load dependent orthogonality
        b_v_l (np.ndarray): natural base vectors for length (each column corresponds to a
            certain mode) for each half-wave number m_i
                    columns 1..n_global_modes: global modes
                    columns (n_global_modes+1)..(n_global_modes+n_dist_modes): dist. modes
                    columns (n_global_modes+n_dist_modes+1)
                            ..(n_global_modes+n_dist_modes+n_local_modes): local modes
                    columns (n_global_modes+n_dist_modes+n_local_modes+1)..n_dof_m: other modes
        length (float): _description_
        m_a (np.ndarray): _description_
        nodes (np.ndarray): _description_
        elements (np.ndarray): _description_
        props (np.ndarray): _description_
        n_global_modes (int): _description_
        n_dist_modes (int): _description_
        n_local_modes (int): _description_
        B_C (str): _description_
        el_props (np.ndarray): _description_

    Returns:
        b_v (np.ndarray): output base vectors (maybe natural, orthogonal or normalized,
            depending on the selected options)

    S. Adany, Oct 11, 2006
    Z. Li modified on Jul 10, 2009
    Z. Li, June 2010
    """

    n_nodes = len(nodes[:, 1])
    n_dof_m = 4 * n_nodes
    total_m = len(m_a)  # Total number of longitudinal terms m_i
    b_v = np.zeros((n_dof_m * total_m, n_dof_m * total_m))

    if GBT_con["couple"] == 1:
        # uncoupled basis
        for i, m_i in enumerate(m_a):
            b_v_m = b_v_l[n_dof_m * i : n_dof_m * (i + 1), n_dof_m * i : n_dof_m * (i + 1)]
            # K_global/Kg_global
            if GBT_con["norm"] in (2, 3) or GBT_con["o_space"] in (2, 3) or GBT_con["orth"] in (2, 3):
                # axial loading or real loading by either GBT_con['orth'] = 2 or GBT_con['orth'] = 3
                if GBT_con["orth"] == 1 or GBT_con["orth"] == 2:
                    nodes_base = deepcopy(nodes)
                    nodes_base[:, 7] = np.ones_like(nodes[:, 7])  # set u_p stress to 1.0 (axial)
                    K_global, Kg_global = analysis.k_kg_global(
                        nodes=nodes_base,
                        elements=elements,
                        el_props=el_props,
                        props=props,
                        length=length,
                        B_C=B_C,
                        m_a=[m_i],
                    )
                else:
                    K_global, Kg_global = analysis.k_kg_global(
                        nodes=nodes,
                        elements=elements,
                        el_props=el_props,
                        props=props,
                        length=length,
                        B_C=B_C,
                        m_a=[m_i],
                    )

            # orthogonalization/normalization begins
            #
            if (
                GBT_con["orth"] == 2
                or GBT_con["orth"] == 3
                or GBT_con["o_space"] == 2
                or GBT_con["o_space"] == 3
                or GBT_con["o_space"] == 4
            ):
                # indices
                if GBT_con["o_space"] == 1:
                    dof_index = np.zeros((5, 2))
                    dof_index[3, 0] = n_global_modes + n_dist_modes + n_local_modes
                    dof_index[3, 1] = n_global_modes + n_dist_modes + n_local_modes + n_nodes - 1
                    dof_index[4, 0] = n_global_modes + n_dist_modes + n_local_modes + n_nodes - 1
                    dof_index[4, 1] = n_dof_m
                else:
                    dof_index = np.zeros((4, 2))
                    dof_index[3, 0] = n_global_modes + n_dist_modes + n_local_modes
                    dof_index[3, 1] = n_dof_m
                dof_index[0, 0] = 0
                dof_index[0, 1] = n_global_modes
                dof_index[1, 0] = n_global_modes
                dof_index[1, 1] = n_global_modes + n_dist_modes
                dof_index[2, 0] = n_global_modes + n_dist_modes
                dof_index[2, 1] = n_global_modes + n_dist_modes + n_local_modes

                # define vectors for other modes, GBT_con['o_space'] = 2, 3, 4
                if GBT_con["o_space"] == 2:
                    a_matrix = spla.null_space(b_v_m[:, dof_index[0, 0] : dof_index[2, 1]].conj().T)
                    b_v_m[:, dof_index[3, 0] : dof_index[3, 1]] = np.linalg.solve(K_global, a_matrix)
                if GBT_con["o_space"] == 3:
                    a_matrix = spla.null_space(b_v_m[:, dof_index[0, 0] : dof_index[2, 1]].conj().T)
                    b_v_m[:, dof_index[3, 0] : dof_index[3, 1]] = np.linalg.solve(Kg_global, a_matrix)
                if GBT_con["o_space"] == 4:
                    a_matrix = spla.null_space(b_v_m[:, dof_index[0, 0] : dof_index[2, 1]].conj().T)
                    b_v_m[:, dof_index[3, 0] : dof_index[3, 1]] = a_matrix

                # orthogonalization for modal basis 2/3 + normalization for normals 2/3
                for dof_sub in dof_index:
                    dof_sub1 = int(dof_sub[1])
                    dof_sub0 = int(dof_sub[0])
                    if dof_sub[1] >= dof_sub[0]:
                        k_global_sub = b_v_m[:, dof_sub0:dof_sub1].conj().T @ K_global @ b_v_m[:, dof_sub0:dof_sub1]
                        kg_global_sub = b_v_m[:, dof_sub0:dof_sub1].conj().T @ Kg_global @ b_v_m[:, dof_sub0:dof_sub1]
                        [eigenvalues, eigenvectors] = spla.eig(a=k_global_sub, b=kg_global_sub)
                        lf_sub = np.real(eigenvalues)
                        indexsub = np.argsort(lf_sub)
                        lf_sub = lf_sub[indexsub]
                        eigenvectors = np.real(eigenvectors[:, indexsub])
                        if GBT_con["norm"] == 2 or GBT_con["norm"] == 3:
                            if GBT_con["norm"] == 2:
                                s_matrix = eigenvectors.conj().T @ k_global_sub @ eigenvectors

                            if GBT_con["norm"] == 3:
                                s_matrix = eigenvectors.conj().T @ kg_global_sub @ eigenvectors

                            s_matrix = np.diag(s_matrix)
                            for j in range(0, int(dof_sub[1] - dof_sub[0])):
                                eigenvectors[:, j] = np.transpose(
                                    np.conj(np.linalg.lstsq(eigenvectors[:, j].conj().T, np.sqrt(s_matrix).conj().T))
                                )

                        b_v_m[:, dof_sub0:dof_sub1] = b_v_m[:, dof_sub0:dof_sub1] @ eigenvectors

            # normalization for GBT_con['o_space'] = 1
            if (GBT_con["norm"] == 2 or GBT_con["norm"] == 3) and GBT_con["o_space"] == 1:
                for j in range(0, n_dof_m):
                    if GBT_con["norm"] == 2:
                        b_v_m[:, j] = np.transpose(
                            np.conj(
                                np.linalg.lstsq(
                                    b_v_m[:, j].conj().T,
                                    np.sqrt(b_v_m[:, j].conj().T @ K_global @ b_v_m[:, j]).conj().T,
                                )
                            )
                        )

                    if GBT_con["norm"] == 3:
                        b_v_m[:, j] = np.transpose(
                            np.conj(
                                np.linalg.lstsq(
                                    b_v_m[:, j].conj().T,
                                    np.sqrt(b_v_m[:, j].conj().T @ Kg_global @ b_v_m[:, j]).conj().T,
                                )
                            )
                        )

            # normalization for GBT_con['norm'] 1
            if GBT_con["norm"] == 1:
                for j in range(0, n_dof_m):
                    b_v_m[:, j] = b_v_m[:, j] / np.sqrt(b_v_m[:, j].conj().T @ b_v_m[:, j])

            b_v[n_dof_m * i : n_dof_m * (i + 1), n_dof_m * i : n_dof_m * (i + 1)] = b_v_m

    else:
        # coupled basis
        # K_global/Kg_global
        if GBT_con["norm"] in (2, 3) or GBT_con["o_space"] in (2, 3) or GBT_con["orth"] in (2, 3):
            # axial loading or real loading by either GBT_con['orth'] = 2 or GBT_con['orth'] = 3
            if GBT_con["orth"] == 1 or GBT_con["orth"] == 2:
                nodes_base = deepcopy(nodes)
                nodes_base[:, 7] = np.ones_like(nodes[:, 7])  # set u_p stress to 1.0 (axial)
            else:
                nodes_base = nodes

        K_global, Kg_global = analysis.k_kg_global(
            nodes=nodes, elements=elements, el_props=el_props, props=props, length=length, B_C=B_C, m_a=m_a
        )

        # orthogonalization/normalization begins
        if (
            GBT_con["orth"] == 2
            or GBT_con["orth"] == 3
            or GBT_con["o_space"] == 2
            or GBT_con["o_space"] == 3
            or GBT_con["o_space"] == 4
        ):
            # indices
            dof_index = np.zeros((4, 2))
            dof_index[0, 0] = 0
            dof_index[0, 1] = n_global_modes
            dof_index[1, 0] = n_global_modes
            dof_index[1, 1] = n_global_modes + n_dist_modes
            dof_index[2, 0] = n_global_modes + n_dist_modes
            dof_index[2, 1] = n_global_modes + n_dist_modes + n_local_modes
            dof_index[3, 0] = n_global_modes + n_dist_modes + n_local_modes
            dof_index[3, 1] = n_dof_m

            n_other_modes = n_dof_m - (n_global_modes + n_dist_modes + n_local_modes)

            b_v_gdl = np.zeros(((len(m_a) + 1) * (n_global_modes + n_dist_modes + n_local_modes), 1))
            b_v_g = np.zeros(((len(m_a) + 1) * n_global_modes, 1))
            b_v_d = np.zeros(((len(m_a) + 1) * n_dist_modes, 1))
            b_v_l = np.zeros(((len(m_a) + 1) * n_local_modes, 1))
            b_v_o = np.zeros(((len(m_a) + 1) * n_other_modes, 1))
            for i, m_i in enumerate(m_a):
                # considering length-dependency on base vectors
                b_v_m = b_v_l[:, n_dof_m * i : n_dof_m * (i + 1)]  # n_dof_m*i:n_dof_m*(i+1)
                b_v_gdl[
                    :,
                    i
                    * (n_global_modes + n_dist_modes + n_local_modes) : (i + 1)
                    * (n_global_modes + n_dist_modes + n_local_modes),
                ] = b_v_m[:, dof_index[1, 1] : dof_index[3, 2]]
                b_v_g[:, i * n_global_modes : (i + 1) * n_global_modes] = b_v_m[:, dof_index[1, 1] : dof_index[1, 2]]
                b_v_d[:, i * n_dist_modes : (i + 1) * n_dist_modes] = b_v_m[:, dof_index[2, 1] : dof_index[2, 2]]
                b_v_l[:, i * n_local_modes : (i + 1) * n_local_modes] = b_v_m[:, dof_index[3, 1] : dof_index[3, 2]]
                b_v_o[:, i * n_other_modes : (i + 1) * n_other_modes] = b_v_m[:, dof_index[4, 1] : dof_index[4, 2]]
                #

            # define vectors for other modes, GBT_con['o_space'] = 3 only
            if GBT_con["o_space"] == 3:
                a_matrix = spla.null_space(b_v_gdl.conj().T)
                b_v_o = np.linalg.solve(K_global, a_matrix)
                for i, m_i in enumerate(m_a):
                    b_v[:, i * n_dof_m + dof_index[3, 0] : i * n_dof_m + dof_index[3, 1]] = b_v_o[
                        :, i * n_other_modes + 1 : (i + 1) * n_other_modes
                    ]

            # define vectors for other modes, GBT_con['o_space'] = 4 only
            if GBT_con["o_space"] == 4:
                a_matrix = spla.null_space(b_v_gdl.conj().T)
                b_v_o = np.linalg.solve(Kg_global, a_matrix)
                for i, m_i in enumerate(m_a):
                    b_v[:, i * n_dof_m + dof_index[3, 0] : i * n_dof_m + dof_index[3, 1]] = b_v_o[
                        :, i * n_other_modes + 1 : (i + 1) * n_other_modes
                    ]

            # define vectors for other modes, GBT_con['o_space'] = 5 only
            if GBT_con["o_space"] == 5:
                a_matrix = spla.null_space(b_v_gdl.conj().T)
                for i, m_i in enumerate(m_a):
                    b_v[:, i * n_dof_m + dof_index[3, 0] : i * n_dof_m + dof_index[3, 1]] = a_matrix[
                        :, i * n_other_modes + 1 : (i + 1) * n_other_modes
                    ]

            # orthogonalization + normalization for normals 2/3
            for i_sub, dof_sub in enumerate(dof_index):
                if dof_sub[2] >= dof_sub[1]:
                    if i_sub == 1:
                        k_global_sub = b_v_g.conj().T * K_global * b_v_g
                        kg_global_sub = b_v_g.conj().T * Kg_global * b_v_g
                    elif i_sub == 2:
                        k_global_sub = b_v_d.conj().T * K_global * b_v_d
                        kg_global_sub = b_v_d.conj().T * Kg_global * b_v_d
                    elif i_sub == 3:
                        k_global_sub = b_v_l.conj().T * K_global * b_v_l
                        kg_global_sub = b_v_l.conj().T * Kg_global * b_v_l
                    elif i_sub == 4:
                        k_global_sub = b_v_o.conj().T * K_global * b_v_o
                        kg_global_sub = b_v_o.conj().T * Kg_global * b_v_o

                    [eigenvalues, eigenvectors] = spla.eig(a=k_global_sub, b=kg_global_sub)
                    lf_sub = np.real(eigenvalues)
                    indexsub = np.argsort(lf_sub)
                    lf_sub = lf_sub[indexsub]
                    eigenvectors = np.real(eigenvectors[:, indexsub])
                    if GBT_con["norm"] == 2 or GBT_con["norm"] == 3:
                        if GBT_con["norm"] == 2:
                            s_matrix = eigenvectors.conj().T @ k_global_sub @ eigenvectors
                        if GBT_con["norm"] == 3:
                            s_matrix = eigenvectors.conj().T @ kg_global_sub @ eigenvectors
                        s_matrix = np.diag(s_matrix)
                        for i in range(0, (dof_sub[1] - dof_sub[0]) * total_m):
                            eigenvectors[:, i] = np.transpose(
                                np.conj(np.linalg.lstsq(eigenvectors[:, i].conj().T, np.sqrt(s_matrix).conj().T))
                            )

                    if i_sub == 1:
                        b_v_orth = b_v_g @ eigenvectors
                    elif i_sub == 2:
                        b_v_orth = b_v_d @ eigenvectors
                    elif i_sub == 3:
                        b_v_orth = b_v_l @ eigenvectors
                    elif i_sub == 4:
                        b_v_orth = b_v_o @ eigenvectors

                    for i, m_i in enumerate(m_a):
                        if i_sub == 1:
                            b_v[:, i * n_dof_m + dof_sub[1] : i * n_dof_m + dof_sub[2]] = b_v_orth[
                                :, i * n_global_modes + 1 : (i + 1) * n_global_modes
                            ]
                        elif i_sub == 2:
                            b_v[:, i * n_dof_m + dof_sub[1] : i * n_dof_m + dof_sub[2]] = b_v_orth[
                                :, i * n_dist_modes + 1 : (i + 1) * n_dist_modes
                            ]
                        elif i_sub == 3:
                            b_v[:, i * n_dof_m + dof_sub[1] : i * n_dof_m + dof_sub[2]] = b_v_orth[
                                :, i * n_local_modes + 1 : (i + 1) * n_local_modes
                            ]
                        elif i_sub == 4:
                            b_v[:, i * n_dof_m + dof_sub[1] : i * n_dof_m + dof_sub[2]] = b_v_orth[
                                :, i * n_other_modes + 1 : (i + 1) * n_other_modes
                            ]

        # normalization for GBT_con['o_space'] = 1
        if (GBT_con["norm"] == 2 or GBT_con["norm"] == 3) and (GBT_con["o_space"] == 1):
            for i in range(0, n_dof_m * total_m):
                if GBT_con["norm"] == 2:
                    b_v[:, i] = np.transpose(
                        np.conj(
                            np.linalg.lstsq(
                                b_v[:, i].conj().T, np.sqrt(b_v[:, i].conj().T @ K_global @ b_v[:, i]).conj().T
                            )
                        )
                    )

                if GBT_con["norm"] == 3:
                    b_v[:, i] = np.transpose(
                        np.conj(
                            np.linalg.lstsq(
                                b_v[:, i].conj().T, np.sqrt(b_v[:, i].conj().T @ Kg_global @ b_v[:, i]).conj().T
                            )
                        )
                    )

        # normalization for GBT_con['norm'] 1
        if GBT_con["norm"] == 1:
            for i in range(0, n_dof_m * total_m):
                b_v[:, i] = np.transpose(
                    np.conj(np.linalg.lstsq(b_v[:, i].conj().T, np.sqrt(b_v[:, i].conj().T @ b_v[:, i]).conj().T))
                )
        #     b_v[n_dof_m*i:n_dof_m*(i+1),n_dof_m*i:n_dof_m*(i+1)] = b_v_m
    return b_v


def mode_select(
    b_v: np.ndarray,
    n_global_modes: int,
    n_dist_modes: int,
    n_local_modes: int,
    GBT_con: GBT_Con,
    n_dof_m: int,
    m_a: np.ndarray,
) -> np.ndarray:
    """this routine selects the required base vectors
        b_v_red forms a reduced space for the calculation, including the
            selected modes only
        b_v_red itself is the final constraint matrix for the selected modes

        note:
        for all if_* indicator: 1 if selected, 0 if eliminated

    Args:
        b_v (np.ndarray): base vectors (each column corresponds to a certain mode)
            columns 1..n_global_modes: global modes
            columns (n_global_modes+1)..(n_global_modes+n_dist_modes): dist. modes
            columns (n_global_modes+n_dist_modes+1)
                    ..(n_global_modes+n_dist_modes+n_local_modes): local modes
            columns (n_global_modes+n_dist_modes+n_local_modes+1)..n_dof: other modes
        n_global_modes (int): _description_
        n_dist_modes (int): _description_
        n_local_modes (int): _description_
        GBT_con (GBT_Con): GBT_con['glob'] - indicator which global modes are selected
            GBT_con['dist'] - indicator which dist. modes are selected
            GBT_con['local'] - indicator whether local modes are selected
            GBT_con['other'] - indicator whether other modes are selected
        n_dof_m (int): 4*n_nodes, total DOF for a single longitudinal term
        m_a (np.ndarray): _description_

    Returns:
        b_v_red (np.ndarray): reduced base vectors (each column corresponds to a certain mode)

    S. Adany, Mar 22, 2004
    BWS May 2004
    modifed on Jul 10, 2009 by Z. Li for general B_C
    Z. Li, June 2010
    """
    n_m = int(sum(GBT_con["glob"]) + sum(GBT_con["dist"]) + sum(GBT_con["local"]) + sum(GBT_con["other"]))
    b_v_red = np.zeros((len(b_v), (len(m_a) + 1) * n_m))
    for i in range(0, len(m_a)):
        #     b_v_m = b_v[n_dof_m*i:n_dof_m*(i+1),n_dof_m*i:n_dof_m*(i+1)]
        n_other_modes = n_dof_m - n_global_modes - n_dist_modes - n_local_modes  # nr of other modes
        #
        nmo = 0
        b_v_red_m = np.zeros((len(b_v), n_m))
        for j in range(0, n_global_modes):
            if j < len(GBT_con["glob"]) and GBT_con["glob"][j] == 1:
                b_v_red_m[:, nmo] = b_v[:, n_dof_m * i + j]
                nmo = nmo + 1

        for j in range(0, n_dist_modes):
            if j < len(GBT_con["dist"]) and GBT_con["dist"][j] == 1:
                b_v_red_m[:, nmo] = b_v[:, n_dof_m * i + n_global_modes + j]
                nmo = nmo + 1

        # if GBT_con['local'] == 1
        #     b_v_red[:,(nmo+1):(nmo+n_local_modes)]
        #         = b_v[:,(n_global_modes+n_dist_modes+1):(n_global_modes+
        #               n_dist_modes+n_local_modes)]
        #     nmo = nmo+n_local_modes
        # end
        for j in range(0, n_local_modes):
            if j < len(GBT_con["local"]) and GBT_con["local"][j] == 1:
                b_v_red_m[:, nmo] = b_v[:, n_dof_m * i + n_global_modes + n_dist_modes + j]
                nmo = nmo + 1

        for j in range(0, n_other_modes):
            if j < len(GBT_con["other"]) and GBT_con["other"][j] == 1:
                b_v_red_m[:, nmo] = b_v[:, n_dof_m * i + n_global_modes + n_dist_modes + n_local_modes + j]
                nmo = nmo + 1

        # if GBT_con['other'] == 1
        #     n_other_modes = len(b_v[:, 1])-n_global_modes - n_dist_modes - n_local_modes
        #            # nr of other modes
        #     b_v_red[:,(nmo+1):(nmo+n_other_modes)]
        #          = b_v[:,(n_global_modes+n_dist_modes+n_local_modes+1):(n_global_modes+
        #                n_dist_modes+n_local_modes+n_other_modes)]
        #     # b_v_red[:,(nmo+1)] = b_v[:,(n_global_modes+n_dist_modes+n_local_modes+1)]
        # end
        b_v_red[:, nmo * i : nmo * (i + 1)] = b_v_red_m

    return b_v_red


def constr_user(nodes: np.ndarray, constraints: np.ndarray, m_a: np.ndarray) -> np.ndarray:
    """this routine creates the constraints matrix, r_user_matrix, as defined by the user

    Args:
        nodes (np.ndarray): same as elsewhere throughout this program
        constraints (np.ndarray): same as 'constraints' throughout this program
        m_a (np.ndarray): longitudinal terms to be included for this length

    Returns:
        r_user_matrix (np.ndarray): the constraints matrix (in other words: base vectors) so that
            displ_orig = r_user_matrix * displ_new

    S. Adany, Feb 26, 2004
    Z. Li, Aug 18, 2009 for general b.c.
    Z. Li, June 2010
    """
    n_nodes = len(nodes[:, 1])
    n_dof_m = 4 * n_nodes
    dof_reg = np.ones((n_dof_m, 1))
    r_user_matrix = np.eye(n_dof_m * len(m_a))
    for i in range(0, len(m_a)):
        #
        r_user_m_matrix = np.eye(n_dof_m)
        # to consider free DOFs
        for j in range(0, n_nodes):
            for k in range(3, 7):
                if nodes[j, k] == 0:
                    if k == 3:
                        dof_e = j * 2 + 1 - 1
                    elif k == 5:
                        dof_e = (j + 1) * 2 - 1
                    elif k == 4:
                        dof_e = n_nodes * 2 + j * 2 + 1 - 1
                    elif k == 6:
                        dof_e = n_nodes * 2 + (j + 1) * 2 - 1
                    else:
                        raise ValueError("Invalid k value")

                    dof_reg[dof_e, 0] = 0

        # to consider master-slave constraints
        for j in range(0, len(constraints)):
            if len(constraints[j, :]) >= 5:
                # nr of eliminated DOF
                node_e = constraints[j, 0]
                if constraints[j, 1] == 0:
                    dof_e = node_e * 2 + 1 - 1
                elif constraints[j, 1] == 2:
                    dof_e = (node_e + 1) * 2 - 1
                elif constraints[j, 1] == 1:
                    dof_e = n_nodes * 2 + node_e * 2 + 1 - 1
                elif constraints[j, 1] == 3:
                    dof_e = n_nodes * 2 + (node_e + 1) * 2 - 1

                # nr of kept DOF
                node_k = constraints[j, 3]
                if constraints[j, 4] == 0:
                    dof_k = node_k * 2 + 1 - 1
                elif constraints[j, 4] == 2:
                    dof_k = (node_k + 1) * 2 - 1
                elif constraints[j, 4] == 1:
                    dof_k = n_nodes * 2 + node_k * 2 + 1 - 1
                elif constraints[j, 4] == 3:
                    dof_k = n_nodes * 2 + (node_k + 1) * 2 - 1
                else:
                    raise ValueError("Invalid constraints[j, 4] value")

                # to modify r_user_matrix
                r_user_m_matrix[:, dof_k] = r_user_m_matrix[:, dof_k] + constraints[j, 2] * r_user_m_matrix[:, dof_e]
                dof_reg[dof_e, 0] = 0

        # to eliminate columns from r_user_matrix
        k = -1
        r_u_matrix = np.zeros_like(r_user_m_matrix)
        for j in range(0, n_dof_m):
            if dof_reg[j, 0] == 1:
                k = k + 1
                r_u_matrix[:, k] = r_user_m_matrix[:, j]

        r_user_m_matrix = r_u_matrix[:, 0:k]
        r_user_matrix[i * n_dof_m : (i + 1) * n_dof_m, i * k : (i + 1) * k] = r_user_m_matrix

    return r_user_matrix


def mode_constr(
    nodes: np.ndarray, elements: np.ndarray, node_props: np.ndarray, main_nodes: np.ndarray, meta_elements: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """this routine creates the constraint matrices necessary for mode
    separation/classification for each specified half-wave number m_i

    assumptions
      GBT-like assumptions are used
      the cross-section must not be closed and must not contain closed parts

      must check whether 'Warp' works well for any open section !!!

    notes:
      m-el_i-? is positive if the starting nodes of m-el_i-? coincides with
         the given m-nodes, otherwise negative
      nodes types: 1-corner, 2-edge, 3-sub
      sub-nodes numbers are the original one, of course

    Args:
        nodes (np.ndarray): standard parameter
        elements (np.ndarray): standard parameter
        node_props (np.ndarray): array of [original nodes nr, new nodes nr, nr of adj elements,
            nodes type]
        main_nodes (np.ndarray): array of
            [nr, x, z, orig nodes nr, nr of adj meta-elements, m_i-el_i-1, m_i-el_i-2, ...]
        meta_elements (np.ndarray): array of
            [nr, main-nodes-1, main-nodes-2, nr of sub-nodes, sub-no-1, sub-nod-2, ...]

    Returns:
        r_x (np.ndarray): constaint matrices for x dofs
        r_z (np.ndarray): constaint matrices for z dofs
        r_ys (np.ndarray): constaint matrices for y dofs of sub-nodes
        r_yd (np.ndarray): constaint matrices for y dofs of main nodes for distortional buckling
        r_ys (np.ndarray): constaint matrices for y dofs of indefinite (?independent?) main nodes

    S. Adany, Mar 10, 2004
    Z. Li, Jul 10, 2009
    """
    # to create r_x and r_z constraint matrices
    [r_x, r_z] = constr_xz_y(main_nodes, meta_elements)
    #
    # to create r_ys constraint matrix for the y DOFs of sub-nodes
    r_ys = constr_ys_ym(nodes, main_nodes, meta_elements, node_props)
    #
    # to create r_yd for y DOFs of main nodes for distortional buckling
    r_yd = constr_yd_yg(nodes, elements, node_props, r_ys, len(main_nodes))
    #
    # to create r_ud for y DOFs of indefinite main nodes
    r_ud = constr_yu_yd(main_nodes, meta_elements)

    return r_x, r_z, r_yd, r_ys, r_ud


def y_dofs(
    nodes: np.ndarray,
    elements: np.ndarray,
    main_nodes: np.ndarray,
    n_main_nodes: int,
    n_dist_modes: int,
    r_yd: np.ndarray,
    r_ud: np.ndarray,
    sect_props: Sect_Props,
    el_props: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """this routine creates y-DOFs of main nodes for global buckling and
    distortional buckling, however:
       only involves single half-wave number m_i

    assumptions
      GBT-like assumptions are used
      the cross-section must not be closed and must not contain closed parts

    Args:
        nodes (np.ndarray): standard parameter
        elements (np.ndarray): standard parameter
        main_nodes (np.ndarray): nodes of 'meta' cross-section
        n_main_nodes (int): _description_
        n_dist_modes (int): _description_
        r_yd (np.ndarray): constrain matrices
        r_ud (np.ndarray): constrain matrices
        sect_props (Sect_Props): _description_
        el_props (np.ndarray): _description_

    Returns:
        d_y (np.ndarray): y-DOFs of main nodes for global buckling and distortional buckling
            (each column corresponds to a certain mode)

    S. Adany, Mar 10, 2004, modified Aug 29, 2006
    Z. Li, Dec 22, 2009
    """
    w_o = np.zeros((len(nodes), 2))
    w_o[int(elements[0, 1]), 0] = int(elements[0, 1])
    w_no = 0

    # compute the unit warping
    # code from cutwp_prop2:232-249
    for _ in range(0, len(elements)):
        i = 0
        while (np.any(w_o[:, 0] == elements[i, 1]) and np.any(w_o[:, 0] == elements[i, 2])) or (
            not np.any(w_o[:, 0] == elements[i, 1]) and not np.any(w_o[:, 0] == elements[i, 2])
        ):
            i = i + 1
        s_n = int(elements[i, 1])
        f_n = int(elements[i, 2])
        p_o = (
            (nodes[s_n, 1] - sect_props["x0"]) * (nodes[f_n, 2] - sect_props["y0"])
            - (nodes[f_n, 1] - sect_props["x0"]) * (nodes[s_n, 2] - sect_props["y0"])
        ) / el_props[i, 1]
        if w_o[s_n, 0] == 0:
            w_o[s_n, 0] = s_n
            w_o[s_n, 1] = w_o[f_n, 1] - p_o * el_props[i, 1]
        elif w_o[int(elements[i, 2]), 1] == 0:
            w_o[f_n, 0] = f_n
            w_o[f_n, 1] = w_o[s_n, 1] + p_o * el_props[i, 1]
        w_no = w_no + 1 / (2 * sect_props["A"]) * (w_o[s_n, 1] + w_o[f_n, 1]) * elements[i, 3] * el_props[i, 1]
    w_n = w_no - w_o[:, 1]
    # coord. transform. to the principal axes
    phi = sect_props["phi"]
    rot = np.array(
        [
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)],
        ]
    )
    centre_of_gravity = [
        sect_props["cx"],
        sect_props["cy"],
    ] @ rot

    # CALCULATION FOR GLOBAL AND DISTORTIONAL BUCKLING MODES
    #
    # to create y-DOFs of main nodes for global buckling
    d_y = np.zeros((n_main_nodes, 4))
    for i, m_node in enumerate(main_nodes):
        xz_i = [m_node[1], m_node[2]] @ rot
        d_y[i, 0] = 1
        d_y[i, 1] = xz_i[1] - centre_of_gravity[1]
        d_y[i, 2] = xz_i[0] - centre_of_gravity[0]
        d_y[i, 3] = w_n[int(m_node[3])]

    # for i = 1:4
    #     d_y[:, i] = d_y[:, i]/norm(d_y[:, i])
    # end
    # to count the nr of existing global modes
    n_global_modes = 4
    ind = np.ones(4)
    for i in range(0, 4):
        if len(np.nonzero(d_y[:, i])) == 0:
            ind[i] = 0
            n_global_modes = n_global_modes - 1

    # to eliminate zero columns from d_y
    sdy = d_y
    d_y = np.zeros((len(sdy), int(sum(ind))))
    k = 0
    for i in range(0, 4):
        if ind[i] == 1:
            d_y[:, k] = sdy[:, i]
            k = k + 1

    # to create y-DOFs of main nodes for distortional buckling
    if n_dist_modes > 0:
        d_y = np.concatenate((d_y, np.zeros((len(d_y), n_dist_modes))), axis=1)
        # junk = spla.null_space((r_yd*d_y(:, 1:(n_global_modes+1))).conj().T)
        # junk3 = junk.conj().T*r_yd*junk
        r_chol = np.linalg.cholesky(r_yd).T
        junk = spla.null_space((r_chol @ d_y[:, 0:n_global_modes]).conj().T)
        junk2 = np.linalg.solve(r_chol, junk)

        j_junk1 = spla.null_space(junk2.conj().T)
        j_junk2 = spla.null_space(r_ud.conj().T)
        nj1 = len(j_junk1[0])
        nj2 = len(j_junk2[0])
        j_junk3 = j_junk1
        j_junk3[:, nj1 : nj1 + nj2] = j_junk2
        j_junk4 = spla.null_space(j_junk3.conj().T)

        # d_y(:,(n_global_modes+2):(n_global_modes+1+n_dist_modes)) = j_junk4
        junk3 = j_junk4.conj().T @ r_yd @ j_junk4
        # junk3 = junk2.conj().T*junk2
        #
        [_, eigenvectors] = spla.eig(junk3)
        # eigenvalues = diag(eigenvalues)
        # [eigenvalues, index] = sort(eigenvalues)
        # eigenvectors = eigenvectors[:, index]
        d_y[:, n_global_modes : n_global_modes + n_dist_modes] = j_junk4 @ eigenvectors

    return d_y, n_global_modes


def base_vectors(
    d_y: np.ndarray,
    elements: np.ndarray,
    el_props: np.ndarray,
    length: float,
    m_i: float,
    node_props: np.ndarray,
    n_main_nodes: int,
    n_corner_nodes: int,
    n_sub_nodes: int,
    n_global_modes: int,
    n_dist_modes: int,
    n_local_modes: int,
    r_x: np.ndarray,
    r_z: np.ndarray,
    r_p: np.ndarray,
    r_ys: np.ndarray,
    dof_perm: np.ndarray,
) -> np.ndarray:
    """this routine creates the base vectors for global, dist., local and other modes

    assumptions
      GBT-like assumptions are used
      the cross-section must not be closed and must not contain closed parts

      must check whether 'Warp' works well for any open section !!!

    Args:
        d_y (np.ndarray): _description_
        elements (np.ndarray): standard parameter
        el_props (np.ndarray): standard parameter
        length (float): element length
        m_i (float): number of half-waves
        node_props (np.ndarray): some properties of the nodes
        n_main_nodes (int): number of nodes of given type
        n_corner_nodes (int): number of nodes of given type
        n_sub_nodes (int): number of nodes of given type
        n_global_modes (int): number of given modes
        n_dist_modes (int): number of given modes
        n_local_modes (int): number of given modes
        r_x (np.ndarray): constraint matrix
        r_z (np.ndarray): constraint matrix
        r_p (np.ndarray): constraint matrix
        r_ys (np.ndarray): constraint matrix
        dof_perm (np.ndarray):permutation matrix to re-order the DOFs

    Returns:
        np.ndarray: _description_

    S. Adany, Mar 10, 2004, modified Aug 29, 2006
    Z. Li, Dec 22, 2009
    """
    # DATA PREPARATION
    k_m = m_i * np.pi / length
    n_node_props = len(node_props)
    n_dof = 4 * n_node_props  # nro of DOFs
    n_edge_nodes = n_main_nodes - n_corner_nodes
    # zero out
    b_v_m = np.zeros((n_dof, n_dof))

    # CALCULATION FOR GLOBAL AND DISTORTIONAL BUCKLING MODES
    # to add global and dist y DOFs to base vectors
    b_v_m = d_y[:, 0 : n_global_modes + n_dist_modes]
    b_v_m = np.concatenate((b_v_m, np.zeros((n_dof - len(b_v_m), len(b_v_m[0])))), axis=0)
    #
    # to add x DOFs of corner nodes to the base vectors
    # r_x = r_x/k_m
    b_v_m[n_main_nodes : n_main_nodes + n_corner_nodes, 0 : n_global_modes + n_dist_modes] = (
        r_x @ b_v_m[0:n_main_nodes, 0 : n_global_modes + n_dist_modes]
    )
    #
    # to add z DOFs of corner nodes to the base vectors
    # r_z = r_z/k_m
    b_v_m[n_main_nodes + n_corner_nodes : n_main_nodes + 2 * n_corner_nodes, 0 : n_global_modes + n_dist_modes] = (
        r_z @ b_v_m[0:n_main_nodes, 0 : n_global_modes + n_dist_modes]
    )
    #
    # to add other planar DOFs to the base vectors
    b_v_m[n_main_nodes + 2 * n_corner_nodes : n_dof - n_sub_nodes, 0 : n_global_modes + n_dist_modes] = (
        r_p @ b_v_m[n_main_nodes : n_main_nodes + 2 * n_corner_nodes, 0 : n_global_modes + n_dist_modes]
    )
    #
    # to add y DOFs of sub-nodes to the base vector
    b_v_m[n_dof - n_sub_nodes : n_dof, 0 : n_global_modes + n_dist_modes] = (
        r_ys @ b_v_m[0:n_main_nodes, 0 : n_global_modes + n_dist_modes]
    )
    #
    # division by k_m
    b_v_m[n_main_nodes : n_dof - n_sub_nodes, 0 : n_global_modes + n_dist_modes] = (
        b_v_m[n_main_nodes : n_dof - n_sub_nodes, 0 : n_global_modes + n_dist_modes] / k_m
    )
    #
    # norm base vectors
    for i in range(0, n_global_modes + n_dist_modes):
        b_v_m[:, i] = b_v_m[:, i] / np.linalg.norm(b_v_m[:, i])

    # CALCULATION FOR LOCAL BUCKLING MODES
    n_globdist_modes = n_global_modes + n_dist_modes  # nr of global and dist. modes
    b_v_m = np.concatenate((b_v_m, np.zeros((len(b_v_m), n_local_modes))), axis=1)
    # np.zeros
    b_v_m[0:n_dof, n_globdist_modes : n_globdist_modes + n_local_modes] = np.zeros((n_dof, n_local_modes))

    # rot DOFs for main nodes
    b_v_m[3 * n_main_nodes : 4 * n_main_nodes, n_globdist_modes : n_globdist_modes + n_main_nodes] = np.eye(
        n_main_nodes
    )
    #
    # rot DOFs for sub nodes
    if n_sub_nodes > 0:
        b_v_m[
            4 * n_main_nodes + 2 * n_sub_nodes : 4 * n_main_nodes + 3 * n_sub_nodes,
            n_globdist_modes + n_main_nodes : n_globdist_modes + n_main_nodes + n_sub_nodes,
        ] = np.eye(n_sub_nodes)

    # x, z DOFs for edge nodes
    k = 0
    for i in range(0, n_node_props):
        if node_props[i, 3] == 2:
            el_i = np.nonzero(np.any(elements[:, 1] == i) or np.any(elements[:, 2] == i))  # adjacent element
            alfa = el_props[el_i, 2]
            b_v_m[n_main_nodes + 2 * n_corner_nodes + k, n_globdist_modes + n_main_nodes + n_sub_nodes + k] = -np.sin(
                alfa
            )  # x
            b_v_m[
                n_main_nodes + 2 * n_corner_nodes + n_edge_nodes + k, n_globdist_modes + n_main_nodes + n_sub_nodes + k
            ] = np.cos(
                alfa
            )  # z
            k = k + 1

    # x, z DOFs for sub-nodes
    if n_sub_nodes > 0:
        k = 0
        for i in range(0, n_node_props):
            if node_props[i, 3] == 3:
                el_i = np.nonzero(np.any(elements[:, 1] == i) or np.any(elements[:, 2] == i))  # adjacent element
                alfa = el_props[el_i[0], 2]
                b_v_m[4 * n_main_nodes + k, n_globdist_modes + n_main_nodes + n_sub_nodes + n_edge_nodes + k] = -np.sin(
                    alfa
                )  # x
                b_v_m[
                    4 * n_main_nodes + n_sub_nodes + k, n_globdist_modes + n_main_nodes + n_sub_nodes + n_edge_nodes + k
                ] = np.cos(
                    alfa
                )  # z
                k = k + 1

    # CALCULATION FOR OTHER BUCKLING MODES
    #
    # # first among the "others": uniform y
    # b_v_m[1:n_main_nodes,(n_globdist_modes+n_local_modes+1)]
    #     = np.zeros(n_main_nodes, 1)+np.sqrt(1 / (n_main_nodes+n_sub_nodes))
    # b_v_m[(n_dof-n_sub_nodes+1):n_dof,(n_globdist_modes+n_local_modes+1)]
    #     = np.zeros(n_sub_nodes, 1)+np.sqrt(1 / (n_main_nodes+n_sub_nodes))
    #
    ## old way
    # n_other_modes = n_dof - n_globdist_modes - n_local_modes
    # n_elements = len(elements[:, 1])
    # b_v_m[1:n_dof,(n_globdist_modes+n_local_modes+1):(n_globdist_modes+
    #       n_local_modes+2*n_elements)] = np.zeros(n_dof, 2*n_elements)
    # temp_elem = elements[:, 2:3]
    # for i = 1:n_elements
    #     #
    #     alfa = el_props[i, 3]
    #     #
    #     # find nodes on the one side of the current element
    #     n_nod = 1
    #     nods=[]
    #     nods(1) = elements[i, 2]
    #     temp_elem = elements[:, 2:3]
    #     temp_elem[i, 1] = 0
    #     temp_elem[i, 2] = 0
    #     new = 1
    #     while new>0
    #         new = 0
    #         for j = 1:n_elements
    #             for k_local = 1:len(nods)
    #                 if (nods(k_local) == temp_elem[j, 1])
    #                     n_nod = n_nod+1
    #                     new = new+1
    #                     nods(n_nod) = temp_elem[j, 2]
    #                     temp_elem[j, 1] = 0
    #                     temp_elem[j, 2] = 0
    #
    #                 if (nods(k_local) == temp_elem[j, 2])
    #                     n_nod = n_nod+1
    #                     new = new+1
    #                     nods(n_nod) = temp_elem[j, 1]
    #                     temp_elem[j, 1] = 0
    #                     temp_elem[j, 2] = 0
    #
    #     #
    #     # create the base-vectors for membrane SHEAR modes
    #     s = np.sqrt(1/n_nod)
    #     for j = 1:n_nod
    #         old_dof_y = 2 * (nods[j])
    #         b_v_m[old_dof_y,(n_globdist_modes+n_local_modes+i)] = s
    #
    #     # create the base-vectors for membrane TRANSVERSE modes
    #     for j = 1:n_nod
    #         old_dof_x = 2 * (nods[j])-1
    #         old_dof_z = 2*n_node_props+2 * (nods[j])-1
    #         b_v_m[old_dof_x,(n_globdist_modes+n_local_modes+n_elements+i)] = s*np.cos(alfa)
    #         b_v_m[old_dof_z,(n_globdist_modes+n_local_modes+n_elements+i)] = s*np.sin(alfa)
    #
    #
    # end
    ## new way
    n_elements = len(elements)
    b_v_m = np.concatenate((b_v_m, np.zeros((len(b_v_m), 2 * n_elements))), axis=1)
    for i, elem in enumerate(elements):
        alfa = el_props[i, 2]

        # find nodes on the one side of the current element
        n_nod1 = int(elem[1])
        n_nod2 = int(elem[2])

        # create the base-vectors for membrane SHEAR modes
        b_v_m[(n_nod1 - 1) * 2, n_globdist_modes + n_local_modes + i] = 0.5
        b_v_m[(n_nod2 - 1) * 2, n_globdist_modes + n_local_modes + i] = -0.5

        # create the base-vectors for membrane TRANSVERSE modes
        b_v_m[(n_nod1 - 1) * 2, n_globdist_modes + n_local_modes + n_elements + i] = -0.5 * np.cos(alfa)
        b_v_m[(n_nod2 - 1) * 2, n_globdist_modes + n_local_modes + n_elements + i] = 0.5 * np.cos(alfa)
        b_v_m[2 * n_node_props + (n_nod1 - 1) * 2, n_globdist_modes + n_local_modes + n_elements + i] = 0.5 * np.sin(
            alfa
        )
        b_v_m[2 * n_node_props + (n_nod2 - 1) * 2, n_globdist_modes + n_local_modes + n_elements + i] = -0.5 * np.sin(
            alfa
        )

    # RE_ORDERING DOFS
    b_v_m[:, 0 : n_globdist_modes + n_local_modes] = dof_perm @ b_v_m[:, 0 : n_globdist_modes + n_local_modes]

    return b_v_m


def constr_xz_y(main_nodes: np.ndarray, meta_elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """this routine creates the constraint matrix, Rxz, that defines relationship
    between x, z displacements DOFs [for internal main nodes, referred also as corner nodes]
    and the longitudinal y displacements DOFs [for all the main nodes]
    if GBT-like assumptions are used

    to make this routine length-independent, Rxz is not multiplied here by
    (1/k_m), thus it has to be multiplied outside of this routine!

    additional assumption: cross section is opened!

    note:
        m-el_i-? is positive if the starting nodes of m-el_i-? coincides with
        the given m-nodes, otherwise negative

    Args:
        main_nodes (np.ndarray): array of
            [nr, x, z, orig nodes nr, nr of adj meta-elements, m-el_i-1, m-el_i-2, ...]
        meta_elements (np.ndarray): array of
            [nr, main-nodes-1, main-nodes-2, nr of sub-nodes, sub-no-1, sub-nod-2, ...]

    Returns:
        r_x (np.ndarray): x dof restraints
        r_z (np.ndarray): z dof restraints

    S. Adany, Feb 05, 2004
    """
    meta_elements_data = np.zeros((len(meta_elements), 5))
    for i, m_elem in enumerate(meta_elements):
        node1 = int(m_elem[1])
        node2 = int(m_elem[2])
        x_1 = main_nodes[node1, 1]
        x_2 = main_nodes[node2, 1]
        z_1 = main_nodes[node1, 2]
        z_2 = main_nodes[node2, 2]
        b_i = np.sqrt((x_2 - x_1) ** 2 + (z_2 - z_1) ** 2)
        a_i = np.arctan2(z_2 - z_1, x_2 - x_1)
        s_i = (z_2 - z_1) / b_i
        c_i = (x_2 - x_1) / b_i
        meta_elements_data[i, 0] = b_i  # elements width, b_strip
        meta_elements_data[i, 1] = 1 / meta_elements_data[i, 0]  # 1/b_strip
        meta_elements_data[i, 2] = a_i  # elements inclination
        meta_elements_data[i, 3] = s_i  # np.sin
        meta_elements_data[i, 4] = c_i  # np.cos
    #     meta_elements_data[i, 5] = s_i/b_i # np.sin/b_strip
    #     meta_elements_data[i, 6] = c_i/b_i # np.cos/b_strip

    # to count the number of corner nodes, and of main nodes
    n_main_nodes = len(main_nodes[:, 0])
    n_corner_nodes = 0
    for m_node in main_nodes:
        if m_node[4] > 1:
            n_corner_nodes = n_corner_nodes + 1

    r_x = np.zeros((n_corner_nodes, n_main_nodes))
    r_z = np.zeros((n_corner_nodes, n_main_nodes))
    k = 0
    for i, m_node in enumerate(main_nodes):
        if m_node[4] > 1:
            # to select two non-parallel meta-elements (elem1, elem2)
            elem1 = int(m_node[5])
            elem1_flag = int(round((m_node[5] - elem1) * 10))
            j = 6
            while np.sin(meta_elements_data[int(np.real(m_node[j])), 2] - meta_elements_data[elem1, 2]) == 0:
                j = j + 1

            elem2 = int(m_node[j])
            elem2_flag = int(round((m_node[j] - elem2) * 10))

            # to define main-nodes that play (order: main_nodes1, main_nodes2, main_nodes3)
            main_nodes2 = int(i)
            main_nodes1 = int(meta_elements[elem1, elem1_flag])
            main_nodes3 = int(meta_elements[elem2, elem2_flag])

            # to calculate elements of Rxz matrix
            r_1 = meta_elements_data[elem1, 1]
            alfa1 = meta_elements_data[elem1, 2]
            sin1 = meta_elements_data[elem1, 3]
            cos1 = meta_elements_data[elem1, 4]
            if elem1_flag == 2:
                alfa1 = alfa1 - np.pi
                sin1 = -sin1
                cos1 = -cos1

            r_2 = meta_elements_data[elem2, 1]
            alfa2 = meta_elements_data[elem2, 2]
            sin2 = meta_elements_data[elem2, 3]
            cos2 = meta_elements_data[elem2, 4]
            if elem2 == 1:
                alfa2 = alfa2 - np.pi
                sin2 = -sin2
                cos2 = -cos2

            det = np.sin(alfa2 - alfa1)

            # to form Rxz matrix
            r_x[k, main_nodes1] = sin2 * r_1 / det
            r_x[k, main_nodes2] = (-sin1 * r_2 - sin2 * r_1) / det
            r_x[k, main_nodes3] = sin1 * r_2 / det

            r_z[k, main_nodes1] = -cos2 * r_1 / det
            r_z[k, main_nodes2] = (cos1 * r_2 + cos2 * r_1) / det
            r_z[k, main_nodes3] = -cos1 * r_2 / det

            k = k + 1

    return r_x, r_z


def constr_planar_xz(
    nodes: np.ndarray,
    elements: np.ndarray,
    props: np.ndarray,
    node_props: np.ndarray,
    dof_perm: np.ndarray,
    m_i: float,
    length: float,
    B_C: str,
    el_props: np.ndarray,
) -> np.ndarray:
    """this routine creates the constraint matrix, r_p, that defines relationship
    between x, z DOFs of any non-corner nodes + teta DOFs of all nodes,
    and the x, z displacements DOFs of corner nodes
    if GBT-like assumptions are used

    Args:
        nodes (np.ndarray): standard parameter
        elements (np.ndarray): standard parameter
        props (np.ndarray): standard parameter
        node_props (np.ndarray): array of [original nodes nr, new nodes nr, nr of adj elements,
            nodes type]
        dof_perm (np.ndarray): permutation matrix, so that
            (orig-displacements-vect) = (dof_perm) � (new-displacements - vector)
        m_i (float): _description_
        length (float): element length
        B_C (str): standard parameter
        el_props (np.ndarray): standard parameter

    Returns:
        r_p (np.ndarray):constraint matrix, r_p, that defines relationship
            between x, z DOFs of any non-corner nodes + teta DOFs of all nodes,
            and the x, z displacements DOFs of corner nodes

    S. Adany, Feb 06, 2004
    Z. Li, Jul 10, 2009
    """
    # to count corner-, edge- and sub-nodes
    n_node_props = len(node_props)
    n_corner_nodes = 0
    n_edge_nodes = 0
    n_sub_nodes = 0
    for n_prop in node_props:
        if n_prop[3] == 1:
            n_corner_nodes = n_corner_nodes + 1

        if n_prop[3] == 2:
            n_edge_nodes = n_edge_nodes + 1

        if n_prop[3] == 3:
            n_sub_nodes = n_sub_nodes + 1

    n_main_nodes = n_corner_nodes + n_edge_nodes  # nr of main nodes

    n_dof = 4 * n_node_props  # nro of DOFs

    # to create the full global stiffness matrix (for transverse bending)
    K_global = analysis.kglobal_transv(
        nodes=nodes, elements=elements, el_props=el_props, props=props, length=length, B_C=B_C, m_i=m_i
    )

    # to re-order the DOFs
    K_global = dof_perm.conj().T @ K_global @ dof_perm

    # to have partitions of K_global
    k_global_pp = K_global[
        n_main_nodes + 2 * n_corner_nodes : n_dof - n_sub_nodes, n_main_nodes + 2 * n_corner_nodes : n_dof - n_sub_nodes
    ]
    k_global_pc = K_global[
        n_main_nodes + 2 * n_corner_nodes : n_dof - n_sub_nodes, n_main_nodes : n_main_nodes + 2 * n_corner_nodes
    ]

    # to form the constraint matrix
    # [r_p]=-inv(k_global_pp) * k_global_pc

    r_p = -np.linalg.solve(k_global_pp, k_global_pc)

    return r_p


def constr_yd_yg(
    nodes: np.ndarray, elements: np.ndarray, node_props: np.ndarray, r_ys: np.ndarray, n_main_nodes: int
) -> np.ndarray:
    """this routine creates the constraint matrix, r_yd, that defines relationship
    between base vectors for distortional buckling,
    and base vectors for global buckling,
    but for y DOFs of main nodes only

    Args:
        nodes (np.ndarray): standard parameter
        elements (np.ndarray): standard parameter
        node_props (np.ndarray): array of [original nodes nr, new nodes nr, nr of adj elements,
            nodes type]
        r_ys (np.ndarray): constraint matrix, see function 'constr_ys_ym'
        n_main_nodes (int): nr of main nodes

    Returns:
        r_yd (np.ndarray): constraint matrix for y DOFs in distortional buckling

    S. Adany, Mar 04, 2004
    """
    n_nodes = len(nodes)
    a_matrix = np.zeros((n_nodes, n_nodes))
    for elem in elements:
        node1 = int(elem[1])
        node2 = int(elem[2])
        d_x = nodes[node2, 1] - nodes[node1, 1]
        d_z = nodes[node2, 2] - nodes[node1, 2]
        d_area = np.sqrt(d_x * d_x + d_z * d_z) * elem[3]
        ind = np.nonzero(node_props[:, 0] == node1)
        node1 = int(node_props[ind, 1])
        ind = np.nonzero(node_props[:, 0] == node2)
        node2 = int(node_props[ind, 1])
        a_matrix[node1, node1] = a_matrix[node1, node1] + 2 * d_area
        a_matrix[node2, node2] = a_matrix[node2, node2] + 2 * d_area
        a_matrix[node1, node2] = a_matrix[node1, node2] + d_area
        a_matrix[node2, node1] = a_matrix[node2, node1] + d_area

    r_ysm = np.zeros((n_nodes, n_main_nodes))
    r_ysm[0:n_main_nodes, 0:n_main_nodes] = np.eye(n_main_nodes)
    r_ysm[n_main_nodes:n_nodes, 0:n_main_nodes] = r_ys
    r_yd = r_ysm.conj().T @ a_matrix @ r_ysm

    return r_yd


def constr_ys_ym(
    nodes: np.ndarray, main_nodes: np.ndarray, meta_elements: np.ndarray, node_props: np.ndarray
) -> np.ndarray:
    """this routine creates the constraint matrix, r_ys, that defines relationship
    between y DOFs of sub-nodes,
    and the y displacements DOFs of main nodes
    by linear interpolation

    Args:
        nodes (np.ndarray): standard parameter
        main_nodes (np.ndarray): array of
            [nr, x, z, orig nodes nr, nr of adj meta-elements, m-el_i-1, m-el_i-2, ...]
        meta_elements (np.ndarray): array of
            [nr, main-nodes-1, main-nodes-2, nr of sub-nodes, sub-no-1, sub-nod-2, ...]
        node_props (np.ndarray): array of [original nodes nr, new nodes nr, nr of adj elements,
            nodes type]

    Returns:
        r_ys (np.ndarray): constraint matrix for y DOFs of sub-nodes

    S. Adany, Feb 06, 2004
    """
    n_sub_nodes = 0
    for n_prop in node_props:
        if n_prop[3] == 3:
            n_sub_nodes = n_sub_nodes + 1

    n_main_nodes = len(main_nodes)

    r_ys = np.zeros((n_sub_nodes, n_main_nodes))

    for m_elem in meta_elements:
        if m_elem[3] > 0:
            nod1 = int(main_nodes[int(m_elem[1]), 3])
            nod3 = int(main_nodes[int(m_elem[2]), 3])
            x_1 = nodes[nod1, 1]
            x_3 = nodes[nod3, 1]
            z_1 = nodes[nod1, 2]
            z_3 = nodes[nod3, 2]
            b_m = np.sqrt((x_3 - x_1) ** 2 + (z_3 - z_1) ** 2)
            n_new1 = int(node_props[nod1, 1])
            n_new3 = int(node_props[nod3, 1])
            for j in range(0, int(m_elem[3])):
                nod2 = int(m_elem[j + 4])
                x_2 = nodes[nod2, 1]
                z_2 = nodes[nod2, 2]
                b_s = np.sqrt((x_2 - x_1) ** 2 + (z_2 - z_1) ** 2)
                n_new2 = int(node_props[nod2, 1])
                r_ys[n_new2 - n_main_nodes, n_new1] = (b_m - b_s) / b_m
                r_ys[n_new2 - n_main_nodes, n_new3] = b_s / b_m

    return r_ys


def constr_yu_yd(main_nodes: np.ndarray, meta_elements: np.ndarray) -> np.ndarray:
    """this routine creates the constraint matrix, r_ud, that defines relationship
    between y displacements DOFs of indefinite main nodes
    and the y displacements DOFs of definite main nodes
    (definite main nodes = those main nodes which unambiguously define the y displacements pattern
     indefinite main nodes = those nodes the y DOF of which can be calculated
                             from the y DOF of definite main nodes
     note: for open sections with one single branch only there are no indefinite nodes)

    important assumption: cross section is opened!

    note:
        m-el_i-? is positive if the starting nodes of m-el_i-? coincides with
        the given m-nodes, otherwise negative

    Args:
        main_nodes (np.ndarray): array of
            [nr, x, z, orig nodes nr, nr of adj meta-elements, m-el_i-1, m-el_i-2, ...]
        meta_elements (np.ndarray): array of
            [nr, main-nodes-1, main-nodes-2, nr of sub-nodes, sub-no-1, sub-nod-2, ...]

    Returns:
        r_ud (np.ndarray): constraint matrix between definite and indefinite main nodes

    S. Adany, Mar 10, 2004
    """
    # to calculate some data of main elements (stored in meta_elements_data)
    meta_elements_data = np.zeros((len(meta_elements), 5))
    for i, m_elem in enumerate(meta_elements):
        node1 = int(m_elem[1])
        node2 = int(m_elem[2])
        x_1 = main_nodes[node1, 1]
        x_2 = main_nodes[node2, 1]
        z_1 = main_nodes[node1, 2]
        z_2 = main_nodes[node2, 2]
        b_i = np.sqrt((x_2 - x_1) ** 2 + (z_2 - z_1) ** 2)
        a_i = np.arctan2(z_2 - z_1, x_2 - x_1)
        s_i = (z_2 - z_1) / b_i
        c_i = (x_2 - x_1) / b_i
        meta_elements_data[i, 0] = b_i  # elements width, b_strip
        meta_elements_data[i, 1] = 1 / meta_elements_data[i, 0]  # 1/b_strip
        meta_elements_data[i, 2] = a_i  # elements inclination
        meta_elements_data[i, 3] = s_i  # np.sin
        meta_elements_data[i, 4] = c_i  # np.cos
    #     meta_elements_data[i, 5] = s_i/b_i # np.sin/b_strip
    #     meta_elements_data[i, 6] = c_i/b_i # np.cos/b_strip

    # to count the number of corner nodes, and of main nodes
    n_main_nodes = len(main_nodes)
    n_corner_nodes = 0
    for m_node in main_nodes:
        if m_node[4] > 1:
            n_corner_nodes = n_corner_nodes + 1

    # to register definite and indefinite nodes
    node_reg = np.ones((n_main_nodes, 1))
    for i, m_node in enumerate(main_nodes):
        if m_node[4] > 2:
            # to select two non-parallel meta-elements (elem1, elem2)
            elem1 = int(np.real(m_node[5]))
            j = 6
            while np.sin(meta_elements_data[int(np.real(m_node[j])), 2] - meta_elements_data[elem1, 2]) == 0:
                j = j + 1

            elem2 = int(np.real(m_node[j]))

            # to set far nodes of adjacent unselected elements to indefinite (node_reg == 0)
            for j in range(1, m_node[4]):
                elem3 = int(np.real(m_node[j + 5]))
                if elem3 != elem2:
                    if meta_elements[elem3, 1] != i:
                        node_reg[int(meta_elements[elem3, 1])] = 0
                    else:
                        node_reg[int(meta_elements[elem3, 2])] = 0

    # to create r_ud matrix
    r_ud = np.zeros((n_main_nodes, n_main_nodes))

    # for definite nodes
    for i in range(0, n_main_nodes):
        if node_reg[i] == 1:
            r_ud[i, i] = 1

    # for indefinite nodes
    for i, m_node in enumerate(main_nodes):
        if m_node[4] > 2:
            # to select the two meta-elements that play (elem1, elem2)
            elem1 = int(m_node[5])
            elem1_flag = m_node[5] - elem1
            j = 6
            while np.sin(meta_elements_data[int(np.real(m_node[j])), 2] - meta_elements_data[elem1, 2]) == 0:
                j = j + 1

            elem2 = int(m_node[j])
            elem2_flag = m_node[j] - elem2

            # to define main-nodes that play (order: main_nodes1, main_nodes2, main_nodes3)
            main_nodes2 = int(i)
            if elem1_flag == 0.1:
                main_nodes1 = int(meta_elements[elem1, 2])
            else:
                main_nodes1 = int(meta_elements[elem1, 1])

            if elem2_flag == 0.1:
                main_nodes3 = int(meta_elements[elem2, 2])
            else:
                main_nodes3 = int(meta_elements[elem2, 1])

            # to calculate data necessary for r_ud
            r_1 = meta_elements_data[elem1, 1]
            alfa1 = meta_elements_data[elem1, 2]
            sin1 = meta_elements_data[elem1, 3]
            cos1 = meta_elements_data[elem1, 4]
            if elem1 > 0:
                alfa1 = alfa1 - np.pi
                sin1 = -sin1
                cos1 = -cos1

            r_2 = meta_elements_data[elem2, 1]
            alfa2 = meta_elements_data[elem2, 2]
            sin2 = meta_elements_data[elem2, 3]
            cos2 = meta_elements_data[elem2, 4]
            if elem2 < 0:
                alfa2 = alfa2 - np.pi
                sin2 = -sin2
                cos2 = -cos2

            det = np.sin(alfa2 - alfa1)

            r_mat = np.array([[r_1, -r_1, 0], [0, r_2, -r_2]])
            c_s = np.array([[sin2, -sin1], [-cos2, cos1]])
            csr = c_s @ r_mat / det

            for j in range(1, main_nodes[i, 4]):
                elem3 = int(m_node[j + 5])
                elem3_flag = m_node[j + 5] - elem3
                if elem3 != elem2:
                    if meta_elements[elem3, 1] != i:
                        main_nodes4 = int(meta_elements[elem3, 1])
                    else:
                        main_nodes4 = int(meta_elements[elem3, 2])

                    r_3 = meta_elements_data[elem3, 1]
                    alfa3 = meta_elements_data[elem3, 2]
                    sin3 = meta_elements_data[elem3, 3]
                    cos3 = meta_elements_data[elem3, 4]
                    if elem3_flag == 0.2:
                        alfa3 = alfa3 - np.pi
                        sin3 = -sin3
                        cos3 = -cos3

                    rud = -1 / r_3 * np.array([cos3, sin3]) @ csr
                    rud[0, 1] = rud[0, 1] + 1
                    r_ud[main_nodes4, main_nodes1] = rud[0, 0]
                    r_ud[main_nodes4, main_nodes2] = rud[0, 1]
                    r_ud[main_nodes4, main_nodes3] = rud[0, 2]

    # to completely eliminate indefinite nodes from r_ud (if necessary)
    k = 1
    while k == 1:
        k = 0
        for i in range(0, n_main_nodes):
            if node_reg[i] == 0:
                if np.nonzero(r_ud[:, i]):
                    k = 1
                    indices = np.nonzero(r_ud[:, i])
                    for ind in indices:
                        r_ud[ind, :] = r_ud[ind, :] + r_ud[i, :] * r_ud[ind, i]
                        r_ud[ind, i] = 0

    return r_ud


def base_properties(
    nodes: np.ndarray, elements: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int, int, np.ndarray]:
    """this routine creates all the data for defining the base vectors from the
    cross section properties

    Args:
        nodes (np.ndarray): standard parameters
        elements (np.ndarray): standard parameters

    Returns:
        main_nodes (np.ndarray): array of
            [nr, x, z, orig nodes nr, nr of adj meta-elements, m-el_i-1, m-el_i-2, ...]
        meta_elements (np.ndarray): array of
            [nr, main-nodes-1, main-nodes-2, nr of sub-nodes, sub-no-1, sub-nod-2, ...]
        node_props (np.ndarray): array of
            [original nodes nr, new nodes nr, nr of adj elements, nodes type]
        n_main_nodes (int): number of main nodes
        n_corner_nodes (int): number of corner nodes
        n_sub_nodes (int): number of sub-nodes
        n_dist_modes (int): number of distortional modes
        n_local_modes (int): number of local modes
        dof_perm (np.ndarray): permutation matrix, so that
            (orig-displacements-vect) = (dof_perm) � (new-displacements-vector)

    S. Adany, Aug 28, 2006
    B. Schafer, Aug 29, 2006
    Z. Li, Dec 22, 2009
    """
    [main_nodes, meta_elements, node_props] = meta_elems(nodes=nodes, elements=elements)
    [n_main_nodes, n_corner_nodes, n_sub_nodes] = node_class(node_props=node_props)
    [n_dist_modes, n_local_modes] = mode_nr(n_main_nodes, n_corner_nodes, n_sub_nodes, main_nodes)
    dof_perm = dof_ordering(node_props)

    return (
        main_nodes,
        meta_elements,
        node_props,
        n_main_nodes,
        n_corner_nodes,
        n_sub_nodes,
        n_dist_modes,
        n_local_modes,
        dof_perm,
    )


def meta_elems(nodes: np.ndarray, elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """this routine re-organises the basic input data
    to eliminate internal subdividing nodes
    to form meta-elements (corner-to-corner or corner-to-free edge)
    to identify main nodes (corner nodes and free edge nodes)

    important assumption: cross section is opened!

    note:
      m-el_i-? is positive if the starting nodes of m-el_i-? coincides with
         the given m-nodes, otherwise negative
      nodes types: 1-corner, 2-edge, 3-sub
      sub-nodes numbers are the original ones, of course

    Args:
        nodes (np.ndarray): standard parameter
        elements (np.ndarray): standard parameter

    Returns:
        main_nodes (np.ndarray): main nodes (i.e. corner and free edge nodes) array of
            [nr, x, z, orig nodes nr, nr of adj meta-elements, m-el_i-1, m-el_i-2, ...]?
        meta_elements (np.ndarray): elements connecting main nodes array of
            [nr, main-nodes-1, main-nodes-2, nr of sub-nodes, sub-no-1, sub-nod-2, ...]
        node_props (np.ndarray): properties of main nodes array of
            [original nodes nr, new nodes nr, nr of adj elements, nodes type]

    S. Adany, Feb 06, 2004
    """
    n_nodes = len(nodes)
    #
    # to count nr of elements connecting to each nodes
    # + register internal nodes to be eliminated
    # + set nodes type (node_props[:, 4])
    node_props = np.zeros((n_nodes, 4))
    node_props[:, 0] = nodes[:, 0]
    for i in range(0, n_nodes):
        els = []
        for j, elem in enumerate(elements):
            if i in (elem[1], elem[2]):
                els.append(j)  # zli: element no. containing this nodes

        node_props[i, 2] = len(els)
        if len(els) == 1:
            node_props[i, 3] = 2

        if len(els) >= 2:
            node_props[i, 3] = 1

        if len(els) == 2:
            n_1 = i
            n_2 = int(elements[els[0], 1])
            # zli: the first nodes of the first elements containing this nodes
            if n_2 == n_1:
                n_2 = int(elements[els[0], 2])

            n_3 = int(elements[els[1], 1])
            if n_3 == n_1:
                n_3 = int(elements[els[1], 2])

            a_1 = np.arctan2(nodes[n_2, 2] - nodes[n_1, 2], nodes[n_2, 1] - nodes[n_1, 1])  # ?
            a_2 = np.arctan2(nodes[n_1, 2] - nodes[n_3, 2], nodes[n_1, 1] - nodes[n_3, 1])
            if abs(a_1 - a_2) < 1e-7:
                node_props[i, 2] = 0
                node_props[i, 3] = 3

    # to create meta-elements (with the original nodes numbers)
    meta_elements_temp = np.zeros((len(elements), 5))
    meta_elements_temp[:, 0:3] = elements[:, 0:3]
    for i in range(0, n_nodes):
        if node_props[i, 2] == 0:
            els = []
            for j, m_elem in enumerate(meta_elements_temp):
                if i in (m_elem[1], m_elem[2]):
                    els.append(j)

            node1 = int(meta_elements_temp[els[0], 1])
            if node1 == i:
                node1 = int(meta_elements_temp[els[0], 2])

            node2 = int(meta_elements_temp[els[1], 1])
            if node2 == i:
                node2 = int(meta_elements_temp[els[1], 2])

            meta_elements_temp[els[0], 1] = node1
            meta_elements_temp[els[0], 2] = node2
            meta_elements_temp[els[1], 1] = -1
            meta_elements_temp[els[1], 2] = -1
            meta_elements_temp[els[0], 3] = meta_elements_temp[els[0], 3] + 1  # zli:
            if 3 + meta_elements_temp[els[0], 3] >= len(meta_elements_temp[0]):
                meta_elements_temp = np.c_[meta_elements_temp, np.zeros(len(meta_elements_temp))]
            meta_elements_temp[els[0], int(3 + meta_elements_temp[els[0], 3])] = i  # zli:deleted elements no.

    # to eliminate disappearing elements (nodes numbers are still the original ones!)
    n_meta_elements = 0  # nr of meta-elements
    meta_elements_list = []
    for m_elem_t in meta_elements_temp:
        if m_elem_t[1] != -1 and m_elem_t[2] != -1:
            meta_elements_list.append(m_elem_t)
            meta_elements_list[-1][0] = n_meta_elements
            n_meta_elements = n_meta_elements + 1
    meta_elements = np.array(meta_elements_list)

    # to create array of main-nodes
    # (first and fourth columns assign the new vs. original numbering,
    # + node_assign tells the original vs. new numbering)
    n_main_nodes = 0  # nr of main nodes
    main_nodes_list = []
    for i, node in enumerate(nodes):
        if node_props[i, 2] != 0:
            main_nodes_list.append([n_main_nodes, node[1], node[2], i, node_props[i, 2]])
            node_props[i, 1] = n_main_nodes
            n_main_nodes = n_main_nodes + 1
    main_nodes = np.array(main_nodes_list)

    # to re-number nodes in the array meta_elements (only for main nodes, of course)
    for i, n_props in enumerate(node_props):
        if node_props[i, 2] != 0:
            for m_elem in meta_elements:
                if m_elem[1] == i:
                    m_elem[1] = n_props[1]

                if m_elem[2] == i:
                    m_elem[2] = n_props[1]

    # to assign meta-elements to main-nodes
    for i in range(0, n_main_nodes):
        k = 5
        for j, m_elem in enumerate(meta_elements):
            if m_elem[1] == i:
                if len(main_nodes[0]) <= k:
                    main_nodes = np.c_[main_nodes, np.zeros(n_main_nodes)]
                main_nodes[i, k] = j + 0.2
                k = k + 1

            if m_elem[2] == i:
                if len(main_nodes[0]) <= k:
                    main_nodes = np.c_[main_nodes, np.zeros(n_main_nodes)]
                main_nodes[i, k] = j + 0.1
                k = k + 1

    # to finish node_assign with the new numbers of subdividing nodes
    n_sub_nodes = 0  # nr of subdividing nodes
    for n_prop in node_props:
        if n_prop[2] == 0:
            n_prop[1] = n_main_nodes + n_sub_nodes
            n_sub_nodes = n_sub_nodes + 1

    return main_nodes, meta_elements, node_props


def mode_nr(n_main_nodes: int, n_corner_nodes: int, n_sub_nodes: int, main_nodes: np.ndarray) -> Tuple[int, int]:
    """this routine determines the number of distortional and local buckling modes
    if GBT-like assumptions are used

    Args:
        n_main_nodes (int): number of main nodes
        n_corner_nodes (int): number of corner nodes
        n_sub_nodes (int): number of sub nodes
        main_nodes (np.ndarray): array of
            [nr, x, z, orig nodes nr, nr of adj meta-elements, m-el_i-1, m-el_i-2, ...]

    Returns:
        n_dist_modes (int): number of distortional modes
        n_local_modes (int): number of local modes

    S. Adany, Feb 09, 2004
    """
    # to count the number of distortional modes
    n_dist_modes = n_main_nodes - 4
    for i in range(0, n_main_nodes):
        if main_nodes[i, 4] > 2:
            n_dist_modes = n_dist_modes - (main_nodes[i, 4] - 2)

    n_dist_modes = max(0, n_dist_modes)

    # to count the number of local modes
    n_edge_nodes = n_main_nodes - n_corner_nodes  # nr of edge nodes
    n_local_modes = n_main_nodes + 2 * n_sub_nodes + n_edge_nodes

    return n_dist_modes, n_local_modes


def dof_ordering(node_props: np.ndarray) -> np.ndarray:
    """this routine re-orders the DOFs,
    according to the need of forming shape vectors for various buckling modes

    notes:
    (1)  nodes types: 1-corner, 2-edge, 3-sub
    (2)  the re-numbering of long. displacements. DOFs of main nodes, which may be
         necessary for dist. buckling, is not included here but handled
         separately when forming Ry constraint matrix

    Args:
        node_props (np.ndarray): array of [original nodes nr, new nodes nr, nr of adj elements,
            nodes type]

    Returns:
        dof_perm (np.ndarray): permutation matrix, so that
            (orig-displacements-vect) = (dof_perm) � (new-displacements-vector)

    S. Adany, Feb 06, 2004
    """
    # to count corner-, edge- and sub-nodes
    n_node_props = len(node_props)
    n_corner_nodes = 0
    n_edge_nodes = 0
    n_sub_nodes = 0
    for n_prop in node_props:
        if n_prop[3] == 1:
            n_corner_nodes = n_corner_nodes + 1
        if n_prop[3] == 2:
            n_edge_nodes = n_edge_nodes + 1
        if n_prop[3] == 3:
            n_sub_nodes = n_sub_nodes + 1

    n_main_nodes = n_corner_nodes + n_edge_nodes  # nr of main nodes

    # to form permutation matrix
    dof_perm = np.zeros((4 * n_node_props, 4 * n_node_props))

    # x DOFs
    i_c = 0
    i_e = 0
    i_s = 0
    for i, n_prop in enumerate(node_props):
        if n_prop[3] == 1:  # corner nodes
            dof_perm[2 * i, n_main_nodes + i_c] = 1
            i_c = i_c + 1
        if n_prop[3] == 2:  # edge nodes
            dof_perm[2 * i, n_main_nodes + 2 * n_corner_nodes + i_e] = 1
            i_e = i_e + 1
        if n_prop[3] == 3:  # sub nodes
            dof_perm[2 * i, 4 * n_main_nodes + i_s] = 1
            i_s = i_s + 1

    # y DOFs
    i_c = 0
    i_s = 0
    for i, n_prop in enumerate(node_props):
        if n_prop[3] == 1 or n_prop[3] == 2:  # corner or edge nodes
            dof_perm[2 * i + 1, i_c] = 1
            i_c = i_c + 1
        if n_prop[3] == 3:  # sub nodes
            dof_perm[2 * i + 1, 4 * n_main_nodes + 3 * n_sub_nodes + i_s] = 1
            i_s = i_s + 1

    # z DOFs
    i_c = 0
    i_e = 0
    i_s = 0
    for i, n_prop in enumerate(node_props):
        if n_prop[3] == 1:  # corner nodes
            dof_perm[2 * n_node_props + 2 * i, n_main_nodes + n_corner_nodes + i_c] = 1
            i_c = i_c + 1
        if n_prop[3] == 2:  # edge nodes
            dof_perm[2 * n_node_props + 2 * i, n_main_nodes + 2 * n_corner_nodes + n_edge_nodes + i_e] = 1
            i_e = i_e + 1
        if n_prop[3] == 3:  # sub nodes
            dof_perm[2 * n_node_props + 2 * i, 4 * n_main_nodes + n_sub_nodes + i_s] = 1
            i_s = i_s + 1

    # teta DOFs
    i_c = 0
    i_s = 0
    for i, n_prop in enumerate(node_props):
        if n_prop[3] == 1 or n_prop[3] == 2:  # corner or edge nodes
            dof_perm[2 * n_node_props + 2 * i + 1, 3 * n_main_nodes + i_c] = 1
            i_c = i_c + 1
        if n_prop[3] == 3:  # sub nodes
            dof_perm[2 * n_node_props + 2 * i + 1, 4 * n_main_nodes + 2 * n_sub_nodes + i_s] = 1
            i_s = i_s + 1

    return dof_perm


def classify(
    props: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    lengths: np.ndarray,
    shapes: np.ndarray,
    GBT_con: GBT_Con,
    B_C: str,
    m_all: np.ndarray,
    sect_props: Sect_Props,
) -> List[np.ndarray]:
    """modal classificaiton

    Args:
        props (np.ndarray): [mat_num stiff_x E_y nu_x nu_y G_bulk] 6 x nmats
        nodes (np.ndarray): [nodes# x z dof_x dof_z dof_y dofrot stress] n_nodes x 8
        elements (np.ndarray): [elements# node_i node_j thick mat_num] n_elements x 5
        lengths (np.ndarray): lengths to be analysed
        shapes (np.ndarray): array of mode shapes dof x lengths x mode
        GBT_con (GBT_Con): _description_
            method:
                method = 1 = vector norm
                method = 2 = strain energy norm
                method = 3 = work norm
        B_C (str): _description_
        m_all (np.ndarray): _description_
        sect_props (Sect_Props): _description_

    Returns:
        clas (List[np.ndarray]): array of # classification

    BWS August 29, 2006
    modified SA, Oct 10, 2006
    Z.Li, June 2010
    """
    n_nodes = len(nodes)
    n_dof_m = 4 * n_nodes

    # CLEAN UP INPUT
    # clean u_p 0's, multiple terms. or out-of-order terms in m_all
    m_all = analysis.m_sort(m_all)

    # FIND BASE PROPERTIES
    el_props = analysis.elem_prop(nodes=nodes, elements=elements)
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
    ] = base_properties(nodes=nodes_base, elements=elements)
    [r_x, r_z, r_yd, r_ys, r_ud] = mode_constr(
        nodes=nodes_base, elements=elements, node_props=node_props, main_nodes=main_nodes, meta_elements=meta_elements
    )
    [d_y, n_global_modes] = y_dofs(
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

    # loop for the lengths
    n_lengths = len(lengths)
    l_i = 0  # length_index = one
    clas = []
    while l_i < n_lengths:
        length = lengths[l_i]
        # longitudinal terms included in the analysis for this length
        m_a = m_all[l_i]
        b_v_l = base_column(
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
        # orthonormal vectors
        b_v = base_update(
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

        # classification
        clas_modes = np.zeros((len(shapes[l_i, 0]), 4))
        for mod in range(0, len(shapes[l_i][0])):
            clas_modes[mod, 0:4] = mode_class(
                b_v=b_v,
                displacements=shapes[l_i][:, mod],
                n_global_modes=n_global_modes,
                n_dist_modes=n_dist_modes,
                n_local_modes=n_local_modes,
                m_a=m_a,
                n_dof_m=n_dof_m,
                GBT_con=GBT_con,
            )
        clas.append(clas_modes)
        l_i = l_i + 1  # length index = length index + one

    return clas


def mode_class(
    b_v: np.ndarray,
    displacements: np.ndarray,
    n_global_modes: int,
    n_dist_modes: int,
    n_local_modes: int,
    m_a: np.ndarray,
    n_dof_m: int,
    GBT_con: GBT_Con,
) -> np.ndarray:
    """to determine mode contribution in the current displacement

    Args:
        b_v (np.ndarray): base vectors (each column corresponds to a certain mode)
            columns 1..n_global_modes: global modes
            columns (n_global_modes+1)..(n_global_modes+n_dist_modes): dist. modes
            columns (n_global_modes+n_dist_modes+1)
                    ..(n_global_modes+n_dist_modes+n_local_modes): local modes
            columns (n_global_modes+n_dist_modes+n_local_modes+1)..n_dof: other modes
        displacements (np.ndarray): vector of nodal displacements
        n_global_modes (int): number of modes
        n_dist_modes (int): number of modes
        n_local_modes (int): number of modes
        m_a (np.ndarray): _description_
        n_dof_m (int): _description_
        GBT_con (GBT_Con): _description_
            GBT_con['couple'] - by GBT_con, coupled basis vs uncoupled basis for general B.C.
                especially for non-simply supported B.C.
                1: uncoupled basis, the basis will be block diagonal
                2: coupled basis, the basis is fully spanned

    Returns:
        clas_gdlo (np.ndarray): array with the contributions of the modes in percentage
            elem1: global, elem2: dist, elem3: local, elem4: other

    S. Adany, Mar 10, 2004
    Z. Li, June 2010
    """
    total_m = len(m_a)  # Total number of longitudinal terms m_i
    # indices
    dof_index = np.zeros((4, 2))
    dof_index[0, 0] = 0
    dof_index[0, 1] = n_global_modes
    dof_index[1, 0] = n_global_modes
    dof_index[1, 1] = n_global_modes + n_dist_modes
    dof_index[2, 0] = n_global_modes + n_dist_modes
    dof_index[2, 1] = n_global_modes + n_dist_modes + n_local_modes
    dof_index[3, 0] = n_global_modes + n_dist_modes + n_local_modes
    dof_index[3, 1] = n_dof_m

    if GBT_con["couple"] == 1:
        # uncoupled basis
        for i in range(0, len(m_a)):
            b_v_m = b_v[n_dof_m * i : n_dof_m * (i + 1), n_dof_m * i : n_dof_m * (i + 1)]

            # classification
            clas = np.linalg.lstsq(
                b_v_m[:, dof_index[0, 0] : dof_index[3, 1]], displacements[n_dof_m * i : n_dof_m * (i + 1)]
            )

            cl_gdlo = np.zeros((4, 5 * n_dof_m))
            for j in range(0, 4):
                n_modes = dof_index[j, 1] - dof_index[i, 0]
                cl_gdlo[i, j * n_modes : j * n_modes + n_modes] = clas[dof_index[j, 0] : dof_index[j, 1]]

        #     # L1 norm
        #     for m_n = 1:4
        #         clas_gdlo1(m_n) = sum(abs(cl_gdlo(m_n,:)))
        #
        #     norm_sum = sum(clas_gdlo1)
        #     clas_gdlo1 = clas_gdlo1/norm_sum*100

        # L2 norm
        clas_gdlo = np.zeros((1, 5))
        for m_n in range(0, 4):
            clas_gdlo[m_n] = np.linalg.norm(cl_gdlo[m_n, :])

        norm_sum = sum(clas_gdlo)
        clas_gdlo = clas_gdlo / norm_sum * 100
    else:
        # coupled basis
        # classification
        clas = np.linalg.lstsq(b_v, displacements)
        v_gdlo = np.zeros((4, (total_m + 1) * n_dof_m))
        clas_gdlo = np.zeros((1, 5))
        for i in range(0, 4):
            for j in range(0, total_m):
                n_modes = dof_index[i, 2] - dof_index[i, 1] + 1
                v_gdlo[i, j * n_modes : j * n_modes + n_modes] = clas[
                    j * n_dof_m + dof_index[i, 1] : j * n_dof_m + dof_index[i, 2]
                ]

            #         # L1 norm
            #         clas_gdlo1(i) = sum(abs(v_gdlo(i,:)))
            # L2 norm
            clas_gdlo[i] = np.linalg.norm(v_gdlo[i, :])

        #     # L1 norm
        #     NormSum1 = sum(clas_gdlo1)
        #     clas_gdlo1 = clas_gdlo1/NormSum1*100
        # L2 norm
        norm_sum = sum(clas_gdlo)
        clas_gdlo = clas_gdlo / norm_sum * 100

    return clas_gdlo


def node_class(node_props: np.ndarray) -> Tuple[int, int, int]:
    """this routine determines how many nodes of the various types exist

    notes:
      node types in node_props: 1-corner, 2-edge, 3-sub
      sub-node numbers are the original one, of course

    Args:
        node_props (np.ndarray): array of [original node nr, new node nr, nr of adj elems,
            node type]

    Returns:
        n_main_nodes (int): number of main nodes
        n_corner_nodes (int): number of corner nodes
        n_sub_nodes (int): number of sub-nodes

    S. Adany, Feb 09, 2004
    """
    # to count corner-, edge- and sub-nodes
    n_corner_nodes = 0
    n_edge_nodes = 0
    n_sub_nodes = 0
    for n_prop in node_props:
        if n_prop[3] == 1:
            n_corner_nodes = n_corner_nodes + 1
        elif n_prop[3] == 2:
            n_edge_nodes = n_edge_nodes + 1
        elif n_prop[3] == 3:
            n_sub_nodes = n_sub_nodes + 1

    n_main_nodes = n_corner_nodes + n_edge_nodes  # nr of main nodes

    return n_main_nodes, n_corner_nodes, n_sub_nodes
