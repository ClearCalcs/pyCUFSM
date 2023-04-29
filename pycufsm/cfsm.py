from copy import deepcopy
from scipy import linalg as spla
import numpy as np
import pycufsm.analysis

# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE, CPEng
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


def base_column(
    nodes_base, elements, props, length, b_c, m_a, el_props, node_props, n_main_nodes,
    n_corner_nodes, n_sub_nodes, n_global_modes, n_dist_modes, n_local_modes, dof_perm, r_x, r_z,
    r_ys, d_y
):
    # this routine creates base vectors for a column with length length for all the
    # specified longitudinal terms in m_a

    # assumptions
    #   orthogonalization is not performed unless the user wants
    #   orthogonalization is done by solving the eigen-value problem within each sub-space
    #   normalization is not done

    # input data
    #   nodes, elements, props - basic data
    #   b_c: ['S-S'] a string specifying boundary conditions to be analyzed:
    #'S-S' simply-pimply supported boundary condition at loaded edges
    #'C-C' clamped-clamped boundary condition at loaded edges
    #'S-C' simply-clamped supported boundary condition at loaded edges
    #'C-F' clamped-free supported boundary condition at loaded edges
    #'C-bulk' clamped-guided supported boundary condition at loaded edges
    #   m_a - longitudinal terms (half-wave numbers)

    # output data
    #   b_v_l - base vectors (each column corresponds to a certain mode)
    #   assemble for each half-wave number m_i on its diagonal
    #   b_v_l = diag(b_v_m)
    #   for each half-wave number m_i, b_v_m
    #           columns 1..n_global_modes: global modes
    #           columns (n_global_modes+1)..(n_global_modes+n_dist_modes): dist. modes
    #           columns (n_global_modes+n_dist_modes+1)
    #                    ..(n_global_modes+n_dist_modes+n_local_modes): local modes
    #           columns (n_global_modes+n_dist_modes+n_local_modes+1)..n_dof: other modes
    #   n_global_modes, n_dist_modes, n_local_modes - number of bulk, D, L modes, respectively
    #

    # S. Adany, Aug 28, 2006
    # B. Schafer, Aug 29, 2006
    # Z. Li, Dec 22, 2009
    # Z. Li, June 2010

    # construct the base for all the longitudinal terms
    n_nodes = len(nodes_base)
    n_dof_m = 4 * n_nodes
    total_m = len(m_a)
    b_v_l = np.zeros((n_dof_m * total_m, n_dof_m * total_m))
    for i, m_i in enumerate(m_a):
        # to create r_p constraint matrix for the rest of planar DOFs
        r_p = constr_planar_xz(
            nodes_base, elements, props, node_props, dof_perm, m_i, length, b_c, el_props
        )
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
            dof_perm=dof_perm
        )
        b_v_l[(n_dof_m * i):n_dof_m * (i+1), (n_dof_m * i):n_dof_m * (i+1)] = b_v_m

    return b_v_l


def base_update(
    gbt_con, b_v_l, length, m_a, nodes, elements, props, n_global_modes, n_dist_modes,
    n_local_modes, b_c, el_props
):
    # this routine optionally makes orthogonalization and normalization of base vectors

    # assumptions
    #   orthogonalization is done by solving the EV problem for each sub-space
    #   three options for normalization is possible, set by 'gbt_con['norm']' parameter

    # input data
    #   gbt_con['o_space'] - by gbt_con, choices of ST/O mode
    #         1: ST basis
    #         2: O space (null space of GDL) with respect to k_global
    #         3: O space (null space of GDL) with respect to kg_global
    #         4: O space (null space of GDL) in vector sense
    #   gbt_con['norm'] - by gbt_con, code for normalization (if normalization is done at all)
    #         0: no normalization,
    #         1: vector norm
    #         2: strain energy norm
    #         3: work norm
    #   b_v_l - natural base vectors for length (each column corresponds to a certain mode)
    #   for each half-wave number m_i
    #           columns 1..n_global_modes: global modes
    #           columns (n_global_modes+1)..(n_global_modes+n_dist_modes): dist. modes
    #           columns (n_global_modes+n_dist_modes+1)
    #                    ..(n_global_modes+n_dist_modes+n_local_modes): local modes
    #           columns (n_global_modes+n_dist_modes+n_local_modes+1)..n_dof_m: other modes
    #   length - length
    #   m_a, nodes, elements, props - as usual
    #   n_global_modes, n_dist_modes, n_local_modes - nr of modes
    #   b_c - boundary condition, 'S-S','C-C',...etc., as usual
    #   gbt_con['couple'] - by gbt_con, coupled basis vs uncoupled basis for general B.C.
    #             especially for non-simply supported B.C.
    #         1: uncoupled basis, the basis will be block diagonal
    #         2: coupled basis, the basis is fully spanned
    #   gbt_con['orth'] - by gbt_con, natural basis vs modal basis
    #         1: natural basis
    #         2: modal basis, axial orthogonality
    #         3: modal basis, load dependent orthogonality

    # output data
    #   b_v - output base vectors (maybe natural, orthogonal or normalized,
    #         depending on the selected options)

    # S. Adany, Oct 11, 2006
    # Z. Li modified on Jul 10, 2009
    # Z. Li, June 2010

    n_nodes = len(nodes[:, 1])
    n_dof_m = 4 * n_nodes
    total_m = len(m_a)  # Total number of longitudinal terms m_i
    b_v = np.zeros((n_dof_m * total_m, n_dof_m * total_m))

    if gbt_con['couple'] == 1:
        # uncoupled basis
        for i, m_i in enumerate(m_a):
            b_v_m = b_v_l[n_dof_m * i:n_dof_m * (i+1), n_dof_m * i:n_dof_m * (i+1)]
            # k_global/kg_global
            if gbt_con['norm'] == 2 or gbt_con['norm'] == 3 \
                or gbt_con['o_space'] == 2 or gbt_con['o_space'] == 3 \
                    or gbt_con['orth'] == 2 or gbt_con['orth'] == 3:
                # axial loading or real loading by either gbt_con['orth'] = 2 or gbt_con['orth'] = 3
                if gbt_con['orth'] == 1 or gbt_con['orth'] == 2:
                    nodes_base = deepcopy(nodes)
                    nodes_base[:, 7] = np.ones_like(nodes[:, 7])  # set u_p stress to 1.0 (axial)
                    [k_global, kg_global] = create_k_globals(
                        m_i=m_i,
                        nodes=nodes_base,
                        elements=elements,
                        el_props=el_props,
                        props=props,
                        length=length,
                        b_c=b_c
                    )
                else:
                    [k_global, kg_global] = create_k_globals(
                        m_i=m_i,
                        nodes=nodes,
                        elements=elements,
                        el_props=el_props,
                        props=props,
                        length=length,
                        b_c=b_c
                    )

            # orthogonalization/normalization begins
            #
            if gbt_con['orth'] == 2 or gbt_con['orth'] == 3 \
                or gbt_con['o_space'] == 2 or gbt_con['o_space'] == 3 or gbt_con['o_space'] == 4:
                # indices
                if gbt_con['o_space'] == 1:
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

                # define vectors for other modes, gbt_con['o_space'] = 2, 3, 4
                if gbt_con['o_space'] == 2:
                    a_matrix = spla.null_space(b_v_m[:, dof_index[0, 0]:dof_index[2, 1]].conj().T)
                    b_v_m[:, dof_index[3, 0]:dof_index[3, 1]] = np.linalg.solve(k_global, a_matrix)
                if gbt_con['o_space'] == 3:
                    a_matrix = spla.null_space(b_v_m[:, dof_index[0, 0]:dof_index[2, 1]].conj().T)
                    b_v_m[:, dof_index[3, 0]:dof_index[3, 1]] = np.linalg.solve(kg_global, a_matrix)
                if gbt_con['o_space'] == 4:
                    a_matrix = spla.null_space(b_v_m[:, dof_index[0, 0]:dof_index[2, 1]].conj().T)
                    b_v_m[:, dof_index[3, 0]:dof_index[3, 1]] = a_matrix

                # orthogonalization for modal basis 2/3 + normalization for normals 2/3
                for dof_sub in dof_index:
                    dof_sub1 = int(dof_sub[1])
                    dof_sub0 = int(dof_sub[0])
                    if dof_sub[1] >= dof_sub[0]:
                        k_global_sub \
                            = b_v_m[:, dof_sub0:dof_sub1].conj().T @ \
                            k_global @ b_v_m[:, dof_sub0:dof_sub1]
                        kg_global_sub \
                            = b_v_m[:, dof_sub0:dof_sub1].conj().T @ \
                            kg_global @ b_v_m[:, dof_sub0:dof_sub1]
                        [eigenvalues, eigenvectors] = spla.eig(a=k_global_sub, b=kg_global_sub)
                        lf_sub = np.real(eigenvalues)
                        indexsub = np.argsort(lf_sub)
                        lf_sub = lf_sub[indexsub]
                        eigenvectors = np.real(eigenvectors[:, indexsub])
                        if gbt_con['norm'] == 2 or gbt_con['norm'] == 3:
                            if gbt_con['norm'] == 2:
                                s_matrix = eigenvectors.conj().T @ k_global_sub @ eigenvectors

                            if gbt_con['norm'] == 3:
                                s_matrix = eigenvectors.conj().T @ kg_global_sub @ eigenvectors

                            s_matrix = np.diag(s_matrix)
                            for j in range(0, int(dof_sub[1] - dof_sub[0])):
                                eigenvectors[:, j] = np.transpose(
                                    np.conj(
                                        np.linalg.lstsq(
                                            eigenvectors[:, j].conj().T,
                                            np.sqrt(s_matrix).conj().T
                                        )
                                    )
                                )

                        b_v_m[:, dof_sub0:dof_sub1] \
                            = b_v_m[:, dof_sub0:dof_sub1] @ eigenvectors

            # normalization for gbt_con['o_space'] = 1
            if (gbt_con['norm'] == 2 or gbt_con['norm'] == 3) and gbt_con['o_space'] == 1:
                for j in range(0, n_dof_m):
                    if gbt_con['norm'] == 2:
                        b_v_m[:, j] = np.transpose(
                            np.conj(
                                np.linalg.lstsq(
                                    b_v_m[:, j].conj().T,
                                    np.sqrt(b_v_m[:, j].conj().T @ k_global @ b_v_m[:, j]).conj().T
                                )
                            )
                        )

                    if gbt_con['norm'] == 3:
                        b_v_m[:, j] = np.transpose(
                            np.conj(
                                np.linalg.lstsq(
                                    b_v_m[:, j].conj().T,
                                    np.sqrt(b_v_m[:, j].conj().T @ kg_global @ b_v_m[:, j]).conj().T
                                )
                            )
                        )

            # normalization for gbt_con['norm'] 1
            if gbt_con['norm'] == 1:
                for j in range(0, n_dof_m):
                    b_v_m[:, j] = b_v_m[:, j] / np.sqrt(b_v_m[:, j].conj().T @ b_v_m[:, j])

            b_v[n_dof_m * i:n_dof_m * (i+1), n_dof_m * i:n_dof_m * (i+1)] = b_v_m

    else:
        # coupled basis
        # k_global/kg_global
        if gbt_con['norm'] == 2 or gbt_con['norm'] == 3 \
            or gbt_con['o_space'] == 2 or gbt_con['o_space'] == 3 \
                or gbt_con['orth'] == 2 or gbt_con['orth'] == 3:
            # axial loading or real loading by either gbt_con['orth'] = 2 or gbt_con['orth'] = 3
            if gbt_con['orth'] == 1 or gbt_con['orth'] == 2:
                nodes_base = deepcopy(nodes)
                nodes_base[:, 7] = np.ones_like(nodes[:, 7])  # set u_p stress to 1.0 (axial)
            else:
                nodes_base = nodes

            # ZERO OUT THE GLOBAL MATRICES
            k_global = np.zeros((4 * n_nodes * total_m, 4 * n_nodes * total_m))
            kg_global = np.zeros((4 * n_nodes * total_m, 4 * n_nodes * total_m))

            # ASSEMBLE THE GLOBAL STIFFNESS MATRICES
            for i, elem in enumerate(elements):
                # Generate element stiffness matrix (k_local) in local coordinates
                thick = elem[3]
                b_strip = el_props[i, 1]
                mat_num = int(elem[4])
                row = int((np.argwhere(props[:, 0] == mat_num)).reshape(1))
                mat = props[row]
                stiff_x = mat[1]
                stiff_y = mat[2]
                nu_x = mat[3]
                nu_y = mat[4]
                bulk = mat[5]
                k_l = pycufsm.analysis.klocal(
                    stiff_x=stiff_x,
                    stiff_y=stiff_y,
                    nu_x=nu_x,
                    nu_y=nu_y,
                    bulk=bulk,
                    thick=thick,
                    length=length,
                    b_strip=b_strip,
                    b_c=b_c,
                    m_a=m_a
                )
                # Generate geometric stiffness matrix (kg_local) in local coordinates
                node_i = int(elem[1])
                node_j = int(elem[2])
                ty_1 = nodes_base[node_i][7] * thick
                ty_2 = nodes_base[node_j][7] * thick
                kg_l = pycufsm.analysis.kglocal(
                    length=length, b_strip=b_strip, ty_1=ty_1, ty_2=ty_2, b_c=b_c, m_a=m_a
                )
                # Transform k_local and kg_local into global coordinates
                alpha = el_props[i, 2]
                [k_local, kg_local] = pycufsm.analysis.trans(
                    alpha=alpha, k_local=k_l, kg_local=kg_l, m_a=m_a
                )

                # Add element contribution of k_local to full matrix k_global
                # and kg_local to kg_global
                [k_global, kg_global] = pycufsm.analysis.assemble(
                    k_global=k_global,
                    kg_global=kg_global,
                    k_local=k_local,
                    kg_local=kg_local,
                    node_i=node_i,
                    node_j=node_j,
                    n_nodes=n_nodes,
                    m_a=m_a
                )

        # orthogonalization/normalization begins
        if gbt_con['orth'] == 2 or gbt_con['orth'] == 3 \
            or gbt_con['o_space'] == 2 or gbt_con['o_space'] == 3 or gbt_con['o_space'] == 4:
            # indices
            dof_index[0, 0] = 0
            dof_index[0, 1] = n_global_modes
            dof_index[1, 0] = n_global_modes
            dof_index[1, 1] = n_global_modes + n_dist_modes
            dof_index[2, 0] = n_global_modes + n_dist_modes
            dof_index[2, 1] = n_global_modes + n_dist_modes + n_local_modes
            dof_index[3, 0] = n_global_modes + n_dist_modes + n_local_modes
            dof_index[3, 1] = n_dof_m

            n_other_modes = n_dof_m - (n_global_modes+n_dist_modes+n_local_modes)

            b_v_gdl = np.zeros(((len(m_a) + 1) * (n_global_modes+n_dist_modes+n_local_modes), 1))
            b_v_g = np.zeros(((len(m_a) + 1) * n_global_modes, 1))
            b_v_d = np.zeros(((len(m_a) + 1) * n_dist_modes, 1))
            b_v_l = np.zeros(((len(m_a) + 1) * n_local_modes, 1))
            b_v_o = np.zeros(((len(m_a) + 1) * n_other_modes, 1))
            for i, m_i in enumerate(m_a):
                # considering length-dependency on base vectors
                b_v_m = b_v_l[:, n_dof_m * i:n_dof_m * (i+1)]  # n_dof_m*i:n_dof_m*(i+1)
                b_v_gdl[:, i * (n_global_modes + n_dist_modes + n_local_modes):(i + 1) *
                        (n_global_modes+n_dist_modes+n_local_modes)] \
                    = b_v_m[:, dof_index[1, 1]:dof_index[3, 2]]
                b_v_g[:,
                      i * n_global_modes:(i+1) * n_global_modes] = b_v_m[:,
                                                                         dof_index[1,
                                                                                   1]:dof_index[1,
                                                                                                2]]
                b_v_d[:, i * n_dist_modes:(i+1) * n_dist_modes] = b_v_m[:,
                                                                        dof_index[2,
                                                                                  1]:dof_index[2,
                                                                                               2]]
                b_v_l[:, i * n_local_modes:(i+1) * n_local_modes] = b_v_m[:,
                                                                          dof_index[3,
                                                                                    1]:dof_index[3,
                                                                                                 2]]
                b_v_o[:, i * n_other_modes:(i+1) * n_other_modes] = b_v_m[:,
                                                                          dof_index[4,
                                                                                    1]:dof_index[4,
                                                                                                 2]]
                #

            # define vectors for other modes, gbt_con['o_space'] = 3 only
            if gbt_con['o_space'] == 3:
                a_matrix = spla.null_space(b_v_gdl.conj().T)
                b_v_o = np.linalg.solve(k_global, a_matrix)
                for i, m_i in enumerate(m_a):
                    b_v[:, i*n_dof_m+dof_index[3, 0]:i*n_dof_m+dof_index[3, 1]] \
                        = b_v_o[:, i*n_other_modes+1:(i+1)*n_other_modes]

            # define vectors for other modes, gbt_con['o_space'] = 4 only
            if gbt_con['o_space'] == 4:
                a_matrix = spla.null_space(b_v_gdl.conj().T)
                b_v_o = np.linalg.solve(kg_global, a_matrix)
                for i, m_i in enumerate(m_a):
                    b_v[:, i*n_dof_m+dof_index[3, 0]:i*n_dof_m+dof_index[3, 1]] \
                        = b_v_o[:, i*n_other_modes+1:(i+1)*n_other_modes]

            # define vectors for other modes, gbt_con['o_space'] = 5 only
            if gbt_con['o_space'] == 5:
                a_matrix = spla.null_space(b_v_gdl.conj().T)
                for i, m_i in enumerate(m_a):
                    b_v[:, i*n_dof_m+dof_index[3, 0]:i*n_dof_m+dof_index[3, 1]] \
                        = a_matrix[:, i*n_other_modes+1:(i+1)*n_other_modes]

            # orthogonalization + normalization for normals 2/3
            for i_sub, dof_sub in enumerate(dof_index):
                if dof_sub[2] >= dof_sub[1]:
                    if i_sub == 1:
                        k_global_sub = b_v_g.conj().T * k_global * b_v_g
                        kg_global_sub = b_v_g.conj().T * kg_global * b_v_g
                    elif i_sub == 2:
                        k_global_sub = b_v_d.conj().T * k_global * b_v_d
                        kg_global_sub = b_v_d.conj().T * kg_global * b_v_d
                    elif i_sub == 3:
                        k_global_sub = b_v_l.conj().T * k_global * b_v_l
                        kg_global_sub = b_v_l.conj().T * kg_global * b_v_l
                    elif i_sub == 4:
                        k_global_sub = b_v_o.conj().T * k_global * b_v_o
                        kg_global_sub = b_v_o.conj().T * kg_global * b_v_o

                    [eigenvalues, eigenvectors] = spla.eig(a=k_global_sub, b=kg_global_sub)
                    lf_sub = np.real(eigenvalues)
                    indexsub = np.argsort(lf_sub)
                    lf_sub = lf_sub[indexsub]
                    eigenvectors = np.real(eigenvectors[:, indexsub])
                    if gbt_con['norm'] == 2 or gbt_con['norm'] == 3:
                        if gbt_con['norm'] == 2:
                            s_matrix = eigenvectors.conj().T @ k_global_sub @ eigenvectors
                        if gbt_con['norm'] == 3:
                            s_matrix = eigenvectors.conj().T @ kg_global_sub @ eigenvectors
                        s_matrix = np.diag(s_matrix)
                        for i in range(0, (dof_sub[1] - dof_sub[0]) * total_m):
                            eigenvectors[:, i] = np.transpose(
                                np.conj(
                                    np.linalg.lstsq(
                                        eigenvectors[:, i].conj().T,
                                        np.sqrt(s_matrix).conj().T
                                    )
                                )
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
                            b_v[:, i*n_dof_m+dof_sub[1]:i*n_dof_m+dof_sub[2]] \
                                = b_v_orth[:, i*n_global_modes+1:(i+1)*n_global_modes]
                        elif i_sub == 2:
                            b_v[:, i*n_dof_m+dof_sub[1]:i*n_dof_m+dof_sub[2]] \
                                = b_v_orth[:, i*n_dist_modes+1:(i+1)*n_dist_modes]
                        elif i_sub == 3:
                            b_v[:, i*n_dof_m+dof_sub[1]:i*n_dof_m+dof_sub[2]] \
                                = b_v_orth[:, i*n_local_modes+1:(i+1)*n_local_modes]
                        elif i_sub == 4:
                            b_v[:, i*n_dof_m+dof_sub[1]:i*n_dof_m+dof_sub[2]] \
                                = b_v_orth[:, i*n_other_modes+1:(i+1)*n_other_modes]

        # normalization for gbt_con['o_space'] = 1
        if (gbt_con['norm'] == 2 or gbt_con['norm'] == 3) and (gbt_con['o_space'] == 1):
            for i in range(0, n_dof_m * total_m):
                if gbt_con['norm'] == 2:
                    b_v[:, i] = np.transpose(
                        np.conj(
                            np.linalg.lstsq(
                                b_v[:, i].conj().T,
                                np.sqrt(b_v[:, i].conj().T @ k_global @ b_v[:, i]).conj().T
                            )
                        )
                    )

                if gbt_con['norm'] == 3:
                    b_v[:, i] = np.transpose(
                        np.conj(
                            np.linalg.lstsq(
                                b_v[:, i].conj().T,
                                np.sqrt(b_v[:, i].conj().T @ kg_global @ b_v[:, i]).conj().T
                            )
                        )
                    )

        # normalization for gbt_con['norm'] 1
        if gbt_con['norm'] == 1:
            for i in range(0, n_dof_m * total_m):
                b_v[:, i] = np.transpose(
                    np.conj(
                        np.linalg.lstsq(
                            b_v[:, i].conj().T,
                            np.sqrt(b_v[:, i].conj().T @ b_v[:, i]).conj().T
                        )
                    )
                )
        #     b_v[n_dof_m*i:n_dof_m*(i+1),n_dof_m*i:n_dof_m*(i+1)] = b_v_m
    return b_v


def mode_select(b_v, n_global_modes, n_dist_modes, n_local_modes, gbt_con, n_dof_m, m_a):
    # this routine selects the required base vectors
    #   b_v_red forms a reduced space for the calculation, including the
    #       selected modes only
    #   b_v_red itself is the final constraint matrix for the selected modes
    #
    #
    # input data
    #   b_v - base vectors (each column corresponds to a certain mode)
    #           columns 1..n_global_modes: global modes
    #           columns (n_global_modes+1)..(n_global_modes+n_dist_modes): dist. modes
    #           columns (n_global_modes+n_dist_modes+1)
    #                    ..(n_global_modes+n_dist_modes+n_local_modes): local modes
    #           columns (n_global_modes+n_dist_modes+n_local_modes+1)..n_dof: other modes
    #   n_global_modes, n_dist_modes, n_local_modes - number of global, distortional
    #                    and local buckling modes, respectively
    #   gbt_con['glob'] - indicator which global modes are selected
    #   gbt_con['dist'] - indicator which dist. modes are selected
    #   gbt_con['local'] - indicator whether local modes are selected
    #   gbt_con['other'] - indicator whether other modes are selected
    #   n_dof_m: 4*n_nodes, total DOF for a singal longitudinal term

    # output data
    #   b_v_red - reduced base vectors (each column corresponds to a certain mode)

    #
    # note:
    #   for all if_* indicator: 1 if selected, 0 if eliminated
    #
    #
    # S. Adany, Mar 22, 2004
    # BWS May 2004
    # modifed on Jul 10, 2009 by Z. Li for general b_c
    # Z. Li, June 2010

    n_m = int(
        sum(gbt_con['glob']) + sum(gbt_con['dist']) + sum(gbt_con['local']) + sum(gbt_con['other'])
    )
    b_v_red = np.zeros((len(b_v), (len(m_a) + 1) * n_m))
    for i in range(0, len(m_a)):
        #     b_v_m = b_v[n_dof_m*i:n_dof_m*(i+1),n_dof_m*i:n_dof_m*(i+1)]
        n_other_modes = n_dof_m - n_global_modes - n_dist_modes - n_local_modes  # nr of other modes
        #
        nmo = 0
        b_v_red_m = np.zeros((len(b_v), n_m))
        for j in range(0, n_global_modes):
            if j < len(gbt_con['glob']) and gbt_con['glob'][j] == 1:
                b_v_red_m[:, nmo] = b_v[:, n_dof_m*i + j]
                nmo = nmo + 1

        for j in range(0, n_dist_modes):
            if j < len(gbt_con['dist']) and gbt_con['dist'][j] == 1:
                b_v_red_m[:, nmo] = b_v[:, n_dof_m*i + n_global_modes + j]
                nmo = nmo + 1

        # if gbt_con['local'] == 1
        #     b_v_red[:,(nmo+1):(nmo+n_local_modes)]
        #         = b_v[:,(n_global_modes+n_dist_modes+1):(n_global_modes+
        #               n_dist_modes+n_local_modes)]
        #     nmo = nmo+n_local_modes
        # end
        for j in range(0, n_local_modes):
            if j < len(gbt_con['local']) and gbt_con['local'][j] == 1:
                b_v_red_m[:, nmo] = b_v[:, n_dof_m*i + n_global_modes + n_dist_modes + j]
                nmo = nmo + 1

        for j in range(0, n_other_modes):
            if j < len(gbt_con['other']) and gbt_con['other'][j] == 1:
                b_v_red_m[:,
                          nmo] = b_v[:,
                                     n_dof_m*i + n_global_modes + n_dist_modes + n_local_modes + j]
                nmo = nmo + 1

        # if gbt_con['other'] == 1
        #     n_other_modes = len(b_v[:, 1])-n_global_modes - n_dist_modes - n_local_modes
        #            # nr of other modes
        #     b_v_red[:,(nmo+1):(nmo+n_other_modes)]
        #          = b_v[:,(n_global_modes+n_dist_modes+n_local_modes+1):(n_global_modes+
        #                n_dist_modes+n_local_modes+n_other_modes)]
        #     # b_v_red[:,(nmo+1)] = b_v[:,(n_global_modes+n_dist_modes+n_local_modes+1)]
        # end
        b_v_red[:, nmo * i:nmo * (i+1)] = b_v_red_m

    return b_v_red


def constr_user(nodes, constraints, m_a):
    #
    # this routine creates the constraints matrix, r_user_matrix, as defined by the user
    #
    #
    # input/output data
    #   nodes - same as elsewhere throughout this program
    #   constraints - same as 'constraints' throughout this program
    #   m_a - longitudinal terms to be included for this length

    #   r_user_matrix - the constraints matrix (in other words: base vectors) so that
    #               displ_orig = r_user_matrix * displ_new

    # S. Adany, Feb 26, 2004
    # Z. Li, Aug 18, 2009 for general b.c.
    # Z. Li, June 2010

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
                        dof_e = j*2 + 1 - 1
                    elif k == 5:
                        dof_e = (j+1) * 2 - 1
                    elif k == 4:
                        dof_e = n_nodes*2 + j*2 + 1 - 1
                    elif k == 6:
                        dof_e = n_nodes*2 + (j+1) * 2 - 1

                    dof_reg[dof_e, 0] = 0

        # to consider master-slave constraints
        for j in range(0, len(constraints)):
            if len(constraints[j, :]) >= 5:
                # nr of eliminated DOF
                node_e = constraints[j, 0]
                if constraints[j, 1] == 0:
                    dof_e = node_e*2 + 1 - 1
                elif constraints[j, 1] == 2:
                    dof_e = (node_e+1) * 2 - 1
                elif constraints[j, 1] == 1:
                    dof_e = n_nodes*2 + node_e*2 + 1 - 1
                elif constraints[j, 1] == 3:
                    dof_e = n_nodes*2 + (node_e+1) * 2 - 1

                # nr of kept DOF
                node_k = constraints[j, 3]
                if constraints[j, 4] == 0:
                    dof_k = node_k*2 + 1 - 1
                elif constraints[j, 4] == 2:
                    dof_k = (node_k+1) * 2 - 1
                elif constraints[j, 4] == 1:
                    dof_k = n_nodes*2 + node_k*2 + 1 - 1
                elif constraints[j, 4] == 3:
                    dof_k = n_nodes*2 + (node_k+1) * 2 - 1

                # to modify r_user_matrix
                r_user_m_matrix[:, dof_k] = r_user_m_matrix[:, dof_k] \
                    + constraints[j, 2]*r_user_m_matrix[:, dof_e]
                dof_reg[dof_e, 0] = 0

        # to eliminate columns from r_user_matrix
        k = -1
        r_u_matrix = np.zeros_like(r_user_m_matrix)
        for j in range(0, n_dof_m):
            if dof_reg[j, 0] == 1:
                k = k + 1
                r_u_matrix[:, k] = r_user_m_matrix[:, j]

        r_user_m_matrix = r_u_matrix[:, 0:k]
        r_user_matrix[i * n_dof_m:(i+1) * n_dof_m, i * k:(i+1) * k] = r_user_m_matrix

    return r_user_matrix


def mode_constr(nodes, elements, node_props, main_nodes, meta_elements):
    #
    # this routine creates the constraint matrices necessary for mode
    # separation/classification for each specified half-wave number m_i
    #
    # assumptions
    #   GBT-like assumptions are used
    #   the cross-section must not be closed and must not contain closed parts
    #
    #   must check whether 'Warp' works well for any open section !!!
    #
    #
    # input/output data
    #   nodes, elements, props  - same as elsewhere throughout this program
    #   main_nodes [main nodes] - array of
    #         [nr, x, z, orig nodes nr, nr of adj meta-elements, m_i-el_i-1, m_i-el_i-2, ...]
    #   meta_elements [meta-elements] - array of
    #         [nr, main-nodes-1, main-nodes-2, nr of sub-nodes, sub-no-1, sub-nod-2, ...]
    #   node_props - array of [original nodes nr, new nodes nr, nr of adj elements, nodes type]
    #
    #
    # notes:
    #   m-el_i-? is positive if the starting nodes of m-el_i-? coincides with
    #      the given m-nodes, otherwise negative
    #   nodes types: 1-corner, 2-edge, 3-sub
    #   sub-nodes numbers are the original one, of course
    #
    # S. Adany, Mar 10, 2004
    # Z. Li, Jul 10, 2009

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
    nodes, elements, main_nodes, n_main_nodes, n_dist_modes, r_yd, r_ud, sect_props, el_props
):

    # this routine creates y-DOFs of main nodes for global buckling and
    # distortional buckling, however:
    #    only involves single half-wave number m_i
    #
    # assumptions
    #   GBT-like assumptions are used
    #   the cross-section must not be closed and must not contain closed parts

    # input data
    #   nodes, elements - same as elsewhere throughout this program
    #   main_nodes [main nodes] - nodes of 'meta' cross-section
    #   n_main_nodes, n_corner_nodes, n_sub_nodes
    #          - number of main nodes, corner nodes and sub-nodes, respectively
    #   n_dist_modes, n_local_modes - number of distortional and local buckling modes, respectively
    #   r_yd, r_ud - constraint matrices
    #
    # output data
    #   d_y - y-DOFs of main nodes for global buckling and distortional buckling
    #   (each column corresponds to a certain mode)
    #
    #
    # S. Adany, Mar 10, 2004, modified Aug 29, 2006
    # Z. Li, Dec 22, 2009

    w_o = np.zeros((len(nodes), 2))
    w_o[int(elements[0, 1]), 0] = int(elements[0, 1])
    w_no = 0

    # compute the unit warping
    # code from cutwp_prop2:232-249
    for _ in range(0, len(elements)):
        i = 0
        while (np.any(w_o[:, 0] == elements[i, 1]) and np.any(w_o[:, 0] == elements[i, 2])) \
            or (not np.any(w_o[:, 0] == elements[i, 1]) \
                and not np.any(w_o[:, 0] == elements[i, 2])):
            i = i + 1
        s_n = int(elements[i, 1])
        f_n = int(elements[i, 2])
        p_o = ((nodes[s_n, 1] - sect_props['x0']) \
            * (nodes[f_n, 2] - sect_props['y0']) \
            - (nodes[f_n, 1] - sect_props['x0']) \
                * (nodes[s_n, 2] - sect_props['y0'])) \
                / el_props[i, 1]
        if w_o[s_n, 0] == 0:
            w_o[s_n, 0] = s_n
            w_o[s_n, 1] = w_o[f_n, 1] - p_o * el_props[i, 1]
        elif w_o[int(elements[i, 2]), 1] == 0:
            w_o[f_n, 0] = f_n
            w_o[f_n, 1] = w_o[s_n, 1] + p_o * el_props[i, 1]
        w_no = w_no + 1 / (2*sect_props['A']) * (w_o[s_n, 1] + w_o[f_n, 1]) \
            * elements[i, 3] * el_props[i, 1]
    w_n = w_no - w_o[:, 1]
    # coord. transform. to the principal axes
    phi = sect_props['phi']
    rot = np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)],
    ])
    centre_of_gravity = [
        sect_props['cx'],
        sect_props['cy'],
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
        if np.nonzero(d_y[:, i]) == []:
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
        j_junk3[:, nj1:nj1 + nj2] = j_junk2
        j_junk4 = spla.null_space(j_junk3.conj().T)

        # d_y(:,(n_global_modes+2):(n_global_modes+1+n_dist_modes)) = j_junk4
        junk3 = j_junk4.conj().T @ r_yd @ j_junk4
        # junk3 = junk2.conj().T*junk2
        #
        [_, eigenvectors] = spla.eig(junk3)
        # eigenvalues = diag(eigenvalues)
        # [eigenvalues, index] = sort(eigenvalues)
        # eigenvectors = eigenvectors[:, index]
        d_y[:, n_global_modes:n_global_modes + n_dist_modes] = j_junk4 @ eigenvectors

    return d_y, n_global_modes


def base_vectors(
    d_y, elements, el_props, length, m_i, node_props, n_main_nodes, n_corner_nodes, n_sub_nodes,
    n_global_modes, n_dist_modes, n_local_modes, r_x, r_z, r_p, r_ys, dof_perm
):
    #
    # this routine creates the base vectors for global, dist., local and other modes
    #
    # assumptions
    #   GBT-like assumptions are used
    #   the cross-section must not be closed and must not contain closed parts
    #
    #   must check whether 'Warp' works well for any open section !!!
    #
    #
    # input data
    #   elements, el_props - same as elsewhere throughout this program
    #   length, m_i - member length and number of half-waves, respectively
    #   main_nodes [main nodes] - nodes of 'meta' cross-section
    #   meta_elements [meta-elements] - elements of 'meta' cross-section
    #   node_props - some properties of the nodes
    #   n_main_nodes, n_corner_nodes, n_sub_nodes
    #           - number of main nodes, corner nodes and sub-nodes, respectively
    #   n_dist_modes, n_local_modes - number of distortional and local buckling modes, respectively
    #   r_x, r_z, r_p, r_ys, - constraint matrices
    #   dof_perm - permutation matrix to re-order the DOFs
    #
    # output data
    #   n_other_modes - nr of other modes
    #   b_v_m - base vectors for single half-wave number m_i
    #            (each column corresponds to a certain mode)
    #           columns 1..n_global_modes: global modes
    #           columns (n_global_modes+1)..(n_global_modes+n_dist_modes): dist. modes
    #           columns (n_global_modes+n_dist_modes+1)
    #                    ..(n_global_modes+n_dist_modes+n_local_modes): local modes
    #           columns (n_global_modes+n_dist_modes+n_local_modes+1)..n_dof: other modes
    #
    # note:
    #   more details on the input variables can be found in the routines called
    #      in this routine
    #

    # S. Adany, Mar 10, 2004, modified Aug 29, 2006
    # Z. Li, Dec 22, 2009

    # DATA PREPARATION
    k_m = m_i * np.pi / length
    n_node_props = len(node_props)
    n_dof = 4 * n_node_props  # nro of DOFs
    n_edge_nodes = n_main_nodes - n_corner_nodes
    # zero out
    b_v_m = np.zeros((n_dof, n_dof))

    # CALCULATION FOR GLOBAL AND DISTORTIONAL BUCKLING MODES
    # to add global and dist y DOFs to base vectors
    b_v_m = d_y[:, 0:n_global_modes + n_dist_modes]
    b_v_m = np.concatenate((b_v_m, np.zeros((n_dof - len(b_v_m), len(b_v_m[0])))), axis=0)
    #
    # to add x DOFs of corner nodes to the base vectors
    # r_x = r_x/k_m
    b_v_m[n_main_nodes:n_main_nodes + n_corner_nodes, 0:n_global_modes
          + n_dist_modes] = r_x @ b_v_m[0:n_main_nodes, 0:n_global_modes + n_dist_modes]
    #
    # to add z DOFs of corner nodes to the base vectors
    # r_z = r_z/k_m
    b_v_m[n_main_nodes + n_corner_nodes:n_main_nodes + 2*n_corner_nodes, 0:n_global_modes
          + n_dist_modes] = r_z @ b_v_m[0:n_main_nodes, 0:n_global_modes + n_dist_modes]
    #
    # to add other planar DOFs to the base vectors
    b_v_m[n_main_nodes + 2*n_corner_nodes:n_dof - n_sub_nodes, 0:n_global_modes
          + n_dist_modes] = r_p @ b_v_m[n_main_nodes:n_main_nodes + 2*n_corner_nodes,
                                        0:n_global_modes + n_dist_modes]
    #
    # to add y DOFs of sub-nodes to the base vector
    b_v_m[n_dof - n_sub_nodes:n_dof, 0:n_global_modes
          + n_dist_modes] = r_ys @ b_v_m[0:n_main_nodes, 0:n_global_modes + n_dist_modes]
    #
    # division by k_m
    b_v_m[n_main_nodes:n_dof - n_sub_nodes,
          0:n_global_modes + n_dist_modes] = b_v_m[n_main_nodes:n_dof - n_sub_nodes,
                                                   0:n_global_modes + n_dist_modes] / k_m
    #
    # norm base vectors
    for i in range(0, n_global_modes + n_dist_modes):
        b_v_m[:, i] = b_v_m[:, i] / np.linalg.norm(b_v_m[:, i])

    # CALCULATION FOR LOCAL BUCKLING MODES
    n_globdist_modes = n_global_modes + n_dist_modes  # nr of global and dist. modes
    b_v_m = np.concatenate((b_v_m, np.zeros((len(b_v_m), n_local_modes))), axis=1)
    # np.zeros
    b_v_m[0:n_dof,
          n_globdist_modes:n_globdist_modes + n_local_modes] = np.zeros((n_dof, n_local_modes))

    # rot DOFs for main nodes
    b_v_m[3 * n_main_nodes:4 * n_main_nodes,
          n_globdist_modes:n_globdist_modes + n_main_nodes] = np.eye(n_main_nodes)
    #
    # rot DOFs for sub nodes
    if n_sub_nodes > 0:
        b_v_m[4*n_main_nodes + 2*n_sub_nodes:4*n_main_nodes + 3*n_sub_nodes, n_globdist_modes
              + n_main_nodes:n_globdist_modes + n_main_nodes + n_sub_nodes] = np.eye(n_sub_nodes)

    # x, z DOFs for edge nodes
    k = 0
    for i in range(0, n_node_props):
        if node_props[i, 3] == 2:
            el_i = np.nonzero(
                np.any(elements[:, 1] == i) or np.any(elements[:, 2] == i)
            )  # adjacent element
            alfa = el_props[el_i, 2]
            b_v_m[n_main_nodes + 2*n_corner_nodes + k,
                  n_globdist_modes + n_main_nodes + n_sub_nodes + k] = -np.sin(alfa)  # x
            b_v_m[n_main_nodes + 2*n_corner_nodes + n_edge_nodes + k,
                  n_globdist_modes + n_main_nodes + n_sub_nodes + k] = np.cos(alfa)  # z
            k = k + 1

    # x, z DOFs for sub-nodes
    if n_sub_nodes > 0:
        k = 0
        for i in range(0, n_node_props):
            if node_props[i, 3] == 3:
                el_i = np.nonzero(
                    np.any(elements[:, 1] == i) or np.any(elements[:, 2] == i)
                )  # adjacent element
                alfa = el_props[el_i[0], 2]
                b_v_m[4*n_main_nodes + k, n_globdist_modes + n_main_nodes + n_sub_nodes
                      + n_edge_nodes + k] = -np.sin(alfa)  # x
                b_v_m[4*n_main_nodes + n_sub_nodes + k, n_globdist_modes + n_main_nodes
                      + n_sub_nodes + n_edge_nodes + k] = np.cos(alfa)  # z
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
        b_v_m[(n_nod1-1) * 2, n_globdist_modes + n_local_modes + i] = 0.5
        b_v_m[(n_nod2-1) * 2, n_globdist_modes + n_local_modes + i] = -0.5

        # create the base-vectors for membrane TRANSVERSE modes
        b_v_m[(n_nod1-1) * 2,
              n_globdist_modes + n_local_modes + n_elements + i] = -0.5 * np.cos(alfa)
        b_v_m[(n_nod2-1) * 2,
              n_globdist_modes + n_local_modes + n_elements + i] = 0.5 * np.cos(alfa)
        b_v_m[2*n_node_props + (n_nod1-1) * 2,
              n_globdist_modes + n_local_modes + n_elements + i] = 0.5 * np.sin(alfa)
        b_v_m[2*n_node_props + (n_nod2-1) * 2,
              n_globdist_modes + n_local_modes + n_elements + i] = -0.5 * np.sin(alfa)

    # RE_ORDERING DOFS
    b_v_m[:, 0:n_globdist_modes
          + n_local_modes] = dof_perm @ b_v_m[:, 0:n_globdist_modes + n_local_modes]

    return b_v_m


def constr_xz_y(main_nodes, meta_elements):
    # this routine creates the constraint matrix, Rxz, that defines relationship
    # between x, z displacements DOFs [for internal main nodes, referred also as corner nodes]
    # and the longitudinal y displacements DOFs [for all the main nodes]
    # if GBT-like assumptions are used
    #
    # to make this routine length-independent, Rxz is not multiplied here by
    # (1/k_m), thus it has to be multiplied outside of this routine!
    #
    # additional assumption: cross section is opened!
    #
    #
    # input/output data
    #   main_nodes [main nodes] - array of
    #            [nr, x, z, orig nodes nr, nr of adj meta-elements, m-el_i-1, m-el_i-2, ...]
    #   meta_elements [meta-elements] - array of
    #            [nr, main-nodes-1, main-nodes-2, nr of sub-nodes, sub-no-1, sub-nod-2, ...]
    #
    #   note:
    #   m-el_i-? is positive if the starting nodes of m-el_i-? coincides with
    #   the given m-nodes, otherwise negative
    #
    # S. Adany, Feb 05, 2004
    #
    #
    # to calculate some data of main elements (stored in meta_elements_data)
    meta_elements_data = np.zeros((len(meta_elements), 5))
    for i, m_elem in enumerate(meta_elements):
        node1 = int(m_elem[1])
        node2 = int(m_elem[2])
        x_1 = main_nodes[node1, 1]
        x_2 = main_nodes[node2, 1]
        z_1 = main_nodes[node1, 2]
        z_2 = main_nodes[node2, 2]
        b_i = np.sqrt((x_2 - x_1)**2 + (z_2 - z_1)**2)
        a_i = np.arctan2(z_2 - z_1, x_2 - x_1)
        s_i = (z_2-z_1) / b_i
        c_i = (x_2-x_1) / b_i
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
            while np.sin(meta_elements_data[int(np.real(m_node[j])), 2]
                         - meta_elements_data[elem1, 2]) == 0:
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
            r_x[k, main_nodes2] = (-sin1 * r_2 - sin2*r_1) / det
            r_x[k, main_nodes3] = sin1 * r_2 / det

            r_z[k, main_nodes1] = -cos2 * r_1 / det
            r_z[k, main_nodes2] = (cos1*r_2 + cos2*r_1) / det
            r_z[k, main_nodes3] = -cos1 * r_2 / det

            k = k + 1

    return r_x, r_z


def constr_planar_xz(nodes, elements, props, node_props, dof_perm, m_i, length, b_c, el_props):
    #
    # this routine creates the constraint matrix, r_p, that defines relationship
    # between x, z DOFs of any non-corner nodes + teta DOFs of all nodes,
    # and the x, z displacements DOFs of corner nodes
    # if GBT-like assumptions are used
    #
    #
    # input/output data
    #   nodes, elements, props  - same as elsewhere throughout this program
    #   node_props - array of [original nodes nr, new nodes nr, nr of adj elements, nodes type]
    #   dof_perm - permutation matrix, so that
    #            (orig-displacements-vect) = (dof_perm) � (new-displacements - vector)
    #
    # S. Adany, Feb 06, 2004
    # Z. Li, Jul 10, 2009
    #
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
    k_global = kglobal_transv(nodes, elements, props, m_i, length, b_c, el_props)

    # to re-order the DOFs
    k_global = dof_perm.conj().T @ k_global @ dof_perm

    # to have partitions of k_global
    k_global_pp = k_global[n_main_nodes + 2*n_corner_nodes:n_dof - n_sub_nodes,
                           n_main_nodes + 2*n_corner_nodes:n_dof - n_sub_nodes]
    k_global_pc = k_global[n_main_nodes + 2*n_corner_nodes:n_dof - n_sub_nodes,
                           n_main_nodes:n_main_nodes + 2*n_corner_nodes]

    # to form the constraint matrix
    #[r_p]=-inv(k_global_pp) * k_global_pc

    r_p = -np.linalg.solve(k_global_pp, k_global_pc)

    return r_p


def constr_yd_yg(nodes, elements, node_props, r_ys, n_main_nodes):
    #
    # this routine creates the constraint matrix, r_yd, that defines relationship
    # between base vectors for distortional buckling,
    # and base vectors for global buckling,
    # but for y DOFs of main nodes only
    #
    #
    # input/output data
    #   nodes, elements - same as elsewhere throughout this program
    #   node_props - array of [original nodes nr, new nodes nr, nr of adj elements, nodes type]
    #   r_ys - constrain matrix, see function 'constr_ys_ym'
    #   n_main_nodes - nr of main nodes
    #
    # S. Adany, Mar 04, 2004
    #
    n_nodes = len(nodes)
    a_matrix = np.zeros((n_nodes, n_nodes))
    for elem in elements:
        node1 = int(elem[1])
        node2 = int(elem[2])
        d_x = nodes[node2, 1] - nodes[node1, 1]
        d_z = nodes[node2, 2] - nodes[node1, 2]
        d_area = np.sqrt(d_x*d_x + d_z*d_z) * elem[3]
        ind = np.nonzero(node_props[:, 0] == node1)
        node1 = int(node_props[ind, 1])
        ind = np.nonzero(node_props[:, 0] == node2)
        node2 = int(node_props[ind, 1])
        a_matrix[node1, node1] = a_matrix[node1, node1] + 2*d_area
        a_matrix[node2, node2] = a_matrix[node2, node2] + 2*d_area
        a_matrix[node1, node2] = a_matrix[node1, node2] + d_area
        a_matrix[node2, node1] = a_matrix[node2, node1] + d_area

    r_ysm = np.zeros((n_nodes, n_main_nodes))
    r_ysm[0:n_main_nodes, 0:n_main_nodes] = np.eye(n_main_nodes)
    r_ysm[n_main_nodes:n_nodes, 0:n_main_nodes] = r_ys
    r_yd = r_ysm.conj().T @ a_matrix @ r_ysm

    return r_yd


def constr_ys_ym(nodes, main_nodes, meta_elements, node_props):
    # this routine creates the constraint matrix, r_ys, that defines relationship
    # between y DOFs of sub-nodes,
    # and the y displacements DOFs of main nodes
    # by linear interpolation
    #
    #
    # input/output data
    #   nodes - same as elsewhere throughout this program
    #   main_nodes [main nodes] - array of
    #            [nr, x, z, orig nodes nr, nr of adj meta-elements, m-el_i-1, m-el_i-2, ...]
    #   meta_elements [meta-elements] - array of
    #            [nr, main-nodes-1, main-nodes-2, nr of sub-nodes, sub-no-1, sub-nod-2, ...]
    #   node_props - array of [original nodes nr, new nodes nr, nr of adj elements, nodes type]
    #
    # S. Adany, Feb 06, 2004
    #
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
            b_m = np.sqrt((x_3 - x_1)**2 + (z_3 - z_1)**2)
            n_new1 = int(node_props[nod1, 1])
            n_new3 = int(node_props[nod3, 1])
            for j in range(0, int(m_elem[3])):
                nod2 = int(m_elem[j + 4])
                x_2 = nodes[nod2, 1]
                z_2 = nodes[nod2, 2]
                b_s = np.sqrt((x_2 - x_1)**2 + (z_2 - z_1)**2)
                n_new2 = int(node_props[nod2, 1])
                r_ys[n_new2 - n_main_nodes, n_new1] = (b_m-b_s) / b_m
                r_ys[n_new2 - n_main_nodes, n_new3] = b_s / b_m

    return r_ys


def constr_yu_yd(main_nodes, meta_elements):
    #
    # this routine creates the constraint matrix, r_ud, that defines relationship
    # between y displacements DOFs of indefinite main nodes
    # and the y displacements DOFs of definite main nodes
    # (definite main nodes = those main nodes which unambiguously define the y displacements pattern
    #  indefinite main nodes = those nodes the y DOF of which can be calculated
    #                          from the y DOF of definite main nodes
    #  note: for open sections with one single branch only there are no indefinite nodes)
    #
    # important assumption: cross section is opened!
    #
    #
    # input/output data
    #   main_nodes [main nodes] - array of
    #            [nr, x, z, orig nodes nr, nr of adj meta-elements, m-el_i-1, m-el_i-2, ...]
    #   meta_elements [meta-elements] - array of
    #            [nr, main-nodes-1, main-nodes-2, nr of sub-nodes, sub-no-1, sub-nod-2, ...]
    #
    #   note:
    #   m-el_i-? is positive if the starting nodes of m-el_i-? coincides with
    #   the given m-nodes, otherwise negative
    #
    # S. Adany, Mar 10, 2004
    #
    #
    # to calculate some data of main elements (stored in meta_elements_data)
    meta_elements_data = np.zeros((len(meta_elements), 5))
    for i, m_elem in enumerate(meta_elements):
        node1 = int(m_elem[1])
        node2 = int(m_elem[2])
        x_1 = main_nodes[node1, 1]
        x_2 = main_nodes[node2, 1]
        z_1 = main_nodes[node1, 2]
        z_2 = main_nodes[node2, 2]
        b_i = np.sqrt((x_2 - x_1)**2 + (z_2 - z_1)**2)
        a_i = np.arctan2(z_2 - z_1, x_2 - x_1)
        s_i = (z_2-z_1) / b_i
        c_i = (x_2-x_1) / b_i
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
            while np.sin(meta_elements_data[int(np.real(m_node[j])), 2]
                         - meta_elements_data[elem1, 2]) == 0:
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
            while np.sin(meta_elements_data[int(np.real(m_node[j])), 2]
                         - meta_elements_data[elem1, 2]) == 0:
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


def base_properties(nodes, elements):
    # this routine creates all the data for defining the base vectors from the
    # cross section properties
    #
    # input data
    #   nodes, elements- basic data#
    # output data
    #   main_nodes <main nodes> - array of
    #           [nr, x, z, orig nodes nr, nr of adj meta-elements, m-el_i-1, m-el_i-2, ...]
    #   meta_elements <meta-elements> - array of
    #           [nr, main-nodes-1, main-nodes-2, nr of sub-nodes, sub-no-1, sub-nod-2, ...]
    #   node_props - array of [original nodes nr, new nodes nr, nr of adj elements,
    #   nodes type]
    #   n_global_modes, n_dist_modes, n_local_modes, n_other_modes
    #            - number of bulk, D, L, O modes, respectively
    #   n_main_nodes, n_corner_nodes, n_sub_nodes
    #            - number of main nodes, corner nodes and sub-nodes, respectively
    #   dof_perm - permutation matrix, so that
    #            (orig-displacements-vect) = (dof_perm) � (new-displacements-vector)
    #
    # S. Adany, Aug 28, 2006
    # B. Schafer, Aug 29, 2006
    # Z. Li, Dec 22, 2009

    [main_nodes, meta_elements, node_props] = meta_elems(nodes=nodes, elements=elements)
    [n_main_nodes, n_corner_nodes, n_sub_nodes] = node_class(node_props=node_props)
    [n_dist_modes, n_local_modes] = mode_nr(n_main_nodes, n_corner_nodes, n_sub_nodes, main_nodes)
    dof_perm = dof_ordering(node_props)

    return main_nodes, meta_elements, node_props, n_main_nodes, \
        n_corner_nodes, n_sub_nodes, n_dist_modes, n_local_modes, dof_perm


def meta_elems(nodes, elements):
    # this routine re-organises the basic input data
    #  to eliminate internal subdividing nodes
    #  to form meta-elements (corner-to-corner or corner-to-free edge)
    #  to identify main nodes (corner nodes and free edge nodes)
    #
    # important assumption: cross section is opened!
    #
    # input/output data
    #   nodes, elements - same as elsewhere throughout this program
    #   main_nodes <main nodes> - array of
    #            [nr, x, z, orig nodes nr, nr of adj meta-elements, m-el_i-1, m-el_i-2, ...]
    #   meta_elements <meta-elements> - array of
    #            [nr, main-nodes-1, main-nodes-2, nr of sub-nodes, sub-no-1, sub-nod-2, ...]
    #   node_props - array of [original nodes nr, new nodes nr, nr of adj elements, nodes type]
    #
    # note:
    #   m-el_i-? is positive if the starting nodes of m-el_i-? coincides with
    #      the given m-nodes, otherwise negative
    #   nodes types: 1-corner, 2-edge, 3-sub
    #   sub-nodes numbers are the original ones, of course
    #
    # S. Adany, Feb 06, 2004
    #
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
            if elem[1] == i or elem[2] == i:
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

            a_1 = np.arctan2(nodes[n_2, 2] - nodes[n_1, 2], nodes[n_2, 1] - nodes[n_1, 1])  #?
            a_2 = np.arctan2(nodes[n_1, 2] - nodes[n_3, 2], nodes[n_1, 1] - nodes[n_3, 1])
            if abs(a_1 - a_2) < 1E-7:
                node_props[i, 2] = 0
                node_props[i, 3] = 3

    # to create meta-elements (with the original nodes numbers)
    meta_elements_temp = np.zeros((len(elements), 5))
    meta_elements_temp[:, 0:3] = elements[:, 0:3]
    for i in range(0, n_nodes):
        if node_props[i, 2] == 0:
            els = []
            for j, m_elem in enumerate(meta_elements_temp):
                if m_elem[1] == i or m_elem[2] == i:
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
            meta_elements_temp[els[0],
                               int(3
                                   + meta_elements_temp[els[0], 3])] = i  # zli:deleted elements no.

    # to eliminate disappearing elements (nodes numbers are still the original ones!)
    n_meta_elements = 0  # nr of meta-elements
    meta_elements = []
    for m_elem_t in meta_elements_temp:
        if m_elem_t[1] != -1 and m_elem_t[2] != -1:
            meta_elements.append(m_elem_t)
            meta_elements[-1][0] = n_meta_elements
            n_meta_elements = n_meta_elements + 1
    meta_elements = np.array(meta_elements)

    # to create array of main-nodes
    #(first and fourth columns assign the new vs. original numbering,
    # + node_assign tells the original vs. new numbering)
    n_main_nodes = 0  # nr of main nodes
    main_nodes = []
    for i, node in enumerate(nodes):
        if node_props[i, 2] != 0:
            main_nodes.append([n_main_nodes, node[1], node[2], i, node_props[i, 2]])
            node_props[i, 1] = n_main_nodes
            n_main_nodes = n_main_nodes + 1
    main_nodes = np.array(main_nodes)

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


def mode_nr(n_main_nodes, n_corner_nodes, n_sub_nodes, main_nodes):
    #
    # this routine determines the number of distortional and local buckling modes
    # if GBT-like assumptions are used
    #
    #
    # input/output data
    #   n_main_nodes, n_sub_nodes - number of main nodes and sub_nodes, respectively
    #   main_nodes [main nodes] - array of
    #            [nr, x, z, orig nodes nr, nr of adj meta-elements, m-el_i-1, m-el_i-2, ...]
    #   n_dist_modes, n_local_modes - number of distortional and local buckling modes, respectively
    #
    # S. Adany, Feb 09, 2004
    #
    #
    # to count the number of distortional modes
    n_dist_modes = n_main_nodes - 4
    for i in range(0, n_main_nodes):
        if main_nodes[i, 4] > 2:
            n_dist_modes = n_dist_modes - (main_nodes[i, 4] - 2)

    if n_dist_modes < 0:
        n_dist_modes = 0

    # to count the number of local modes
    n_edge_nodes = n_main_nodes - n_corner_nodes  # nr of edge nodes
    n_local_modes = n_main_nodes + 2*n_sub_nodes + n_edge_nodes

    return n_dist_modes, n_local_modes


def dof_ordering(node_props):
    # this routine re-orders the DOFs,
    # according to the need of forming shape vectors for various buckling modes
    #
    # input/output data
    #   node_props - array of [original nodes nr, new nodes nr, nr of adj elements, nodes type]
    #   dof_perm - permutation matrix, so that
    #            (orig-displacements-vect) = (dof_perm) � (new-displacements-vector)
    #
    # notes:
    # (1)  nodes types: 1-corner, 2-edge, 3-sub
    # (2)  the re-numbering of long. displacements. DOFs of main nodes, which may be
    #      necessary for dist. buckling, is not included here but handled
    #      separately when forming Ry constraint matrix
    #
    # S. Adany, Feb 06, 2004
    #
    #
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
            dof_perm[2 * i, n_main_nodes + 2*n_corner_nodes + i_e] = 1
            i_e = i_e + 1
        if n_prop[3] == 3:  # sub nodes
            dof_perm[2 * i, 4*n_main_nodes + i_s] = 1
            i_s = i_s + 1

    # y DOFs
    i_c = 0
    i_s = 0
    for i, n_prop in enumerate(node_props):
        if n_prop[3] == 1 or n_prop[3] == 2:  # corner or edge nodes
            dof_perm[2*i + 1, i_c] = 1
            i_c = i_c + 1
        if n_prop[3] == 3:  # sub nodes
            dof_perm[2*i + 1, 4*n_main_nodes + 3*n_sub_nodes + i_s] = 1
            i_s = i_s + 1

    # z DOFs
    i_c = 0
    i_e = 0
    i_s = 0
    for i, n_prop in enumerate(node_props):
        if n_prop[3] == 1:  # corner nodes
            dof_perm[2*n_node_props + 2*i, n_main_nodes + n_corner_nodes + i_c] = 1
            i_c = i_c + 1
        if n_prop[3] == 2:  # edge nodes
            dof_perm[2*n_node_props + 2*i, n_main_nodes + 2*n_corner_nodes + n_edge_nodes + i_e] = 1
            i_e = i_e + 1
        if n_prop[3] == 3:  # sub nodes
            dof_perm[2*n_node_props + 2*i, 4*n_main_nodes + n_sub_nodes + i_s] = 1
            i_s = i_s + 1

    # teta DOFs
    i_c = 0
    i_s = 0
    for i, n_prop in enumerate(node_props):
        if n_prop[3] == 1 or n_prop[3] == 2:  # corner or edge nodes
            dof_perm[2*n_node_props + 2*i + 1, 3*n_main_nodes + i_c] = 1
            i_c = i_c + 1
        if n_prop[3] == 3:  # sub nodes
            dof_perm[2*n_node_props + 2*i + 1, 4*n_main_nodes + 2*n_sub_nodes + i_s] = 1
            i_s = i_s + 1

    return dof_perm


def create_k_globals(m_i, nodes, elements, el_props, props, length, b_c):
    # called from base_update, while only single longitudinal term m_i involved
    #
    # created on Aug 28, 2006, by S. Adany
    # modified on Jul 10, 2009 by Z. Li

    # MATRIX SIZES
    n_nodes = len(nodes)

    # ZERO OUT THE GLOBAL MATRICES
    k_global = np.zeros((n_nodes * 4, n_nodes * 4))
    kg_global = np.zeros((n_nodes * 4, n_nodes * 4))

    # ASSEMBLE THE GLOBAL STIFFNESS MATRICES
    for i, elem in enumerate(elements):
        thick = elem[3]
        b_strip = el_props[i, 1]
        mat_num = int(elem[4])
        row = int((np.argwhere(props[:, 0] == mat_num)).reshape(1))
        mat = props[row]
        stiff_x = mat[1]
        stiff_y = mat[2]
        nu_x = mat[3]
        nu_y = mat[4]
        bulk = mat[5]
        k_l = klocal_m(
            stiff_x=stiff_x,
            stiff_y=stiff_y,
            nu_x=nu_x,
            nu_y=nu_y,
            bulk=bulk,
            thick=thick,
            length=length,
            b_strip=b_strip,
            b_c=b_c,
            m_i=m_i
        )
        # Generate geometric stiffness matrix (kg_local) in local coordinates
        node_i = int(elem[1])
        node_j = int(elem[2])
        ty_1 = nodes[node_i, 7] * thick
        ty_2 = nodes[node_j, 7] * thick
        kg_l = kglocal_m(length=length, b_strip=b_strip, ty_1=ty_1, ty_2=ty_2, b_c=b_c, m_i=m_i)
        # Transform k_local and kg_local into global coordinates
        alpha = el_props[i, 2]
        [k_local, kg_local] = trans_m(alpha, k_l, kg_l)

        # Add element contribution of k_local to full matrix k_global and kg_local to kg_global
        [k_global, kg_global] = assemble_m(
            k_global=k_global,
            kg_global=kg_global,
            k_local=k_local,
            kg_local=kg_local,
            node_i=node_i,
            node_j=node_j,
            n_nodes=n_nodes
        )

    return k_global, kg_global


def klocal_m(stiff_x, stiff_y, nu_x, nu_y, bulk, thick, length, b_strip, m_i, b_c):
    # assemble local elastic stiffness matrix for a single longitudinal term m_i
    #
    # created on Jul 10, 2009 by Z. Li

    # Generate element stiffness matrix (k_local) in local coordinates
    e_1 = stiff_x / (1 - nu_x*nu_y)
    e_2 = stiff_y / (1 - nu_x*nu_y)
    d_x = stiff_x * thick**3 / (12 * (1 - nu_x*nu_y))
    d_y = stiff_y * thick**3 / (12 * (1 - nu_x*nu_y))
    d_1 = nu_x * stiff_y * thick**3 / (12 * (1 - nu_x*nu_y))
    d_xy = bulk * thick**3 / 12
    #
    # k_local = sparse(np.zeros(8*m_i, 8*m_i))
    z_0 = np.zeros((4, 4))
    i = m_i
    j = m_i
    km_mp = np.zeros((4, 4))
    kf_mp = np.zeros((4, 4))
    u_m = i * np.pi
    u_p = j * np.pi
    c_1 = u_m / length
    c_2 = u_p / length
    #
    [i_1, i_2, i_3, i_4, i_5] = pycufsm.analysis.bc_i1_5(b_c, i, j, length)
    #
    # assemble the matrix of Km_mp
    km_mp[0, 0] = e_1*i_1/b_strip + bulk*b_strip*i_5/3
    km_mp[0, 1] = e_2 * nu_x * (-1 / 2 / c_2) * i_3 - bulk*i_5/2/c_2
    km_mp[0, 2] = -e_1 * i_1 / b_strip + bulk*b_strip*i_5/6
    km_mp[0, 3] = e_2 * nu_x * (-1 / 2 / c_2) * i_3 + bulk*i_5/2/c_2

    km_mp[1, 0] = e_2 * nu_x * (-1 / 2 / c_1) * i_2 - bulk*i_5/2/c_1
    km_mp[1, 1] = e_2*b_strip*i_4/3/c_1/c_2 + bulk*i_5/b_strip/c_1/c_2
    km_mp[1, 2] = e_2 * nu_x * (1/2/c_1) * i_2 - bulk*i_5/2/c_1
    km_mp[1, 3] = e_2*b_strip*i_4/6/c_1/c_2 - bulk*i_5/b_strip/c_1/c_2

    km_mp[2, 0] = -e_1 * i_1 / b_strip + bulk*b_strip*i_5/6
    km_mp[2, 1] = e_2 * nu_x * (1/2/c_2) * i_3 - bulk*i_5/2/c_2
    km_mp[2, 2] = e_1*i_1/b_strip + bulk*b_strip*i_5/3
    km_mp[2, 3] = e_2 * nu_x * (1/2/c_2) * i_3 + bulk*i_5/2/c_2

    km_mp[3, 0] = e_2 * nu_x * (-1 / 2 / c_1) * i_2 + bulk*i_5/2/c_1
    km_mp[3, 1] = e_2*b_strip*i_4/6/c_1/c_2 - bulk*i_5/b_strip/c_1/c_2
    km_mp[3, 2] = e_2 * nu_x * (1/2/c_1) * i_2 + bulk*i_5/2/c_1
    km_mp[3, 3] = e_2*b_strip*i_4/3/c_1/c_2 + bulk*i_5/b_strip/c_1/c_2
    km_mp = km_mp * thick
    #
    #
    # assemble the matrix of Kf_mp
    kf_mp[0, 0] = (5040*d_x*i_1 - 504*b_strip**2*d_1*i_2 - 504*b_strip**2*d_1*i_3 \
        + 156*b_strip**4*d_y*i_4 + 2016*b_strip**2*d_xy*i_5) / 420/b_strip**3
    kf_mp[0, 1] = (2520*b_strip*d_x*i_1 - 462*b_strip**3*d_1*i_2 - 42*b_strip**3*d_1*i_3 \
        + 22*b_strip**5*d_y*i_4 + 168*b_strip**3*d_xy*i_5) / 420/b_strip**3
    kf_mp[0, 2] = (-5040*d_x*i_1 + 504*b_strip**2*d_1*i_2 + 504*b_strip**2*d_1*i_3 \
        + 54*b_strip**4*d_y*i_4 - 2016*b_strip**2*d_xy*i_5) / 420/b_strip**3
    kf_mp[0, 3] = (2520*b_strip*d_x*i_1 - 42*b_strip**3*d_1*i_2 - 42*b_strip**3*d_1*i_3 \
        - 13*b_strip**5*d_y*i_4 + 168*b_strip**3*d_xy*i_5) / 420/b_strip**3

    kf_mp[1, 0] = (2520*b_strip*d_x*i_1 - 462*b_strip**3*d_1*i_3 - 42*b_strip**3*d_1*i_2 \
        + 22*b_strip**5*d_y*i_4 + 168*b_strip**3*d_xy*i_5) / 420/b_strip**3
    kf_mp[1, 1] = (1680*b_strip**2*d_x*i_1 - 56*b_strip**4*d_1*i_2 - 56*b_strip**4*d_1*i_3 \
        + 4*b_strip**6*d_y*i_4 + 224*b_strip**4*d_xy*i_5) / 420/b_strip**3
    kf_mp[1, 2] = (-2520*b_strip*d_x*i_1 + 42*b_strip**3*d_1*i_2 + 42*b_strip**3*d_1*i_3 \
        + 13*b_strip**5*d_y*i_4 - 168*b_strip**3*d_xy*i_5) / 420/b_strip**3
    kf_mp[1, 3] = (840*b_strip**2*d_x*i_1 + 14*b_strip**4*d_1*i_2 + 14*b_strip**4*d_1*i_3 \
        - 3*b_strip**6*d_y*i_4 - 56*b_strip**4*d_xy*i_5) / 420/b_strip**3

    kf_mp[2, 0] = kf_mp[0, 2]
    kf_mp[2, 1] = kf_mp[1, 2]
    kf_mp[2, 2] = (5040*d_x*i_1 - 504*b_strip**2*d_1*i_2 - 504*b_strip**2*d_1*i_3 \
        + 156*b_strip**4*d_y*i_4 + 2016*b_strip**2*d_xy*i_5) / 420/b_strip**3
    kf_mp[2, 3] = (-2520*b_strip*d_x*i_1 + 462*b_strip**3*d_1*i_2 + 42*b_strip**3*d_1*i_3 \
        - 22*b_strip**5*d_y*i_4 - 168*b_strip**3*d_xy*i_5) / 420/b_strip**3

    kf_mp[3, 0] = kf_mp[0, 3]
    kf_mp[3, 1] = kf_mp[1, 3]
    kf_mp[3, 2] = (-2520*b_strip*d_x*i_1 + 462*b_strip**3*d_1*i_3 + 42*b_strip**3*d_1*i_2 \
        - 22*b_strip**5*d_y*i_4 - 168*b_strip**3*d_xy*i_5) / 420/b_strip**3 # not symmetric
    kf_mp[3, 3] = (1680*b_strip**2*d_x*i_1 - 56*b_strip**4*d_1*i_2 - 56*b_strip**4*d_1*i_3 \
        + 4*b_strip**6*d_y*i_4 + 224*b_strip**4*d_xy*i_5) / 420/b_strip**3

    # assemble the membrane and flexural stiffness matrices
    kmp = np.concatenate(
        (np.concatenate((km_mp, z_0), axis=1), np.concatenate((z_0, kf_mp), axis=1))
    )
    # add it into local element stiffness matrix by corresponding to m_i
    k_local = kmp

    return k_local


def kglocal_m(length, b_strip, m_i, ty_1, ty_2, b_c):
    # assemble local geometric stiffness matrix for a single longitudinal term m_i

    # created on Jul 10, 2009 by Z. Li

    # Generate geometric stiffness matrix (kg_local) in local coordinates
    # kg_local = sparse(np.zeros(8*m_i, 8*m_i))
    i = m_i
    j = m_i
    gm_mp = np.zeros((4, 4))
    z_0 = np.zeros((4, 4))
    gf_mp = np.zeros((4, 4))
    u_m = i * np.pi
    u_p = j * np.pi
    #
    [_, _, _, i_4, i_5] = pycufsm.analysis.bc_i1_5(b_c, i, j, length)
    #
    # assemble the matrix of gm_mp (symmetric membrane stability matrix)
    gm_mp[0, 0] = b_strip * (3*ty_1 + ty_2) * i_5 / 12
    gm_mp[0, 2] = b_strip * (ty_1+ty_2) * i_5 / 12
    gm_mp[2, 0] = gm_mp[0, 2]
    gm_mp[1, 1] = b_strip * length**2 * (3*ty_1 + ty_2) * i_4 / 12 / u_m / u_p
    gm_mp[1, 3] = b_strip * length**2 * (ty_1+ty_2) * i_4 / 12 / u_m / u_p
    gm_mp[3, 1] = gm_mp[1, 3]
    gm_mp[2, 2] = b_strip * (ty_1 + 3*ty_2) * i_5 / 12
    gm_mp[3, 3] = b_strip * length**2 * (ty_1 + 3*ty_2) * i_4 / 12 / u_m / u_p
    #
    # assemble the matrix of gf_mp (symmetric flexural stability matrix)
    gf_mp[0, 0] = (10*ty_1 + 3*ty_2) * b_strip * i_5 / 35
    gf_mp[0, 1] = (15*ty_1 + 7*ty_2) * b_strip**2 * i_5 / 210 / 2
    gf_mp[1, 0] = gf_mp[0, 1]
    gf_mp[0, 2] = 9 * (ty_1+ty_2) * b_strip * i_5 / 140
    gf_mp[2, 0] = gf_mp[0, 2]
    gf_mp[0, 3] = -(7*ty_1 + 6*ty_2) * b_strip**2 * i_5 / 420
    gf_mp[3, 0] = gf_mp[0, 3]
    gf_mp[1, 1] = (5*ty_1 + 3*ty_2) * b_strip**3 * i_5 / 2 / 420
    gf_mp[1, 2] = (6*ty_1 + 7*ty_2) * b_strip**2 * i_5 / 420
    gf_mp[2, 1] = gf_mp[1, 2]
    gf_mp[1, 3] = -(ty_1 + ty_2) * b_strip**3 * i_5 / 140 / 2
    gf_mp[3, 1] = gf_mp[1, 3]
    gf_mp[2, 2] = (3*ty_1 + 10*ty_2) * b_strip * i_5 / 35
    gf_mp[2, 3] = -(7*ty_1 + 15*ty_2) * b_strip**2 * i_5 / 420
    gf_mp[3, 2] = gf_mp[2, 3]
    gf_mp[3, 3] = (3*ty_1 + 5*ty_2) * b_strip**3 * i_5 / 420 / 2
    # assemble the membrane and flexural stiffness matrices
    kg_mp = np.concatenate(
        (np.concatenate((gm_mp, z_0), axis=1), np.concatenate((z_0, gf_mp), axis=1))
    )
    # add it into local geometric stiffness matrix by corresponding to m_i
    kg_local = kg_mp
    return kg_local


def trans_m(alpha, k_local, kg_local):
    # transfer the local stiffness into global stiffness

    # created on Jul 10, 2009 by Z. Li
    gamma = np.array([[np.cos(alpha), 0, 0, 0, -np.sin(alpha), 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, np.cos(alpha), 0, 0, 0, -np.sin(alpha), 0], [0, 0, 0, 1, 0, 0, 0, 0],
                      [np.sin(alpha), 0, 0, 0, np.cos(alpha), 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, np.sin(alpha), 0, 0, 0, np.cos(alpha), 0], [0, 0, 0, 0, 0, 0, 0, 1]])
    # extend to multi-m
    # for i = 1:m_i
    #     gamma(8*(i-1)+1:8*i, 8*(i-1)+1:8*i) = gam
    # end
    #
    k_global = gamma @ k_local @ gamma.conj().T
    kg_global = gamma @ kg_local @ gamma.conj().T

    return k_global, kg_global


def assemble_m(k_global, kg_global, k_local, kg_local, node_i, node_j, n_nodes):
    # BWS
    # 1997
    # Add the element contribution to the global stiffness matrix for single
    # longitudinal term m_i

    # modifed on Jul 10, 2009 by Z. Li

    # Submatrices for the initial stiffness
    k11 = k_local[0:2, 0:2]
    k12 = k_local[0:2, 2:4]
    k13 = k_local[0:2, 4:6]
    k14 = k_local[0:2, 6:8]
    k21 = k_local[2:4, 0:2]
    k22 = k_local[2:4, 2:4]
    k23 = k_local[2:4, 4:6]
    k24 = k_local[2:4, 6:8]
    k31 = k_local[4:6, 0:2]
    k32 = k_local[4:6, 2:4]
    k33 = k_local[4:6, 4:6]
    k34 = k_local[4:6, 6:8]
    k41 = k_local[6:8, 0:2]
    k42 = k_local[6:8, 2:4]
    k43 = k_local[6:8, 4:6]
    k44 = k_local[6:8, 6:8]
    #
    # Submatrices for the geometric stiffness
    kg11 = kg_local[0:2, 0:2]
    kg12 = kg_local[0:2, 2:4]
    kg13 = kg_local[0:2, 4:6]
    kg14 = kg_local[0:2, 6:8]
    kg21 = kg_local[2:4, 0:2]
    kg22 = kg_local[2:4, 2:4]
    kg23 = kg_local[2:4, 4:6]
    kg24 = kg_local[2:4, 6:8]
    kg31 = kg_local[4:6, 0:2]
    kg32 = kg_local[4:6, 2:4]
    kg33 = kg_local[4:6, 4:6]
    kg34 = kg_local[4:6, 6:8]
    kg41 = kg_local[6:8, 0:2]
    kg42 = kg_local[6:8, 2:4]
    kg43 = kg_local[6:8, 4:6]
    kg44 = kg_local[6:8, 6:8]
    #
    k_2_matrix = np.zeros((4 * n_nodes, 4 * n_nodes))
    k_3_matrix = np.zeros((4 * n_nodes, 4 * n_nodes))
    #
    # The additional terms for k_global are stored in k_2_matrix
    skip = 2 * n_nodes
    k_2_matrix[node_i * 2:node_i*2 + 2, node_i * 2:node_i*2 + 2] = k11
    k_2_matrix[node_i * 2:node_i*2 + 2, node_j * 2:node_j*2 + 2] = k12
    k_2_matrix[node_j * 2:node_j*2 + 2, node_i * 2:node_i*2 + 2] = k21
    k_2_matrix[node_j * 2:node_j*2 + 2, node_j * 2:node_j*2 + 2] = k22
    #
    k_2_matrix[skip + node_i*2:skip + node_i*2 + 2, skip + node_i*2:skip + node_i*2 + 2] = k33
    k_2_matrix[skip + node_i*2:skip + node_i*2 + 2, skip + node_j*2:skip + node_j*2 + 2] = k34
    k_2_matrix[skip + node_j*2:skip + node_j*2 + 2, skip + node_i*2:skip + node_i*2 + 2] = k43
    k_2_matrix[skip + node_j*2:skip + node_j*2 + 2, skip + node_j*2:skip + node_j*2 + 2] = k44
    #
    k_2_matrix[node_i * 2:node_i*2 + 2, skip + node_i*2:skip + node_i*2 + 2] = k13
    k_2_matrix[node_i * 2:node_i*2 + 2, skip + node_j*2:skip + node_j*2 + 2] = k14
    k_2_matrix[node_j * 2:node_j*2 + 2, skip + node_i*2:skip + node_i*2 + 2] = k23
    k_2_matrix[node_j * 2:node_j*2 + 2, skip + node_j*2:skip + node_j*2 + 2] = k24
    #
    k_2_matrix[skip + node_i*2:skip + node_i*2 + 2, node_i * 2:node_i*2 + 2] = k31
    k_2_matrix[skip + node_i*2:skip + node_i*2 + 2, node_j * 2:node_j*2 + 2] = k32
    k_2_matrix[skip + node_j*2:skip + node_j*2 + 2, node_i * 2:node_i*2 + 2] = k41
    k_2_matrix[skip + node_j*2:skip + node_j*2 + 2, node_j * 2:node_j*2 + 2] = k42
    k_global = k_global + k_2_matrix
    #
    # The additional terms for kg_global are stored in k_3_matrix
    k_3_matrix[node_i * 2:node_i*2 + 2, node_i * 2:node_i*2 + 2] = kg11
    k_3_matrix[node_i * 2:node_i*2 + 2, node_j * 2:node_j*2 + 2] = kg12
    k_3_matrix[node_j * 2:node_j*2 + 2, node_i * 2:node_i*2 + 2] = kg21
    k_3_matrix[node_j * 2:node_j*2 + 2, node_j * 2:node_j*2 + 2] = kg22
    #
    k_3_matrix[skip + node_i*2:skip + node_i*2 + 2, skip + node_i*2:skip + node_i*2 + 2] = kg33
    k_3_matrix[skip + node_i*2:skip + node_i*2 + 2, skip + node_j*2:skip + node_j*2 + 2] = kg34
    k_3_matrix[skip + node_j*2:skip + node_j*2 + 2, skip + node_i*2:skip + node_i*2 + 2] = kg43
    k_3_matrix[skip + node_j*2:skip + node_j*2 + 2, skip + node_j*2:skip + node_j*2 + 2] = kg44
    #
    k_3_matrix[node_i * 2:node_i*2 + 2, skip + node_i*2:skip + node_i*2 + 2] = kg13
    k_3_matrix[node_i * 2:node_i*2 + 2, skip + node_j*2:skip + node_j*2 + 2] = kg14
    k_3_matrix[node_j * 2:node_j*2 + 2, skip + node_i*2:skip + node_i*2 + 2] = kg23
    k_3_matrix[node_j * 2:node_j*2 + 2, skip + node_j*2:skip + node_j*2 + 2] = kg24
    #
    k_3_matrix[skip + node_i*2:skip + node_i*2 + 2, node_i * 2:node_i*2 + 2] = kg31
    k_3_matrix[skip + node_i*2:skip + node_i*2 + 2, node_j * 2:node_j*2 + 2] = kg32
    k_3_matrix[skip + node_j*2:skip + node_j*2 + 2, node_i * 2:node_i*2 + 2] = kg41
    k_3_matrix[skip + node_j*2:skip + node_j*2 + 2, node_j * 2:node_j*2 + 2] = kg42
    #
    kg_global = kg_global + k_3_matrix
    return k_global, kg_global


def kglobal_transv(nodes, elements, props, m_i, length, b_c, el_props):
    #
    # this routine creates the global stiffness matrix for planar displacements
    # basically the same way as in the main program, however:
    #   only one half-wave number m_i is considered,
    #   only w, teta terms are considered,
    #   plus stiff_y = nu_x = nu_y = 0 is assumed
    #   plus the longitudinal displacements. DOFs are explicitely eliminated
    #   the multiplication by 'length' (member length) is not done here, must be done
    #      outside of this routine
    #
    # input/output data
    #   nodes, elements, props - same as elsewhere throughout this program
    #   m_i - number of half waves
    #   k_global_transv - global stiffness matrix (geometric not included)
    #
    # S. Adany, Feb 08, 2004
    # Z. Li, Jul 10, 2009
    #
    n_nodes = len(nodes)
    k_global_transv = np.zeros((4 * n_nodes, 4 * n_nodes))
    #
    for i, elem in enumerate(elements):
        thick = elem[3]
        b_strip = el_props[i, 1]
        mat_num = int(elem[4])
        row = int((np.argwhere(props[:, 0] == mat_num)).reshape(1))
        mat = props[row]
        stiff_x = mat[1]
        stiff_y = mat[2]
        nu_x = mat[3]
        nu_y = mat[4]
        bulk = mat[5]
        k_l = klocal_transv(
            stiff_x=stiff_x,
            stiff_y=stiff_y,
            nu_x=nu_x,
            nu_y=nu_y,
            bulk=bulk,
            thick=thick,
            length=length,
            b_strip=b_strip,
            b_c=b_c,
            m_i=m_i
        )

        # Transform k_local and kg_local into global coordinates
        alpha = el_props[i, 2]
        k_local = trans_single(alpha, k_l)

        # Add element contribution of k_local to full matrix k_global and kg_local to kg_global
        node_i = int(elem[1])
        node_j = int(elem[2])
        k_global_transv = assemble_single(
            k_global=k_global_transv,
            k_local=k_local,
            node_i=node_i,
            node_j=node_j,
            n_nodes=n_nodes
        )

    return k_global_transv


def klocal_transv(stiff_x, stiff_y, nu_x, nu_y, bulk, thick, length, b_strip, m_i, b_c):
    #
    # this routine creates the local stiffness matrix for bending terms
    # basically the same way as in the main program, however:
    #   only for single half-wave number m_i
    #   membrane strains practically zero, (membrane moduli are enlarged)
    #   for bending, only transverse terms are considered, (practically: only
    #   keeps the i_1 term, set i_2 through i_5 to be zero)
    # also different from the main program, here only involves one single
    # longitudinal term m_i.
    #
    # input/output data
    #   nodes, elements, props - same as elsewhere throughout this program
    #   k_global_transv - global stiffness matrix (geometric included)
    #
    # Z. Li, Jul 10, 2009

    e_1 = stiff_x / (1 - nu_x*nu_y) * 100000000
    e_2 = stiff_y / (1 - nu_x*nu_y)
    d_x = stiff_x * thick**3 / (12 * (1 - nu_x*nu_y))
    d_y = stiff_y * thick**3 / (12 * (1 - nu_x*nu_y))
    d_1 = nu_x * stiff_y * thick**3 / (12 * (1 - nu_x*nu_y))
    d_xy = bulk * thick**3 / 12

    z_0 = np.zeros((4, 4))
    i = m_i
    j = m_i
    km_mp = np.zeros((4, 4))
    kf_mp = np.zeros((4, 4))
    u_m = i * np.pi
    u_p = j * np.pi
    c_1 = u_m / length
    c_2 = u_p / length

    [i_1, _, _, _, _] = pycufsm.analysis.bc_i1_5(b_c, i, j, length)
    i_2 = 0
    i_3 = 0
    i_4 = 0
    i_5 = 0

    # assemble in-plane stiffness matrix of Km_mp
    km_mp[0, 0] = e_1*i_1/b_strip + bulk*b_strip*i_5/3
    km_mp[0, 1] = e_2 * nu_x * (-1 / 2 / c_2) * i_3 - bulk*i_5/2/c_2
    km_mp[0, 2] = -e_1 * i_1 / b_strip + bulk*b_strip*i_5/6
    km_mp[0, 3] = e_2 * nu_x * (-1 / 2 / c_2) * i_3 + bulk*i_5/2/c_2

    km_mp[1, 0] = e_2 * nu_x * (-1 / 2 / c_1) * i_2 - bulk*i_5/2/c_1
    km_mp[1, 1] = e_2*b_strip*i_4/3/c_1/c_2 + bulk*i_5/b_strip/c_1/c_2
    km_mp[1, 2] = e_2 * nu_x * (1/2/c_1) * i_2 - bulk*i_5/2/c_1
    km_mp[1, 3] = e_2*b_strip*i_4/6/c_1/c_2 - bulk*i_5/b_strip/c_1/c_2

    km_mp[2, 0] = -e_1 * i_1 / b_strip + bulk*b_strip*i_5/6
    km_mp[2, 1] = e_2 * nu_x * (1/2/c_2) * i_3 - bulk*i_5/2/c_2
    km_mp[2, 2] = e_1*i_1/b_strip + bulk*b_strip*i_5/3
    km_mp[2, 3] = e_2 * nu_x * (1/2/c_2) * i_3 + bulk*i_5/2/c_2

    km_mp[3, 0] = e_2 * nu_x * (-1 / 2 / c_1) * i_2 + bulk*i_5/2/c_1
    km_mp[3, 1] = e_2*b_strip*i_4/6/c_1/c_2 - bulk*i_5/b_strip/c_1/c_2
    km_mp[3, 2] = e_2 * nu_x * (1/2/c_1) * i_2 + bulk*i_5/2/c_1
    km_mp[3, 3] = e_2*b_strip*i_4/3/c_1/c_2 + bulk*i_5/b_strip/c_1/c_2
    km_mp = km_mp * thick

    # assemble the bending stiffness matrix of Kf_mp
    kf_mp[0, 0] = (5040*d_x*i_1 - 504*b_strip**2*d_1*i_2 - 504*b_strip**2*d_1*i_3 \
        + 156*b_strip**4*d_y*i_4 + 2016*b_strip**2*d_xy*i_5) / 420/b_strip**3
    kf_mp[0, 1] = (2520*b_strip*d_x*i_1 - 462*b_strip**3*d_1*i_2 - 42*b_strip**3*d_1*i_3 \
        + 22*b_strip**5*d_y*i_4 + 168*b_strip**3*d_xy*i_5) / 420/b_strip**3
    kf_mp[0, 2] = (-5040*d_x*i_1 + 504*b_strip**2*d_1*i_2 + 504*b_strip**2*d_1*i_3 \
        + 54*b_strip**4*d_y*i_4 - 2016*b_strip**2*d_xy*i_5) / 420/b_strip**3
    kf_mp[0, 3] = (2520*b_strip*d_x*i_1 - 42*b_strip**3*d_1*i_2 - 42*b_strip**3*d_1*i_3 \
        - 13*b_strip**5*d_y*i_4 + 168*b_strip**3*d_xy*i_5) / 420/b_strip**3

    kf_mp[1, 0] = (2520*b_strip*d_x*i_1 - 462*b_strip**3*d_1*i_3 - 42*b_strip**3*d_1*i_2 \
        + 22*b_strip**5*d_y*i_4 + 168*b_strip**3*d_xy*i_5) / 420/b_strip**3
    kf_mp[1, 1] = (1680*b_strip**2*d_x*i_1 - 56*b_strip**4*d_1*i_2 - 56*b_strip**4*d_1*i_3 \
        + 4*b_strip**6*d_y*i_4 + 224*b_strip**4*d_xy*i_5) / 420/b_strip**3
    kf_mp[1, 2] = (-2520*b_strip*d_x*i_1 + 42*b_strip**3*d_1*i_2 + 42*b_strip**3*d_1*i_3 \
        + 13*b_strip**5*d_y*i_4 - 168*b_strip**3*d_xy*i_5) / 420/b_strip**3
    kf_mp[1, 3] = (840*b_strip**2*d_x*i_1 + 14*b_strip**4*d_1*i_2 + 14*b_strip**4*d_1*i_3 \
        - 3*b_strip**6*d_y*i_4 - 56*b_strip**4*d_xy*i_5) / 420/b_strip**3

    kf_mp[2, 0] = kf_mp[0, 2]
    kf_mp[2, 1] = kf_mp[1, 2]
    kf_mp[2, 2] = (5040*d_x*i_1 - 504*b_strip**2*d_1*i_2 - 504*b_strip**2*d_1*i_3 \
        + 156*b_strip**4*d_y*i_4 + 2016*b_strip**2*d_xy*i_5) / 420/b_strip**3
    kf_mp[2, 3] = (-2520*b_strip*d_x*i_1 + 462*b_strip**3*d_1*i_2 + 42*b_strip**3*d_1*i_3 \
        - 22*b_strip**5*d_y*i_4 - 168*b_strip**3*d_xy*i_5) / 420/b_strip**3

    kf_mp[3, 0] = kf_mp[0, 3]
    kf_mp[3, 1] = kf_mp[1, 3]
    kf_mp[3, 2] = (-2520*b_strip*d_x*i_1 + 462*b_strip**3*d_1*i_3 + 42*b_strip**3*d_1*i_2 \
        - 22*b_strip**5*d_y*i_4 - 168*b_strip**3*d_xy*i_5) / 420/b_strip**3 # not symmetric
    kf_mp[3, 3] = (1680*b_strip**2*d_x*i_1 - 56*b_strip**4*d_1*i_2 - 56*b_strip**4*d_1*i_3 \
        + 4*b_strip**6*d_y*i_4 + 224*b_strip**4*d_xy*i_5) / 420/b_strip**3

    # assemble the membrane and flexural stiffness matrices
    kmp = np.concatenate(
        (np.concatenate((km_mp, z_0), axis=1), np.concatenate((z_0, kf_mp), axis=1))
    )

    # local stiffness matrix:
    k_local = kmp
    return k_local


def assemble_single(k_global, k_local, node_i, node_j, n_nodes):
    #
    # this routine adds the element contribution to the global stiffness matrix
    # basically it does the same as routine 'assemble', however:
    #   it does not care about kg_global (geom stiff matrix)
    #   only involves single half-wave number m_i

    # S. Adany, Feb 06, 2004
    # Z. Li, Jul 10, 2009
    #
    # submatrices for the initial stiffness
    k11 = k_local[0:2, 0:2]
    k12 = k_local[0:2, 2:4]
    k13 = k_local[0:2, 4:6]
    k14 = k_local[0:2, 6:8]
    k21 = k_local[2:4, 0:2]
    k22 = k_local[2:4, 2:4]
    k23 = k_local[2:4, 4:6]
    k24 = k_local[2:4, 6:8]
    k31 = k_local[4:6, 0:2]
    k32 = k_local[4:6, 2:4]
    k33 = k_local[4:6, 4:6]
    k34 = k_local[4:6, 6:8]
    k41 = k_local[6:8, 0:2]
    k42 = k_local[6:8, 2:4]
    k43 = k_local[6:8, 4:6]
    k44 = k_local[6:8, 6:8]

    k_2_matrix = np.zeros((4 * n_nodes, 4 * n_nodes))

    # the additional terms for k_global are stored in k_2_matrix
    skip = 2 * n_nodes
    k_2_matrix[node_i * 2:node_i*2 + 2, node_i * 2:node_i*2 + 2] = k11
    k_2_matrix[node_i * 2:node_i*2 + 2, node_j * 2:node_j*2 + 2] = k12
    k_2_matrix[node_j * 2:node_j*2 + 2, node_i * 2:node_i*2 + 2] = k21
    k_2_matrix[node_j * 2:node_j*2 + 2, node_j * 2:node_j*2 + 2] = k22

    k_2_matrix[skip + node_i*2:skip + node_i*2 + 2, skip + node_i*2:skip + node_i*2 + 2] = k33
    k_2_matrix[skip + node_i*2:skip + node_i*2 + 2, skip + node_j*2:skip + node_j*2 + 2] = k34
    k_2_matrix[skip + node_j*2:skip + node_j*2 + 2, skip + node_i*2:skip + node_i*2 + 2] = k43
    k_2_matrix[skip + node_j*2:skip + node_j*2 + 2, skip + node_j*2:skip + node_j*2 + 2] = k44

    k_2_matrix[node_i * 2:node_i*2 + 2, skip + node_i*2:skip + node_i*2 + 2] = k13
    k_2_matrix[node_i * 2:node_i*2 + 2, skip + node_j*2:skip + node_j*2 + 2] = k14
    k_2_matrix[node_j * 2:node_j*2 + 2, skip + node_i*2:skip + node_i*2 + 2] = k23
    k_2_matrix[node_j * 2:node_j*2 + 2, skip + node_j*2:skip + node_j*2 + 2] = k24

    k_2_matrix[skip + node_i*2:skip + node_i*2 + 2, node_i * 2:node_i*2 + 2] = k31
    k_2_matrix[skip + node_i*2:skip + node_i*2 + 2, node_j * 2:node_j*2 + 2] = k32
    k_2_matrix[skip + node_j*2:skip + node_j*2 + 2, node_i * 2:node_i*2 + 2] = k41
    k_2_matrix[skip + node_j*2:skip + node_j*2 + 2, node_j * 2:node_j*2 + 2] = k42
    k_global = k_global + k_2_matrix

    return k_global


def classify(props, nodes, elements, lengths, shapes, gbt_con, b_c, m_all, sect_props):
    # , clas_GDLO
    # MODAL CLASSIFICATION

    # input
    # props: [mat_num stiff_x stiff_y nu_x nu_y bulk] 6 x nmats
    # nodes: [nodes# x z dof_x dof_z dof_y dofrot stress] n_nodes x 8
    # elements: [elements# node_i node_j thick mat_num] n_elements x 5
    # lengths: lengths to be analyzed
    # shapes: array of mode shapes dof x lengths x mode
    # method:
    #   method = 1 = vector norm
    #   method = 2 = strain energy norm
    #   method = 3 = work norm
    #
    #
    # output
    # clas: array or # classification

    # BWS August 29, 2006
    # modified SA, Oct 10, 2006
    # Z.Li, June 2010
    n_nodes = len(nodes)
    n_dof_m = 4 * n_nodes

    # CLEAN UP INPUT
    # clean u_p 0's, multiple terms. or out-of-order terms in m_all
    m_all = pycufsm.analysis.m_sort(m_all)

    # FIND BASE PROPERTIES
    el_props = pycufsm.analysis.elem_prop(nodes=nodes, elements=elements)
    # set u_p stress to 1.0 for finding kg_global and k_global for axial modes
    nodes_base = deepcopy(nodes)
    nodes_base[:, 7] = np.ones_like(nodes[:, 7])

    # natural base first
    # properties all the longitudinal terms share
    [main_nodes, meta_elements, node_props, n_main_nodes, \
        n_corner_nodes, n_sub_nodes, n_dist_modes, n_local_modes, dof_perm] \
        = base_properties(nodes=nodes_base, elements=elements)
    [r_x, r_z, r_yd, r_ys, r_ud] = mode_constr(
        nodes=nodes_base,
        elements=elements,
        node_props=node_props,
        main_nodes=main_nodes,
        meta_elements=meta_elements
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
        el_props=el_props
    )

    # loop for the lengths
    n_lengths = len(lengths)
    l_i = 0  # length_index = one
    clas = []
    while l_i < n_lengths:
        length = lengths(l_i)
        # longitudinal terms included in the analysis for this length
        m_a = m_all[l_i]
        b_v_l = base_column(
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
        # orthonormal vectors
        b_v = base_update(
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

        # classification
        clas_modes = np.zeros((len(shapes([l_i][0])), 4))
        for mod in range(0, len(shapes[l_i][0])):
            clas_modes[mod, 0:4] = mode_class(
                b_v=b_v,
                displacements=shapes[l_i][:, mod],
                n_global_modes=n_global_modes,
                n_dist_modes=n_dist_modes,
                n_local_modes=n_local_modes,
                m_a=m_a,
                n_dof_m=n_dof_m,
                gbt_con=gbt_con
            )
        clas.append(clas_modes)
        l_i = l_i + 1  # length index = length index + one

    return clas


def mode_class(
    b_v, displacements, n_global_modes, n_dist_modes, n_local_modes, m_a, n_dof_m, gbt_con
):
    #
    # to determine mode contribution in the current displacement

    # input data
    #   b_v - base vectors (each column corresponds to a certain mode)
    #           columns 1..n_global_modes: global modes
    #           columns (n_global_modes+1)..(n_global_modes+n_dist_modes): dist. modes
    #           columns (n_global_modes+n_dist_modes+1)
    #                    ..(n_global_modes+n_dist_modes+n_local_modes): local modes
    #           columns (n_global_modes+n_dist_modes+n_local_modes+1)..n_dof: other modes
    #   displacements - vector of nodal displacements
    #   n_global_modes, n_dist_modes, n_local_modes
    #            - number of global, distortional and local buckling modes, respectively
    #   gbt_con['couple'] - by gbt_con, coupled basis vs uncoupled basis for general B.C.
    #                       especially for non-simply supported B.C.
    #         1: uncoupled basis, the basis will be block diagonal
    #         2: coupled basis, the basis is fully spanned

    # output data
    #   clas_gdlo - array with the contributions of the modes in percentage
    #               elem1: global, elem2: dist, elem3: local, elem4: other

    # S. Adany, Mar 10, 2004
    # Z. Li, June 2010

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

    if gbt_con['couple'] == 1:
        # uncoupled basis
        for i in range(0, len(m_a)):
            b_v_m = b_v[n_dof_m * i:n_dof_m * (i+1), n_dof_m * i:n_dof_m * (i+1)]

            # classification
            clas = np.linalg.lstsq(
                b_v_m[:, dof_index[0, 0]:dof_index[3, 1]],
                displacements[n_dof_m * i:n_dof_m * (i+1)]
            )

            cl_gdlo = np.zeros((4, 5 * n_modes))
            for j in range(0, 4):
                n_modes = dof_index[j, 1] - dof_index[i, 0]
                cl_gdlo[i, j * n_modes:j*n_modes + n_modes] = clas[dof_index[j, 0]:dof_index[j, 1]]

    #     # L1 norm
    #     for m_n = 1:4
    #         clas_gdlo1(m_n) = sum(abs(cl_gdlo(m_n,:)))
    #
    #     norm_sum = sum(clas_gdlo1)
    #     clas_gdlo1 = clas_gdlo1/norm_sum*100

    # L2 norm
        for m_n in range(0, 4):
            clas_gdlo[m_n] = np.linalg.norm(cl_gdlo[m_n, :])

        norm_sum = sum(clas_gdlo)
        clas_gdlo = clas_gdlo / norm_sum * 100
    else:
        # coupled basis
        # classification
        clas = np.linalg.lstsq(b_v, displacements)
        v_gdlo = np.zeros((4, (total_m+1) * n_modes))
        for i in range(0, 4):
            for j in range(0, total_m):
                n_modes = dof_index[i, 2] - dof_index[i, 1] + 1
                v_gdlo[i, j*n_modes:j*n_modes+n_modes] \
                    = clas[j*n_dof_m + dof_index[i, 1]:j*n_dof_m + dof_index[i, 2]]

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


def trans_single(alpha, k_local):
    #
    # this routine make the local-to-global co-ordinate transformation
    # basically it does the same as routine 'trans', however:
    #   it does not care about kg_local (geom stiff matrix)
    #   only involve one half-wave number m_i

    # S. Adany, Feb 06, 2004
    # Z. Li, Jul 10, 2009
    #
    gamma = np.array([[np.cos(alpha), 0, 0, 0, -np.sin(alpha), 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, np.cos(alpha), 0, 0, 0, -np.sin(alpha), 0], [0, 0, 0, 1, 0, 0, 0, 0],
                      [np.sin(alpha), 0, 0, 0, np.cos(alpha), 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, np.sin(alpha), 0, 0, 0, np.cos(alpha), 0], [0, 0, 0, 0, 0, 0, 0, 1]])

    k_global = gamma @ k_local @ gamma.conj().T

    return k_global


def node_class(node_props):
    #this routine determines how many nodes of the various types exist
    #
    #input/output data
    #   node_props - array of [original node nr, new node nr, nr of adj elems, node type]
    #   nmno,ncno,nsno - number of main nodes, corner nodes and sub-nodes, respectively
    #
    #notes:
    #   node types in node_props: 1-corner, 2-edge, 3-sub
    #   sub-node numbers are the original one, of course
    #
    # S. Adany, Feb 09, 2004

    #to count corner-, edge- and sub-nodes
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

    n_main_nodes = n_corner_nodes + n_edge_nodes  #nr of main nodes

    return n_main_nodes, n_corner_nodes, n_sub_nodes
