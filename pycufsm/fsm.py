from copy import deepcopy
from scipy import linalg as spla
import numpy as np
import pycufsm.analysis
import pycufsm.cfsm
# from scipy.sparse.linalg import eigs
# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE, CPEng
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


def strip(
    props, nodes, elements, lengths, springs, constraints, gbt_con, b_c, m_all, n_eigs, sect_props
):

    # INPUTS
    # props: [mat_num stiff_x stiff_y nu_x nu_y bulk] 6 x n_mats
    # nodes: [node# x y dof_x dof_y dof_z dof_r stress] n_nodes x 8
    # elements: [elem# node_i node_j thick mat_num] n_elements x 5
    # lengths: [L1 L2 L3...] 1 x n_lengths lengths to be analysed
    # could be half-wavelengths for signature curve
    # or physical lengths for general b.c.
    # springs: [node# d.o.f. k_spring k_flag] where 1=x dir 2= y dir 3 = z dir 4 = q dir (twist)
    #     flag says if k_stiff is a foundation stiffness or a total stiffness
    # constraints:: [node# e dof_e coeff node# k dof_k] e=dof to be eliminated
    #     k=kept dof dof_e_node = coeff*dof_k_node_k
    # gbt_con: gbt_con.glob,gbt_con.dist, gbt_con.local, gbt_con.other vectors of 1's
    #  and 0's referring to the inclusion (1) or exclusion of a given mode from the analysis,
    #  gbt_con.o_space - choices of ST/O mode
    #         1: ST basis
    #         2: O space (null space of GDL) with respect to k_global
    #         3: O space (null space of GDL) with respect to kg_global
    #         4: O space (null space of GDL) in vector sense
    #  gbt_con.norm - code for normalization (if normalization is done at all)
    #         0: no normalization,
    #         1: vector norm
    #         2: strain energy norm
    #         3: work norm
    #  gbt_con.couple - coupled basis vs uncoupled basis for general
    #             B.C. especially for non-simply supported B.C.
    #         1: uncoupled basis, the basis will be block diagonal
    #         2: coupled basis, the basis is fully spanned
    #  gbt_con.orth - natural basis vs modal basis
    #         1: natural basis
    #         2: modal basis, axial orthogonality
    #         3: modal basis, load dependent orthogonality
    # b_c: ['S-S'] a string specifying boundary conditions to be analysed:
    # 'S-S' simply-pimply supported boundary condition at loaded edges
    # 'C-C' clamped-clamped boundary condition at loaded edges
    # 'S-C' simply-clamped supported boundary condition at loaded edges
    # 'C-F' clamped-free supported boundary condition at loaded edges
    # 'C-G' clamped-guided supported boundary condition at loaded edges
    # m_all: m_all{length#}=[longitudinal_num# ... longitudinal_num#],
    #       longitudinal terms m for all the lengths in cell notation
    # each cell has a vector including the longitudinal terms for this length
    # n_eigs - the number of eigenvalues to be determined at length (default=10)

    # OUTPUTS
    # curve: buckling curve (load factor) for each length
    # curve{i} = [ length mode# 1
    #             length mode# 2
    #             ...    ...
    #             length mode#]
    # shapes = mode shapes for each length
    # shapes{i} = mode, mode is a matrix, each column corresponds to a mode.

    n_nodes = len(nodes)
    curve = []
    shapes = []
    signature = np.zeros((len(lengths), 1))

    # CLEAN UP INPUT
    # clean u_j 0's, multiple terms. or out-of-order terms in m_all
    m_all = pycufsm.analysis.m_sort(m_all)

    # DETERMINE FLAGS FOR USER CONSTRAINTS AND INTERNAL (AT NODE) B.C.'s
    bc_flag = pycufsm.analysis.constr_bc_flag(nodes=nodes, constraints=constraints)

    # GENERATE STRIP WIDTH AND DIRECTION ANGLE
    el_props = pycufsm.analysis.elem_prop(nodes=nodes, elements=elements)

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

        # ZERO OUT THE GLOBAL MATRICES
        k_global = np.zeros((4 * n_nodes * total_m, 4 * n_nodes * total_m))
        kg_global = np.zeros((4 * n_nodes * total_m, 4 * n_nodes * total_m))

        # ASSEMBLE THE GLOBAL STIFFNESS MATRICES
        for j, elem in enumerate(elements):
            # Generate element stiffness matrix (k_local) in local coordinates
            thick = elem[3]
            b_strip = el_props[j, 1]
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
            node_i = int(elem[1])
            node_j = int(elem[2])

            # Generate geometric stiffness matrix (kg_local) in local coordinates
            ty_1 = nodes[node_i][7] * thick
            ty_2 = nodes[node_j][7] * thick
            kg_l = pycufsm.analysis.kglocal(
                length=length, b_strip=b_strip, ty_1=ty_1, ty_2=ty_2, b_c=b_c, m_a=m_a
            )

            # Transform k_local and kg_local into global coordinates
            alpha = el_props[j, 2]
            [k_local, kg_local] = pycufsm.analysis.trans(
                alpha=alpha, k_local=k_l, kg_local=kg_l, m_a=m_a
            )

            # Add element contribution of k_local to full matrix k_global and kg_local to kg_global
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

        # %ADD SPRING CONTRIBUTIONS TO STIFFNESS
        # %Prior to version 4.3 this was the springs method
        #     %     if ~isempty(springs) %springs variable exists
        #     %         [k_global]=addspring(k_global,springs,n_nodes,length,b_c,m_a)
        #     %     end
        # %Now from version 4.3 this is the new springs method
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
                ks_l = pycufsm.analysis.spring_klocal(
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
                if node_j == 0 or spring[7] == 0:  # spring is to ground
                    # handle the spring to ground during assembly
                    alpha = 0  # use global coordinates for spring
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
                        alpha = np.arctan2(d_y, d_x)
                k_s = pycufsm.analysis.spring_trans(alpha=alpha, k_s=ks_l, m_a=m_a)
                # Add element contribution of k_s to full matrix k_global
                k_global = pycufsm.analysis.spring_assemble(
                    k_global=k_global,
                    k_local=k_s,
                    node_i=node_i,
                    node_j=node_j,
                    n_nodes=n_nodes,
                    m_a=m_a
                )

        # INTERNAL BOUNDARY CONDITIONS (ON THE NODES) AND USER DEFINED CONSTR.
        # Check for user defined constraints too
        if bc_flag == 0:
            # no user defined constraints and fixities.
            r_u0_matrix = 0
            nu0 = 0
        else:
            # size boundary conditions and user constraints for use in r_matrix format
            # d_constrained=r_user*d_unconstrained, d=nodal DOF vector (note by
            # BWS June 5 2006)
            r_user = pycufsm.cfsm.constr_user(nodes=nodes, constraints=constraints, m_a=m_a)
            r_u0_matrix = spla.null_space(r_user.conj().T)
            # Number of boundary conditions and user defined constraints = nu0
            nu0 = len(r_u0_matrix[0])

        # %GENERATION OF cFSM CONSTRAINT MATRIX
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
