import numpy as np

# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE, CPEng
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


def m_sort(m_all):
    for i, m_a in enumerate(m_all):
        # return all the nonzeros longitudinal terms in m_a as a column vector
        m_a = m_a[np.nonzero(m_a)]
        m_a = np.lib.arraysetops.unique(m_a)  # remove repetitive longitudinal terms
        m_all[i] = m_a
    return m_all


def constr_bc_flag(nodes, constraints):
    # this subroutine is to determine flags for user constraints and internal (at node) B.C.'s

    # inputs:
    # nodes: [node# x y dof_x dof_y dof_z dof_r stress] n_nodes x 8
    # constraints:: [node# e dof_e coeff node# k dof_k] e=dof to be eliminated
    #   k=kept dof dofe_node = coeff*dofk_nodek

    # Output:
    # bc_flag: 1 if there are user constraints or node fixities
    # 0 if there is no user constraints and node fixities

    # Z. Li, June 2010

    # Check for boundary conditions on the nodes
    n_nodes = len(nodes)
    for i in range(0, n_nodes):
        for j in range(3, 7):
            if nodes[i, j] == 0:
                return 1

    # Check for user defined constraints too
    if len(constraints) == 0:
        return 0
    else:
        return 1


def addspring(k_global, springs, n_nodes, length, b_c, m_a):
    # BWS
    # August 2000

    # [k_global]=addspring(k_global,springs,n_nodes,length,m_a,b_c)
    # Add spring stiffness to global elastic stiffness matrix

    # k_global is the complete elastic stiffness matrix
    # springs is the definition of any external springs added to the member
    # springs=[node# DOF(x=1,y=2,z=3,theta=4) k_s]

    # modified by Z. Li, Aug. 09, 2009 for general B.C.
    # Z. Li, June 2010

    if len(springs) > 0:
        total_m = len(m_a)  # Total number of longitudinal terms m

        for spring in springs:
            node = spring[0]
            dof = spring[1]
            k_stiff = spring[2]
            k_flag = spring[3]

            if dof == 1:
                r_c = 2*node - 1
            elif dof == 2:
                r_c = 2*n_nodes + 2*node - 1
            elif dof == 3:
                r_c = 2 * node
            elif dof == 4:
                r_c = 2*n_nodes + 2*node
            else:
                r_c = 1
                k_s = 0

            for i in range(0, total_m):
                for j in range(0, total_m):
                    if k_flag == 0:
                        k_s = k_stiff  # k_stiff is the total stiffness and may be added directly
                    else:
                        if dof == 3:
                            k_s = 0  # axial dof with a foundation stiffness has no net stiffness
                        else:
                            [i_1, _, _, _, _] = bc_i1_5(b_c, m_a[i], m_a[j], length)
                            k_s = k_stiff * i_1
                            # k_stiff is a foundation stiffness and an equivalent
                            # total stiffness must be calculated

                    k_global[4*n_nodes*i+r_c-1, 4*n_nodes*j+r_c-1] \
                        = k_global[4*n_nodes*i+r_c-1, 4*n_nodes*j+r_c-1] + k_s

    return k_global


def elem_prop(nodes, elements):
    el_props = np.zeros((len(elements), 3))
    for i, elem in enumerate(elements):
        node_i = int(elem[1])
        node_j = int(elem[2])
        x_i = nodes[node_i, 1]
        y_i = nodes[node_i, 2]
        x_j = nodes[node_j, 1]
        y_j = nodes[node_j, 2]
        d_x = x_j - x_i
        d_y = y_j - y_i
        width = np.sqrt(d_x**2 + d_y**2)
        alpha = np.arctan2(d_y, d_x)
        el_props[i] = [i, width, alpha]
    return el_props


def klocal(stiff_x, stiff_y, nu_x, nu_y, bulk, thick, length, b_strip, b_c, m_a):
    # Generate element stiffness matrix (k_local) in local coordinates

    # Inputs:
    # stiff_x,stiff_y,nu_x,nu_y,bulk: material properties
    # thick: thickness of the strip (element)
    # length: length of the strip in longitudinal direction
    # b_strip: width of the strip in transverse direction
    # b_c: ['S-S'] a string specifying boundary conditions to be analyzed:
    # 'S-S' simply-pimply supported boundary condition at loaded edges
    # 'C-C' clamped-clamped boundary condition at loaded edges
    # 'S-C' simply-clamped supported boundary condition at loaded edges
    # 'C-F' clamped-free supported boundary condition at loaded edges
    # 'C-G' clamped-guided supported boundary condition at loaded edges
    # m_a: longitudinal terms (or half-wave numbers) for this length

    # Output:
    # k_local: local stiffness matrix, a total_m x total_m matrix of 8 by 8 submatrices.
    # k_local=[k_mp]total_m x total_m block matrix
    # each k_mp is the 8 x 8 submatrix in the DOF order [u1 v1 u2 v2 w1 theta1 w2 theta2]'

    # Z. Li June 2008
    # modified by Z. Li, Aug. 09, 2009
    # modified by Z. Li, June 2010

    e_1 = stiff_x / (1 - nu_x*nu_y)
    e_2 = stiff_y / (1 - nu_x*nu_y)
    d_x = stiff_x * thick**3 / (12 * (1 - nu_x*nu_y))
    d_y = stiff_y * thick**3 / (12 * (1 - nu_x*nu_y))
    d_1 = nu_x * stiff_y * thick**3 / (12 * (1 - nu_x*nu_y))
    d_xy = bulk * thick**3 / 12

    total_m = len(m_a)  # Total number of longitudinal terms m

    k_local = np.zeros((8 * total_m, 8 * total_m))
    zero_matrix = np.zeros((4, 4))
    for i in range(0, total_m):
        for j in range(0, total_m):
            km_mp = np.zeros((4, 4))
            kf_mp = np.zeros((4, 4))
            u_i = m_a[i] * np.pi
            u_j = m_a[j] * np.pi
            c_1 = u_i / length
            c_2 = u_j / length

            [i_1, i_2, i_3, i_4, i_5] = bc_i1_5(b_c, m_a[i], m_a[j], length)

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

            # assemble the matrix of Kf_mp
            kf_mp[0, 0] = (5040*d_x*i_1 - 504*b_strip**2*d_1*i_2 - 504*b_strip**2*d_1*i_3 \
                + 156*b_strip**4*d_y*i_4 + 2016*b_strip**2*d_xy*i_5)/420/b_strip**3
            kf_mp[0, 1] = (2520*b_strip*d_x*i_1 - 462*b_strip**3*d_1*i_2 - 42*b_strip**3*d_1*i_3 \
                + 22*b_strip**5*d_y*i_4 + 168*b_strip**3*d_xy*i_5)/420/b_strip**3
            kf_mp[0, 2] = (-5040*d_x*i_1 + 504*b_strip**2*d_1*i_2 + 504*b_strip**2*d_1*i_3 \
                + 54*b_strip**4*d_y*i_4 - 2016*b_strip**2*d_xy*i_5)/420/b_strip**3
            kf_mp[0, 3] = (2520*b_strip*d_x*i_1 - 42*b_strip**3*d_1*i_2 - 42*b_strip**3*d_1*i_3 \
                - 13*b_strip**5*d_y*i_4 + 168*b_strip**3*d_xy*i_5)/420/b_strip**3

            kf_mp[1, 0] = (2520*b_strip*d_x*i_1 - 462*b_strip**3*d_1*i_3 - 42*b_strip**3*d_1*i_2 \
                + 22*b_strip**5*d_y*i_4 + 168*b_strip**3*d_xy*i_5)/420/b_strip**3
            kf_mp[1, 1] = (1680*b_strip**2*d_x*i_1 - 56*b_strip**4*d_1*i_2 - 56*b_strip**4*d_1*i_3 \
                + 4*b_strip**6*d_y*i_4 + 224*b_strip**4*d_xy*i_5)/420/b_strip**3
            kf_mp[1, 2] = (-2520*b_strip*d_x*i_1 + 42*b_strip**3*d_1*i_2 + 42*b_strip**3*d_1*i_3 \
                + 13*b_strip**5*d_y*i_4 - 168*b_strip**3*d_xy*i_5)/420/b_strip**3
            kf_mp[1, 3] = (840*b_strip**2*d_x*i_1 + 14*b_strip**4*d_1*i_2 + 14*b_strip**4*d_1*i_3 \
                - 3*b_strip**6*d_y*i_4 - 56*b_strip**4*d_xy*i_5)/420/b_strip**3

            kf_mp[2, 0] = kf_mp[0, 2]
            kf_mp[2, 1] = kf_mp[1, 2]
            kf_mp[2, 2] = (5040*d_x*i_1 - 504*b_strip**2*d_1*i_2 - 504*b_strip**2*d_1*i_3 \
                + 156*b_strip**4*d_y*i_4 + 2016*b_strip**2*d_xy*i_5)/420/b_strip**3
            kf_mp[2, 3] = (-2520*b_strip*d_x*i_1 + 462*b_strip**3*d_1*i_2 + 42*b_strip**3*d_1*i_3 \
                - 22*b_strip**5*d_y*i_4 - 168*b_strip**3*d_xy*i_5)/420/b_strip**3

            kf_mp[3, 0] = kf_mp[0, 3]
            kf_mp[3, 1] = kf_mp[1, 3]
            kf_mp[3, 2] = (-2520*b_strip*d_x*i_1 + 462*b_strip**3*d_1*i_3 + 42*b_strip**3*d_1*i_2 \
                - 22*b_strip**5*d_y*i_4 - 168*b_strip**3*d_xy*i_5)/420/b_strip**3 # not symmetric
            kf_mp[3, 3] = (1680*b_strip**2*d_x*i_1 - 56*b_strip**4*d_1*i_2 - 56*b_strip**4*d_1*i_3 \
                + 4*b_strip**6*d_y*i_4 + 224*b_strip**4*d_xy*i_5)/420/b_strip**3

            # assemble the membrane and flexural stiffness matrices
            k_mp = np.concatenate((
                np.concatenate((km_mp, zero_matrix),
                               axis=1), np.concatenate((zero_matrix, kf_mp), axis=1)
            ))
            # add it into local element stiffness matrix by corresponding to i
            k_local[8 * i:8 * (i+1), 8 * j:8 * (j+1)] = k_mp
    return k_local


def bc_i1_5(b_c, m_i, m_j, length):
    # Calculate the 5 undetermined parameters i_1,i_2,i_3,i_4,i_5 for local elastic
    # and geometric stiffness matrices.
    # b_c: a string specifying boundary conditions to be analysed:
    #'S-S' simply-pimply supported boundary condition at loaded edges
    #'C-C' clamped-clamped boundary condition at loaded edges
    #'S-C' simply-clamped supported boundary condition at loaded edges
    #'C-F' clamped-free supported boundary condition at loaded edges
    #'C-G' clamped-guided supported boundary condition at loaded edges
    # Outputs:
    # i_1,i_2,i_3,i_4,i_5
    # calculation of i_1 is the integration of y_m*Yn from 0 to length
    # calculation of i_2 is the integration of y_m''*Yn from 0 to length
    # calculation of i_3 is the integration of y_m*Yn'' from 0 to length
    # calculation of i_3 is the integration of y_m*Yn'' from 0 to length
    # calculation of i_4 is the integration of y_m''*Yn'' from 0 to length
    # calculation of i_5 is the integration of y_m'*Yn' from 0 to length

    if b_c == 'S-S':
        # For simply-pimply supported boundary condition at loaded edges
        if m_i == m_j:
            i_1 = length / 2
            i_2 = -m_i**2 * np.pi**2 / length / 2
            i_3 = -m_j**2 * np.pi**2 / length / 2
            i_4 = np.pi**4 * m_i**4 / 2 / length**3
            i_5 = np.pi**2 * m_i**2 / 2 / length
        else:
            i_1 = 0
            i_2 = 0
            i_3 = 0
            i_4 = 0
            i_5 = 0

    elif b_c == 'C-C':
        # For Clamped-clamped boundary condition at loaded edges
        # calculation of i_1 is the integration of y_m*Yn from 0 to length
        if m_i == m_j:
            if m_i == 1:
                i_1 = 3 * length / 8
            else:
                i_1 = length / 4
            i_2 = -(m_i**2 + 1) * np.pi**2 / 4 / length
            i_3 = -(m_j**2 + 1) * np.pi**2 / 4 / length
            i_4 = np.pi**4 * ((m_i**2 + 1)**2 + 4 * m_i**2) / 4 / length**3
            i_5 = (1 + m_i**2) * np.pi**2 / 4 / length
        else:
            if m_i - m_j == 2:
                i_1 = -length / 8
                i_2 = (m_i**2 + 1) * np.pi**2 / 8 / length - m_i * np.pi**2 / 4 / length
                i_3 = (m_j**2 + 1) * np.pi**2 / 8 / length + m_j * np.pi**2 / 4 / length
                i_4 = -(m_i - 1)**2 * (m_j + 1)**2 * np.pi**4 / 8 / length**3
                i_5 = -(1 + m_i*m_j) * np.pi**2 / 8 / length
            elif m_i - m_j == -2:
                i_1 = -length / 8
                i_2 = (m_i**2 + 1) * np.pi**2 / 8 / length + m_i * np.pi**2 / 4 / length
                i_3 = (m_j**2 + 1) * np.pi**2 / 8 / length - m_j * np.pi**2 / 4 / length
                i_4 = -(m_i + 1)**2 * (m_j - 1)**2 * np.pi**4 / 8 / length**3
                i_5 = -(1 + m_i*m_j) * np.pi**2 / 8 / length
            else:
                i_1 = 0
                i_2 = 0
                i_3 = 0
                i_4 = 0
                i_5 = 0

    elif b_c == 'S-C' or b_c == 'C-S':
        # For simply-clamped supported boundary condition at loaded edges
        # calculation of i_1 is the integration of y_m*Yn from 0 to length
        if m_i == m_j:
            i_1 = (1 + (m_i + 1)**2 / m_i**2) * length / 2
            i_2 = -(m_i + 1)**2 * np.pi**2 / length
            i_3 = -(m_i + 1)**2 * np.pi**2 / length
            i_4 = (m_i + 1)**2 * np.pi**4 * ((m_i + 1)**2 + m_i**2) / 2 / length**3
            i_5 = (1 + m_i)**2 * np.pi**2 / length
        else:
            if m_i - m_j == 1:
                i_1 = (m_i+1) * length / 2 / m_i
                i_2 = -(m_i + 1) * m_i * np.pi**2 / 2 / length
                i_3 = -(m_j + 1)**2 * np.pi**2 * (m_i+1) / 2 / length / m_i
                i_4 = (m_i+1) * m_i * (m_j + 1)**2 * np.pi**4 / 2 / length**3
                i_5 = (1+m_i) * (1+m_j) * np.pi**2 / 2 / length
            elif m_i - m_j == -1:
                i_1 = (m_j+1) * length / 2 / m_j
                i_2 = -(m_i + 1)**2 * np.pi**2 * (m_j+1) / 2 / length / m_j
                i_3 = -(m_j + 1) * m_j * np.pi**2 / 2 / length
                i_4 = (m_i + 1)**2 * m_j * (m_j+1) * np.pi**4 / 2 / length**3
                i_5 = (1+m_i) * (1+m_j) * np.pi**2 / 2 / length
            else:
                i_1 = 0
                i_2 = 0
                i_3 = 0
                i_4 = 0
                i_5 = 0

    elif b_c == 'C-F' or b_c == 'F-C':
        # For clamped-free supported boundary condition at loaded edges
        # calculation of i_1 is the integration of y_m*Yn from 0 to length
        if m_i == m_j:
            i_1 = 3*length/2 - 2 * length * (-1)**(m_i - 1) / (m_i - 1/2) / np.pi
            i_2 = (m_i - 1/2)**2 * np.pi**2 * ((-1)**(m_i - 1) / (m_i - 1/2) / np.pi - 1/2) / length
            i_3 = (m_j - 1/2)**2 * np.pi**2 * ((-1)**(m_j - 1) / (m_j - 1/2) / np.pi - 1/2) / length
            i_4 = (m_i - 1/2)**4 * np.pi**4 / 2 / length**3
            i_5 = (m_i - 1/2)**2 * np.pi**2 / 2 / length
        else:
            i_1 = length - length*(-1)**(m_i - 1) / (m_i - 1/2) / np.pi \
                - length*(-1)**(m_j - 1) / (m_j - 1/2) / np.pi
            i_2 = (m_i - 1/2)**2 * np.pi**2 * ((-1)**(m_i - 1) / (m_i - 1/2) / np.pi) / length
            i_3 = (m_j - 1/2)**2 * np.pi**2 * ((-1)**(m_j - 1) / (m_j - 1/2) / np.pi) / length
            i_4 = 0
            i_5 = 0

    elif b_c == 'C-G' or b_c == 'G-C':
        # For clamped-guided supported boundary condition at loaded edges
        # calculation of i_1 is the integration of y_m*Yn from 0 to length
        if m_i == m_j:
            if m_i == 1:
                i_1 = 3 * length / 8
            else:
                i_1 = length / 4
            i_2 = -((m_i - 1/2)**2 + 1/4) * np.pi**2 / length / 4
            i_3 = -((m_i - 1/2)**2 + 1/4) * np.pi**2 / length / 4
            i_4 = ((m_i - 1/2)**2
                   + 1/4)**2 * np.pi**4 / 4 / length**3 + (m_i - 1/2)**2 * np.pi**4 / 4 / length**3
            i_5 = (m_i - 1/2)**2 * np.pi**2 / length / 4 + np.pi**2 / 16 / length
        else:
            if m_i - m_j == 1:
                i_1 = -length / 8
                i_2 = ((m_i - 1/2)**2
                       + 1/4) * np.pi**2 / length / 8 - (m_i - 1/2) * np.pi**2 / length / 8
                i_3 = ((m_j - 1/2)**2
                       + 1/4) * np.pi**2 / length / 8 + (m_j - 1/2) * np.pi**2 / length / 8
                i_4 = -m_j**4 * np.pi**4 / 8 / length**3
                i_5 = -m_j**2 * np.pi**2 / 8 / length
            elif m_i - m_j == -1:
                i_1 = -length / 8
                i_2 = ((m_i - 1/2)**2
                       + 1/4) * np.pi**2 / length / 8 + (m_i - 1/2) * np.pi**2 / length / 8
                i_3 = ((m_j - 1/2)**2
                       + 1/4) * np.pi**2 / length / 8 - (m_j - 1/2) * np.pi**2 / length / 8
                i_4 = -m_i**4 * np.pi**4 / 8 / length**3
                i_5 = -m_i**2 * np.pi**2 / 8 / length
            else:
                i_1 = 0
                i_2 = 0
                i_3 = 0
                i_4 = 0
                i_5 = 0
    return [i_1, i_2, i_3, i_4, i_5]


def kglocal(length, b_strip, ty_1, ty_2, b_c, m_a):
    # Generate geometric stiffness matrix (kg_local) in local coordinates

    # Inputs:
    # length: length of the strip in longitudinal direction
    # b_strip: width of the strip in transverse direction
    # ty_1, ty_2: node stresses
    # b_c: a string specifying boundary conditions to be analysed:
    #'S-S' simply-pimply supported boundary condition at loaded edges
    #'C-C' clamped-clamped boundary condition at loaded edges
    #'S-C' simply-clamped supported boundary condition at loaded edges
    #'C-F' clamped-free supported boundary condition at loaded edges
    #'C-G' clamped-guided supported boundary condition at loaded edges
    # m_a: longitudinal terms (or half-wave numbers) for this length

    # Output:
    # kg_local: local geometric stiffness matrix, a total_m x total_m matrix of 8 by 8 submatrices.
    # kg_local=[kg_mp]total_m x total_m block matrix
    # each kg_mp is the 8 x 8 submatrix in the DOF order [u1 v1 u2 v2 w1 theta1
    # w2 theta2]'

    # Z. Li, June 2008
    # modified by Z. Li, Aug. 09, 2009
    # modified by Z. Li, June 2010

    total_m = len(m_a)  # Total number of longitudinal terms m
    kg_local = np.zeros((8 * total_m, 8 * total_m))

    for i in range(0, total_m):
        for j in range(0, total_m):
            gm_mp = np.zeros((4, 4))
            zero_matrix = np.zeros((4, 4))
            gf_mp = np.zeros((4, 4))
            u_i = m_a[i] * np.pi
            u_j = m_a[j] * np.pi

            [_, _, _, i_4, i_5] = bc_i1_5(b_c, m_a[i], m_a[j], length)

            # assemble the matrix of gm_mp (symmetric membrane stability matrix)
            gm_mp[0, 0] = b_strip * (3*ty_1 + ty_2) * i_5 / 12
            gm_mp[0, 2] = b_strip * (ty_1+ty_2) * i_5 / 12
            gm_mp[2, 0] = gm_mp[0, 2]
            gm_mp[1, 1] = b_strip * length**2 * (3*ty_1 + ty_2) * i_4 / 12 / u_i / u_j
            gm_mp[1, 3] = b_strip * length**2 * (ty_1+ty_2) * i_4 / 12 / u_i / u_j
            gm_mp[3, 1] = gm_mp[1, 3]
            gm_mp[2, 2] = b_strip * (ty_1 + 3*ty_2) * i_5 / 12
            gm_mp[3, 3] = b_strip * length**2 * (ty_1 + 3*ty_2) * i_4 / 12 / u_i / u_j

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
            kg_mp = np.concatenate((
                np.concatenate((gm_mp, zero_matrix),
                               axis=1), np.concatenate((zero_matrix, gf_mp), axis=1)
            ))

            # add it into local geometric stiffness matrix by corresponding to i
            kg_local[8 * i:8 * (i+1), 8 * j:8 * (j+1)] = kg_mp
    return kg_local


def trans(alpha, k_local, kg_local, m_a):
    # Transfer the local stiffness into global stiffness
    # Zhanjie 2008
    # modified by Z. Li, Aug. 09, 2009

    total_m = len(m_a)  # Total number of longitudinal terms m
    gamma = np.zeros((8 * total_m, 8 * total_m))

    gam = np.array([[np.cos(alpha), 0, 0, 0, -np.sin(alpha), 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, np.cos(alpha), 0, 0, 0, -np.sin(alpha), 0], [0, 0, 0, 1, 0, 0, 0, 0],
                    [np.sin(alpha), 0, 0, 0, np.cos(alpha), 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, np.sin(alpha), 0, 0, 0, np.cos(alpha), 0], [0, 0, 0, 0, 0, 0, 0, 1]])

    # extend to multi-m
    for i in range(0, total_m):
        gamma[8 * i:8 * (i+1), 8 * i:8 * (i+1)] = gam

    kglobal = gamma @ k_local @ gamma.conj().T
    kgglobal = gamma @ kg_local @ gamma.conj().T

    return [kglobal, kgglobal]


def assemble(k_global, kg_global, k_local, kg_local, node_i, node_j, n_nodes, m_a):
    # Add the element contribution to the global stiffness matrix

    # Outputs:
    # k_global: global elastic stiffness matrix
    # kg_global: global geometric stiffness matrix
    # k_global and kg_global: total_m x total_m submatrices. Each submatrix is similar to the
    # one used in original CUFSM for single longitudinal term m in the DOF order
    #[u1 v1...un vn w1 01...wn 0n]m'.

    # Z. Li, June 2008
    # modified by Z. Li, Aug. 09, 2009
    # Z. Li, June 2010

    total_m = len(m_a)  # Total number of longitudinal terms m
    k_2_matrix = np.zeros((4 * n_nodes * total_m, 4 * n_nodes * total_m))
    k_3_matrix = np.zeros((4 * n_nodes * total_m, 4 * n_nodes * total_m))
    skip = 2 * n_nodes
    for i in range(0, total_m):
        for j in range(0, total_m):
            # Submatrices for the initial stiffness
            k11 = k_local[8 * i:8*i + 2, 8 * j:8*j + 2]
            k12 = k_local[8 * i:8*i + 2, 8*j + 2:8*j + 4]
            k13 = k_local[8 * i:8*i + 2, 8*j + 4:8*j + 6]
            k14 = k_local[8 * i:8*i + 2, 8*j + 6:8*j + 8]
            k21 = k_local[8*i + 2:8*i + 4, 8 * j:8*j + 2]
            k22 = k_local[8*i + 2:8*i + 4, 8*j + 2:8*j + 4]
            k23 = k_local[8*i + 2:8*i + 4, 8*j + 4:8*j + 6]
            k24 = k_local[8*i + 2:8*i + 4, 8*j + 6:8*j + 8]
            k31 = k_local[8*i + 4:8*i + 6, 8 * j:8*j + 2]
            k32 = k_local[8*i + 4:8*i + 6, 8*j + 2:8*j + 4]
            k33 = k_local[8*i + 4:8*i + 6, 8*j + 4:8*j + 6]
            k34 = k_local[8*i + 4:8*i + 6, 8*j + 6:8*j + 8]
            k41 = k_local[8*i + 6:8*i + 8, 8 * j:8*j + 2]
            k42 = k_local[8*i + 6:8*i + 8, 8*j + 2:8*j + 4]
            k43 = k_local[8*i + 6:8*i + 8, 8*j + 4:8*j + 6]
            k44 = k_local[8*i + 6:8*i + 8, 8*j + 6:8*j + 8]

            k_2_matrix[4*n_nodes*i+(node_i+1)*2-2:4*n_nodes*i+(node_i+1)*2, \
                4*n_nodes*j+(node_i+1)*2-2:4*n_nodes*j+(node_i+1)*2] = k11
            k_2_matrix[4*n_nodes*i+(node_i+1)*2-2:4*n_nodes*i+(node_i+1)*2, \
                4*n_nodes*j+(node_j+1)*2-2:4*n_nodes*j+(node_j+1)*2] = k12
            k_2_matrix[4*n_nodes*i+(node_j+1)*2-2:4*n_nodes*i+(node_j+1)*2, \
                4*n_nodes*j+(node_i+1)*2-2:4*n_nodes*j+(node_i+1)*2] = k21
            k_2_matrix[4*n_nodes*i+(node_j+1)*2-2:4*n_nodes*i+(node_j+1)*2, \
                4*n_nodes*j+(node_j+1)*2-2:4*n_nodes*j+(node_j+1)*2] = k22

            k_2_matrix[4*n_nodes*i+skip+(node_i+1)*2-2:4*n_nodes*i+skip+(node_i+1)*2, \
                4*n_nodes*j+skip+(node_i+1)*2-2:4*n_nodes*j+skip+(node_i+1)*2] = k33
            k_2_matrix[4*n_nodes*i+skip+(node_i+1)*2-2:4*n_nodes*i+skip+(node_i+1)*2, \
                4*n_nodes*j+skip+(node_j+1)*2-2:4*n_nodes*j+skip+(node_j+1)*2] = k34
            k_2_matrix[4*n_nodes*i+skip+(node_j+1)*2-2:4*n_nodes*i+skip+(node_j+1)*2, \
                4*n_nodes*j+skip+(node_i+1)*2-2:4*n_nodes*j+skip+(node_i+1)*2] = k43
            k_2_matrix[4*n_nodes*i+skip+(node_j+1)*2-2:4*n_nodes*i+skip+(node_j+1)*2, \
                4*n_nodes*j+skip+(node_j+1)*2-2:4*n_nodes*j+skip+(node_j+1)*2] = k44

            k_2_matrix[4*n_nodes*i+(node_i+1)*2-2:4*n_nodes*i+(node_i+1)*2, \
                4*n_nodes*j+skip+(node_i+1)*2-2:4*n_nodes*j+skip+(node_i+1)*2] = k13
            k_2_matrix[4*n_nodes*i+(node_i+1)*2-2:4*n_nodes*i+(node_i+1)*2, \
                4*n_nodes*j+skip+(node_j+1)*2-2:4*n_nodes*j+skip+(node_j+1)*2] = k14
            k_2_matrix[4*n_nodes*i+(node_j+1)*2-2:4*n_nodes*i+(node_j+1)*2, \
                4*n_nodes*j+skip+(node_i+1)*2-2:4*n_nodes*j+skip+(node_i+1)*2] = k23
            k_2_matrix[4*n_nodes*i+(node_j+1)*2-2:4*n_nodes*i+(node_j+1)*2, \
                4*n_nodes*j+skip+(node_j+1)*2-2:4*n_nodes*j+skip+(node_j+1)*2] = k24

            k_2_matrix[4*n_nodes*i+skip+(node_i+1)*2-2:4*n_nodes*i+skip+(node_i+1)*2, \
                4*n_nodes*j+(node_i+1)*2-2:4*n_nodes*j+(node_i+1)*2] = k31
            k_2_matrix[4*n_nodes*i+skip+(node_i+1)*2-2:4*n_nodes*i+skip+(node_i+1)*2, \
                4*n_nodes*j+(node_j+1)*2-2:4*n_nodes*j+(node_j+1)*2] = k32
            k_2_matrix[4*n_nodes*i+skip+(node_j+1)*2-2:4*n_nodes*i+skip+(node_j+1)*2, \
                4*n_nodes*j+(node_i+1)*2-2:4*n_nodes*j+(node_i+1)*2] = k41
            k_2_matrix[4*n_nodes*i+skip+(node_j+1)*2-2:4*n_nodes*i+skip+(node_j+1)*2, \
                4*n_nodes*j+(node_j+1)*2-2:4*n_nodes*j+(node_j+1)*2] = k42

            # Submatrices for the initial stiffness
            kg11 = kg_local[8 * i:8*i + 2, 8 * j:8*j + 2]
            kg12 = kg_local[8 * i:8*i + 2, 8*j + 2:8*j + 4]
            kg13 = kg_local[8 * i:8*i + 2, 8*j + 4:8*j + 6]
            kg14 = kg_local[8 * i:8*i + 2, 8*j + 6:8*j + 8]
            kg21 = kg_local[8*i + 2:8*i + 4, 8 * j:8*j + 2]
            kg22 = kg_local[8*i + 2:8*i + 4, 8*j + 2:8*j + 4]
            kg23 = kg_local[8*i + 2:8*i + 4, 8*j + 4:8*j + 6]
            kg24 = kg_local[8*i + 2:8*i + 4, 8*j + 6:8*j + 8]
            kg31 = kg_local[8*i + 4:8*i + 6, 8 * j:8*j + 2]
            kg32 = kg_local[8*i + 4:8*i + 6, 8*j + 2:8*j + 4]
            kg33 = kg_local[8*i + 4:8*i + 6, 8*j + 4:8*j + 6]
            kg34 = kg_local[8*i + 4:8*i + 6, 8*j + 6:8*j + 8]
            kg41 = kg_local[8*i + 6:8*i + 8, 8 * j:8*j + 2]
            kg42 = kg_local[8*i + 6:8*i + 8, 8*j + 2:8*j + 4]
            kg43 = kg_local[8*i + 6:8*i + 8, 8*j + 4:8*j + 6]
            kg44 = kg_local[8*i + 6:8*i + 8, 8*j + 6:8*j + 8]

            k_3_matrix[4*n_nodes*i + (node_i+1) * 2 - 2:4*n_nodes*i + (node_i+1) * 2,
                       4*n_nodes*j + (node_i+1) * 2 - 2:4*n_nodes*j + (node_i+1) * 2] = kg11
            k_3_matrix[4*n_nodes*i + (node_i+1) * 2 - 2:4*n_nodes*i + (node_i+1) * 2,
                       4*n_nodes*j + (node_j+1) * 2 - 2:4*n_nodes*j + (node_j+1) * 2] = kg12
            k_3_matrix[4*n_nodes*i + (node_j+1) * 2 - 2:4*n_nodes*i + (node_j+1) * 2,
                       4*n_nodes*j + (node_i+1) * 2 - 2:4*n_nodes*j + (node_i+1) * 2] = kg21
            k_3_matrix[4*n_nodes*i + (node_j+1) * 2 - 2:4*n_nodes*i + (node_j+1) * 2,
                       4*n_nodes*j + (node_j+1) * 2 - 2:4*n_nodes*j + (node_j+1) * 2] = kg22

            k_3_matrix[4*n_nodes*i + skip + (node_i+1) * 2 - 2:4*n_nodes*i + skip + (node_i+1) * 2,
                       4*n_nodes*j + skip + (node_i+1) * 2 - 2:4*n_nodes*j + skip
                       + (node_i+1) * 2] = kg33
            k_3_matrix[4*n_nodes*i + skip + (node_i+1) * 2 - 2:4*n_nodes*i + skip + (node_i+1) * 2,
                       4*n_nodes*j + skip + (node_j+1) * 2 - 2:4*n_nodes*j + skip
                       + (node_j+1) * 2] = kg34
            k_3_matrix[4*n_nodes*i + skip + (node_j+1) * 2 - 2:4*n_nodes*i + skip + (node_j+1) * 2,
                       4*n_nodes*j + skip + (node_i+1) * 2 - 2:4*n_nodes*j + skip
                       + (node_i+1) * 2] = kg43
            k_3_matrix[4*n_nodes*i + skip + (node_j+1) * 2 - 2:4*n_nodes*i + skip + (node_j+1) * 2,
                       4*n_nodes*j + skip + (node_j+1) * 2 - 2:4*n_nodes*j + skip
                       + (node_j+1) * 2] = kg44

            k_3_matrix[4*n_nodes*i + (node_i+1) * 2 - 2:4*n_nodes*i + (node_i+1) * 2, 4*n_nodes*j
                       + skip + (node_i+1) * 2 - 2:4*n_nodes*j + skip + (node_i+1) * 2] = kg13
            k_3_matrix[4*n_nodes*i + (node_i+1) * 2 - 2:4*n_nodes*i + (node_i+1) * 2, 4*n_nodes*j
                       + skip + (node_j+1) * 2 - 2:4*n_nodes*j + skip + (node_j+1) * 2] = kg14
            k_3_matrix[4*n_nodes*i + (node_j+1) * 2 - 2:4*n_nodes*i + (node_j+1) * 2, 4*n_nodes*j
                       + skip + (node_i+1) * 2 - 2:4*n_nodes*j + skip + (node_i+1) * 2] = kg23
            k_3_matrix[4*n_nodes*i + (node_j+1) * 2 - 2:4*n_nodes*i + (node_j+1) * 2, 4*n_nodes*j
                       + skip + (node_j+1) * 2 - 2:4*n_nodes*j + skip + (node_j+1) * 2] = kg24

            k_3_matrix[4*n_nodes*i + skip + (node_i+1) * 2 - 2:4*n_nodes*i + skip + (node_i+1) * 2,
                       4*n_nodes*j + (node_i+1) * 2 - 2:4*n_nodes*j + (node_i+1) * 2] = kg31
            k_3_matrix[4*n_nodes*i + skip + (node_i+1) * 2 - 2:4*n_nodes*i + skip + (node_i+1) * 2,
                       4*n_nodes*j + (node_j+1) * 2 - 2:4*n_nodes*j + (node_j+1) * 2] = kg32
            k_3_matrix[4*n_nodes*i + skip + (node_j+1) * 2 - 2:4*n_nodes*i + skip + (node_j+1) * 2,
                       4*n_nodes*j + (node_i+1) * 2 - 2:4*n_nodes*j + (node_i+1) * 2] = kg41
            k_3_matrix[4*n_nodes*i + skip + (node_j+1) * 2 - 2:4*n_nodes*i + skip + (node_j+1) * 2,
                       4*n_nodes*j + (node_j+1) * 2 - 2:4*n_nodes*j + (node_j+1) * 2] = kg42

    k_global = k_global + k_2_matrix
    kg_global = kg_global + k_3_matrix

    return [k_global, kg_global]


def spring_klocal(k_u, k_v, k_w, k_q, length, b_c, m_a, discrete, y_s):
    # Generate spring stiffness matrix (k_local) in local coordinates, modified from
    # klocal
    # BWS DEC 2015

    # Inputs:
    # k_u,k_v,k_w,k_q spring stiffness values
    # length: length of the strip in longitudinal direction
    # b_c: ['S-S'] a string specifying boundary conditions to be analyzed:
    #'S-S' simply-pimply supported boundary condition at loaded edges
    #'C-C' clamped-clamped boundary condition at loaded edges
    #'S-C' simply-clamped supported boundary condition at loaded edges
    #'C-F' clamped-free supported boundary condition at loaded edges
    #'C-G' clamped-guided supported boundary condition at loaded edges
    # m_a: longitudinal terms (or half-wave numbers) for this length
    # discrete == 1 if discrete spring
    # y_s = location of discrete spring

    # Output:
    # k_local: local stiffness matrix, a total_m x total_m matrix of 8 by 8 submatrices.
    # k_local=[k_mp]total_m x total_m block matrix
    # each k_mp is the 8 x 8 submatrix in the DOF order [u1 v1 u2 v2 w1 theta1 w2 theta2]'

    total_m = len(m_a)  # Total number of longitudinal terms m
    k_local = np.zeros(8 * total_m, 8 * total_m)
    z_0 = np.zeros(4, 4)
    for i in range(0, total_m):
        for j in range(0, total_m):
            km_mp = np.zeros(4, 4)
            kf_mp = np.zeros(4, 4)
            u_i = m_a[i] * np.pi
            u_j = m_a[j] * np.pi

            if discrete:
                [i_1, i_5] = bc_i1_5_atpoint(
                    b_c=b_c, m_i=m_a[i], m_j=m_a[j], length=length, y_s=y_s
                )
            else:  # foundation spring
                [i_1, _, _, _, i_5] = bc_i1_5(b_c=b_c, m_i=m_a[i], m_j=m_a[j], length=length)
            # assemble the matrix of km_mp
            km_mp = np.array(
                [[k_u * i_1, 0, -k_u * i_1, 0],
                 [0, k_v * i_5 * length**2 / (u_i*u_j), 0, -k_v * i_5 * length**2 / (u_i*u_j)],
                 [-k_u * i_1, 0, k_u * i_1, 0],
                 [0, -k_v * i_5 * length**2 / (u_i*u_j), 0, k_v * i_5 * length**2 / (u_i*u_j)]]
            )
            # assemble the matrix of kf_mp
            kf_mp = np.array([[k_w * i_1, 0, -k_w * i_1, 0], [0, k_q * i_1, 0, -k_q * i_1],
                              [-k_w * i_1, 0, k_w * i_1, 0], [0, -k_q * i_1, 0, k_q * i_1]])
            # assemble the membrane and flexural stiffness matrices
            k_mp = np.concatenate(
                (np.concatenate((km_mp, z_0), axis=1), np.concatenate((z_0, kf_mp), axis=1))
            )

            # add it into local element stiffness matrix by corresponding to m
            k_local[8 * i:8 * (i+1), 8 * j:8 * (j+1)] = k_mp

    return k_local


def bc_i1_5_atpoint(b_c, m_i, m_j, length, y_s):
    # Calculate the value of the longitudinal shape functions for discrete springs

    # b_c: a string specifying boundary conditions to be analyzed:
    #'S-S' simply-pimply supported boundary condition at loaded edges
    #'C-C' clamped-clamped boundary condition at loaded edges
    #'S-C' simply-clamped supported boundary condition at loaded edges
    #'C-F' clamped-free supported boundary condition at loaded edges
    #'C-G' clamped-guided supported boundary condition at loaded edges
    # Outputs:
    # i_1,i_5
    # calculation of i_1 is the value of y_m(y/L)*Yn(y/L)
    # calculation of i_5 is the value of y_m'(y/L)*Yn'(y/L)

    y_i = ym_at_ys(b_c=b_c, m_i=m_i, y_s=y_s, length=length)
    y_j = ym_at_ys(b_c=b_c, m_i=m_j, y_s=y_s, length=length)
    y_i_prime = ymprime_at_ys(b_c=b_c, m_i=m_i, y_s=y_s, length=length)
    y_j_prime = ymprime_at_ys(b_c=b_c, m_i=m_j, y_s=y_s, length=length)
    i_1 = y_i * y_j
    i_5 = y_i_prime * y_j_prime
    return i_1, i_5


def spring_trans(alpha, k_s, m_a):
    # Transfer the local stiffness into global stiffness
    # Zhanjie 2008
    # modified by Z. Li, Aug. 09, 2009
    # adapted for spring Dec 2015

    total_m = len(m_a)  # Total number of longitudinal terms m
    gamma = np.zeros(8 * total_m, 8 * total_m)

    gam = np.array([[np.cos(alpha), 0, 0, 0, -np.sin(alpha), 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, np.cos(alpha), 0, 0, 0, -np.sin(alpha), 0], [0, 0, 0, 1, 0, 0, 0, 0],
                    [np.sin(alpha), 0, 0, 0, np.cos(alpha), 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, np.sin(alpha), 0, 0, 0, np.cos(alpha), 0], [0, 0, 0, 0, 0, 0, 0, 1]])
    # extend to multi-m
    for i in range(0, total_m):
        gamma[8 * i:8 * (i+1), 8 * i:8 * (i+1)] = gam

    ksglobal = gamma @ k_s @ gamma.conj().T

    return ksglobal


def spring_assemble(k_global, k_local, node_i, node_j, n_nodes, m_a):
    # Add the (spring) contribution to the global stiffness matrix

    # Outputs:
    # k_global: global elastic stiffness matrix
    # k_global and Kg: total_m x total_m submatrices. Each submatrix is similar to the
    # one used in original CUFSM for single longitudinal term m in the DOF order
    #[u1 v1...un vn w1 01...wn 0n]m'.

    # Z. Li, June 2008
    # modified by Z. Li, Aug. 09, 2009
    # Z. Li, June 2010
    # adapted for springs BWS Dec 2015

    total_m = len(m_a)  # Total number of longitudinal terms m
    k_2_matrix = np.zeros(4 * n_nodes * total_m, 4 * n_nodes * total_m)
    skip = 2 * n_nodes
    for i in range(0, total_m):
        for j in range(0, total_m):
            # Submatrices for the initial stiffness
            k11 = k_local[8 * i:8*i + 2, 8 * j:8*j + 2]
            k12 = k_local[8 * i:8*i + 2, 8*j + 2:8*j + 4]
            k13 = k_local[8 * i:8*i + 2, 8*j + 4:8*j + 6]
            k14 = k_local[8 * i:8*i + 2, 8*j + 6:8*j + 8]
            k21 = k_local[8*i + 2:8*i + 4, 8 * j:8*j + 2]
            k22 = k_local[8*i + 2:8*i + 4, 8*j + 2:8*j + 4]
            k23 = k_local[8*i + 2:8*i + 4, 8*j + 4:8*j + 6]
            k24 = k_local[8*i + 2:8*i + 4, 8*j + 6:8*j + 8]
            k31 = k_local[8*i + 4:8*i + 6, 8 * j:8*j + 2]
            k32 = k_local[8*i + 4:8*i + 6, 8*j + 2:8*j + 4]
            k33 = k_local[8*i + 4:8*i + 6, 8*j + 4:8*j + 6]
            k34 = k_local[8*i + 4:8*i + 6, 8*j + 6:8*j + 8]
            k41 = k_local[8*i + 6:8*i + 8, 8 * j:8*j + 2]
            k42 = k_local[8*i + 6:8*i + 8, 8*j + 2:8*j + 4]
            k43 = k_local[8*i + 6:8*i + 8, 8*j + 4:8*j + 6]
            k44 = k_local[8*i + 6:8*i + 8, 8*j + 6:8*j + 8]

            k_2_matrix[4*n_nodes*i + (node_i+1) * 2 - 1:4*n_nodes*i + (node_i+1) * 2,
                       4*n_nodes*j + (node_i+1) * 2 - 1:4*n_nodes*j + (node_i+1) * 2] = k11
            if node_j != 0:
                k_2_matrix[4*n_nodes*i + (node_i+1) * 2 - 1:4*n_nodes*i + (node_i+1) * 2,
                           4*n_nodes*j + (node_j+1) * 2 - 1:4*n_nodes*j + (node_j+1) * 2] = k12
                k_2_matrix[4*n_nodes*i + (node_j+1) * 2 - 1:4*n_nodes*i + (node_j+1) * 2,
                           4*n_nodes*j + (node_i+1) * 2 - 1:4*n_nodes*j + (node_i+1) * 2] = k21
                k_2_matrix[4*n_nodes*i + (node_j+1) * 2 - 1:4*n_nodes*i + (node_j+1) * 2,
                           4*n_nodes*j + (node_j+1) * 2 - 1:4*n_nodes*j + (node_j+1) * 2] = k22

            k_2_matrix[4*n_nodes*i + skip + (node_i+1) * 2 - 1:4*n_nodes*i + skip + (node_i+1) * 2,
                       4*n_nodes*j + skip + (node_i+1) * 2 - 1:4*n_nodes*j + skip
                       + (node_i+1) * 2] = k33
            if node_j != 0:
                k_2_matrix[4*n_nodes*i + skip + (node_i+1) * 2 - 1:4*n_nodes*i + skip
                           + (node_i+1) * 2, 4*n_nodes*j + skip + (node_j+1) * 2 - 1:4*n_nodes*j
                           + skip + (node_j+1) * 2] = k34
                k_2_matrix[4*n_nodes*i + skip + (node_j+1) * 2 - 1:4*n_nodes*i + skip
                           + (node_j+1) * 2, 4*n_nodes*j + skip + (node_i+1) * 2 - 1:4*n_nodes*j
                           + skip + (node_i+1) * 2] = k43
                k_2_matrix[4*n_nodes*i + skip + (node_j+1) * 2 - 1:4*n_nodes*i + skip
                           + (node_j+1) * 2, 4*n_nodes*j + skip + (node_j+1) * 2 - 1:4*n_nodes*j
                           + skip + (node_j+1) * 2] = k44

            k_2_matrix[4*n_nodes*i + (node_i+1) * 2 - 1:4*n_nodes*i + (node_i+1) * 2, 4*n_nodes*j
                       + skip + (node_i+1) * 2 - 1:4*n_nodes*j + skip + (node_i+1) * 2] = k13
            if node_j != 0:
                k_2_matrix[4*n_nodes*i + (node_i+1) * 2 - 1:4*n_nodes*i + (node_i+1) * 2,
                           4*n_nodes*j + skip + (node_j+1) * 2 - 1:4*n_nodes*j + skip
                           + (node_j+1) * 2] = k14
                k_2_matrix[4*n_nodes*i + (node_j+1) * 2 - 1:4*n_nodes*i + (node_j+1) * 2,
                           4*n_nodes*j + skip + (node_i+1) * 2 - 1:4*n_nodes*j + skip
                           + (node_i+1) * 2] = k23
                k_2_matrix[4*n_nodes*i + (node_j+1) * 2 - 1:4*n_nodes*i + (node_j+1) * 2,
                           4*n_nodes*j + skip + (node_j+1) * 2 - 1:4*n_nodes*j + skip
                           + (node_j+1) * 2] = k24

            k_2_matrix[4*n_nodes*i + skip + (node_i+1) * 2 - 1:4*n_nodes*i + skip + (node_i+1) * 2,
                       4*n_nodes*j + (node_i+1) * 2 - 1:4*n_nodes*j + (node_i+1) * 2] = k31
            if node_j != 0:
                k_2_matrix[4*n_nodes*i + skip + (node_i+1) * 2 - 1:4*n_nodes*i + skip
                           + (node_i+1) * 2,
                           4*n_nodes*j + (node_j+1) * 2 - 1:4*n_nodes*j + (node_j+1) * 2] = k32
                k_2_matrix[4*n_nodes*i + skip + (node_j+1) * 2 - 1:4*n_nodes*i + skip
                           + (node_j+1) * 2,
                           4*n_nodes*j + (node_i+1) * 2 - 1:4*n_nodes*j + (node_i+1) * 2] = k41
                k_2_matrix[4*n_nodes*i + skip + (node_j+1) * 2 - 1:4*n_nodes*i + skip
                           + (node_j+1) * 2,
                           4*n_nodes*j + (node_j+1) * 2 - 1:4*n_nodes*j + (node_j+1) * 2] = k42

    k_global = k_global + k_2_matrix

    return k_global


def ym_at_ys(b_c, m_i, y_s, length):
    # Longitudinal shape function values
    # written by BWS in 2015
    # could be called in lots of places,  but now (2015) is hardcoded by Zhanjie
    # in several places in the interface
    # written in 2015 because wanted it for a new idea on discrete springs

    if b_c == 'S-S':
        y_m = np.sin(m_i * np.pi * y_s / length)
    elif b_c == 'C-C':
        y_m = np.sin(m_i * np.pi * y_s / length) * np.sin(np.pi * y_s / length)
    elif b_c == 'S-C' or b_c == 'C-S':
        y_m = np.sin((m_i+1) * np.pi * y_s / length
                     ) + (m_i+1) / m_i * np.sin(m_i * np.pi * y_s / length)
    elif b_c == 'C-F' or b_c == 'F-C':
        y_m = 1 - np.cos((m_i-0.5) * np.pi * y_s / length)
    elif b_c == 'C-G' or b_c == 'G-C':
        y_m = np.sin((m_i-0.5) * np.pi * y_s / length) * np.sin(np.pi * y_s / length / 2)

    return y_m


def ymprime_at_ys(b_c, m_i, y_s, length):
    # First Derivative of Longitudinal shape function values
    # written by BWS in 2015
    # could be called in lots of places,  but now (2015) is hardcoded by Zhanjie
    # in several places in the interface
    # written in 2015 because wanted it for a new idea on discrete springs

    if b_c == 'S-S':
        y_m_prime = (np.pi * m_i * np.cos((np.pi * m_i * y_s) / length)) / length
    elif b_c == 'C-C':
        y_m_prime = (np.pi * np.cos((np.pi*y_s) / length) \
            * np.sin((np.pi*m_i*y_s) / length)) / length \
            + (np.pi*m_i * np.sin((np.pi*y_s)/length) \
                * np.cos((np.pi*m_i*y_s)/length)) / length
    elif b_c == 'S-C' or b_c == 'C-S':
        y_m_prime = (np.pi * np.cos((np.pi*y_s * (m_i + 1))/length) * (m_i + 1)) / length \
            + (np.pi * np.cos((np.pi*m_i*y_s)/length)*(m_i + 1)) / length
    elif b_c == 'C-F' or b_c == 'F-C':
        y_m_prime = (np.pi * np.sin((np.pi * y_s * (m_i - 1/2)) / length) * (m_i - 1/2)) / length
    elif b_c == 'C-G' or b_c == 'G-C':
        y_m_prime = (np.pi*np.sin((np.pi*y_s * (m_i - 1/2))/length) \
            * np.cos((np.pi*y_s)/(2*length)))/(2*length) \
            + (np.pi*np.cos((np.pi*y_s*(m_i - 1/2))/length) \
            * np.sin((np.pi*y_s)/(2*length))*(m_i - 1/2))/length

    return y_m_prime
