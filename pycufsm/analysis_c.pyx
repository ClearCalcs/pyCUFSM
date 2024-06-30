# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3, boundscheck=False

# import cython
cimport numpy as np
import numpy as np
np.import_array()

# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE, CPEng
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


cpdef np.ndarray m_sort(np.ndarray m_all):
    """Cleans up longitudinal terms by removing any zero terms, removing any duplicates, and sorting

    Args:
        m_all (np.ndarray): list of all longitudinal terms for each half-wavelength

    Returns:
        m_all (np.ndarray): cleaned list of all longitudinal terms for each half-wavelength
    """
    # Declare looping variables
    cdef int i
    cdef np.ndarray m_a
    for i, m_a in enumerate(m_all):
        # return all the nonzero longitudinal terms in m_a as a column vector
        m_a = m_a[np.nonzero(m_a)]
        m_a = np.unique(m_a)  # remove repetitive longitudinal terms
        m_all[i] = m_a
    return m_all


cpdef int constr_BC_flag(np.ndarray nodes, np.ndarray constraints):
    """this subroutine is to determine flags for user constraints and internal (at node) B.C.'s

    Args:
        nodes (np.ndarray): [node# x y dof_x dof_y dof_z dof_r stress] n_nodes x 8
        constraints (np.ndarray): [node# e dof_e coeff node# k dof_k] e=dof to be eliminated
            k=kept dof dofe_node = coeff*dofk_nodek

    Returns:
        BC_flag (int): 1 if there are user constraints or node fixities
            0 if there is no user constraints and node fixities
    
    Z. Li, June 2010
    """
    # Check for boundary conditions on the nodes
    cdef int n_nodes = len(nodes)
    cdef int i
    cdef int j
    for i in range(0, n_nodes):
        for j in range(3, 7):
            if nodes[i, j] == 0:
                return 1

    # Check for user defined constraints too
    if len(constraints) == 0:
        return 0
    else:
        return 1


cpdef np.ndarray elem_prop(np.ndarray nodes, np.ndarray elements):
    """creates a matrix of element properties for each element (width and slope)

    Args:
        nodes (np.ndarray): standard nodes array
        elements (np.ndarray): standard elements array

    Returns:
        el_props (np.ndarray): [id, width, alpha]
    """
    cdef np.ndarray[np.double_t, ndim=2] el_props = np.zeros((len(elements), 3))
    cdef int n_elems = len(elements)

    # Declare looping variables
    cdef int i
    cdef int node_i
    cdef int node_j
    cdef double x_i
    cdef double y_i
    cdef double x_j
    cdef double y_j
    cdef double d_x
    cdef double d_y
    cdef double width
    cdef double alpha

    for i in range(n_elems):
        node_i = int(elements[i, 1])
        node_j = int(elements[i, 2])
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


cpdef k_kg_global(
    np.ndarray nodes, np.ndarray elements, np.ndarray el_props, np.ndarray props, 
    double length, str B_C, np.ndarray m_a
):
    """Generates element stiffness matrix (K_global) in global coordinates
    Generate geometric stiffness matrix (Kg_global) in global coordinates

    Args:
        nodes (np.ndarray): 
        elements (np.ndarray): 
        el_props (np.ndarray): 
        props (np.ndarray): 
        length (np.ndarray): 
        B_C (str): 
        m_a (np.ndarray): 

    Returns:
        K_global (np.ndarray): global stiffness matrix
        Kg_global (np.ndarray): global geometric stiffness matrix

    B Smith, May 2023
    """
    cdef int total_m = len(m_a)
    cdef int n_nodes = len(nodes)
    cdef int n_elems = len(elements)

    # ZERO OUT THE GLOBAL MATRICES
    cdef np.ndarray[np.double_t, ndim=2] K_global = np.zeros((4 * n_nodes * total_m, 4 * n_nodes * total_m))
    cdef np.ndarray[np.double_t, ndim=2] Kg_global = np.zeros((4 * n_nodes * total_m, 4 * n_nodes * total_m))

    # Declare looping variables
    cdef int i
    cdef double thick
    cdef double b_strip
    cdef int mat_num
    cdef int row
    cdef double stiff_x
    cdef double E_y
    cdef double nu_x
    cdef double nu_y
    cdef double G_bulk
    cdef int node_i
    cdef int node_j
    cdef double Ty_1
    cdef double Ty_2
    cdef double alpha
    cdef np.ndarray[np.double_t, ndim=2] k_l
    cdef np.ndarray[np.double_t, ndim=2] kg_l
    cdef np.ndarray[np.double_t, ndim=2] k_local
    cdef np.ndarray[np.double_t, ndim=2] kg_local

    # ASSEMBLE THE GLOBAL STIFFNESS MATRICES
    for i in range(0, n_elems):
        # Generate element stiffness matrix (k_local) in local coordinates
        # Generate geometric stiffness matrix (kg_local) in local coordinates
        thick = elements[i, 3]
        b_strip = el_props[i, 1]
        mat_num = int(elements[i, 4])
        row = int((np.argwhere(props[:, 0] == mat_num)).reshape(1))
        stiff_x = props[row, 1]
        E_y = props[row, 2]
        nu_x = props[row, 3]
        nu_y = props[row, 4]
        G_bulk = props[row, 5]

        node_i = int(elements[i, 1])
        node_j = int(elements[i, 2])
        Ty_1 = nodes[node_i, 7] * thick
        Ty_2 = nodes[node_j, 7] * thick

        k_l, kg_l = k_kg_local(
            stiff_x=stiff_x,
            E_y=E_y,
            nu_x=nu_x,
            nu_y=nu_y,
            G_bulk=G_bulk,
            thick=thick,
            length=length,
            Ty_1=Ty_1,
            Ty_2=Ty_2,
            b_strip=b_strip,
            B_C=B_C,
            m_a=m_a
        )

        # Transform k_local and kg_local into global coordinates
        alpha = el_props[i, 2]
        gamma = trans(alpha=alpha, total_m=total_m)
        k_local = gamma @ k_l @ gamma.conj().T
        kg_local = gamma @ kg_l @ gamma.conj().T

        # Add element contribution of k_local to full matrix K_global and kg_local to Kg_global
        K_global, Kg_global = assemble(
            K_global=K_global,
            Kg_global=Kg_global,
            k_local=k_local,
            kg_local=kg_local,
            node_i=node_i,
            node_j=node_j,
            n_nodes=n_nodes,
            m_a=m_a
        )

    return K_global, Kg_global


cdef k_kg_local(
    double stiff_x, double E_y, double nu_x, double nu_y, double G_bulk, double thick,
    double length, double Ty_1, double Ty_2, double b_strip, str B_C, np.ndarray m_a
):
    """Generate element stiffness matrix (k_local) in local coordinates
    Generate geometric stiffness matrix (kg_local) in local coordinates

    Args:
        stiff_x (float): material property
        E_y (float): material property
        nu_x (float): material property
        nu_y (float): material property
        G_bulk (float): material property
        thick (float): thickness of the strip (element)
        length (np.ndarray): length of the strip in longitudinal direction
        Ty_1 (float): node stresses
        Ty_2 (float): node stresses
        b_strip (float): width of the strip in transverse direction
        B_C (str): ['S-S'] a string specifying boundary conditions to be analyzed:
            'S-S' simply-pimply supported boundary condition at loaded edges
            'C-C' clamped-clamped boundary condition at loaded edges
            'S-C' simply-clamped supported boundary condition at loaded edges
            'C-F' clamped-free supported boundary condition at loaded edges
            'C-G' clamped-gcdef np.ndarray[np.double_t, ndim=2] uided supported boundary condition
                at loaded edges
        m_a (np.ndarray): longitudinal terms (or half-wave numbers) for this length

    Returns:
        k_local (np.ndarray): local stiffness matrix, a total_m x total_m matrix of 8 by 8 
            submatrices. k_local=[k_mp]total_m x total_m block matrix
            each k_mp is the 8 x 8 submatrix in the DOF order [u1 v1 u2 v2 w1 theta1 w2 theta2]'
        kg_local (np.ndarray): local geometric stiffness matrix, a total_m x total_m matrix of 
            8 by 8 submatrices. kg_local=[kg_mp]total_m x total_m block matrix
            each kg_mp is the 8 x 8 submatrix in the DOF order [u1 v1 u2 v2 w1 theta1

    Z. Li June 2008
    modified by Z. Li, Aug. 09, 2009
    modified by Z. Li, June 2010
    klocal and kglocal merged by B Smith, May 2023
    """
    cdef double E_1 = stiff_x / (1 - nu_x*nu_y)
    cdef double E_2 = E_y / (1 - nu_x*nu_y)
    cdef double d_x = stiff_x * thick**3 / (12 * (1 - nu_x*nu_y))
    cdef double d_y = E_y * thick**3 / (12 * (1 - nu_x*nu_y))
    cdef double d_1 = nu_x * E_y * thick**3 / (12 * (1 - nu_x*nu_y))
    cdef double d_xy = G_bulk * thick**3 / 12

    cdef int total_m = len(m_a)  # Total number of longitudinal terms m

    cdef np.ndarray[np.double_t, ndim=2] k_local = np.zeros((8 * total_m, 8 * total_m), dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] kg_local = np.zeros((8 * total_m, 8 * total_m), dtype=np.double)

    # declare looping variables
    cdef double u_i
    cdef double u_j
    cdef double c_1
    cdef double c_2
    cdef int i
    cdef int j

    for i in range(0, total_m):
        for j in range(0, total_m):
            u_i = m_a[i] * np.pi
            u_j = m_a[j] * np.pi
            c_1 = u_i / length
            c_2 = u_j / length

            [I_1, I_2, I_3, I_4, I_5] = bc_i1_5(B_C=B_C, m_i=m_a[i], m_j=m_a[j], length=length)

            k_local[8 * i:8*i + 4, 8 * j:8*j + 4] = calc_km_mp(
                E_1=E_1,
                E_2=E_2,
                c_1=c_1,
                c_2=c_2,
                b_strip=b_strip,
                G_bulk=G_bulk,
                nu_x=nu_x,
                thick=thick,
                I_1=I_1,
                I_2=I_2,
                I_3=I_3,
                I_4=I_4,
                I_5=I_5
            )
            k_local[8*i + 4:8 * (i+1), 8*j + 4:8 * (j+1)] = calc_kf_mp(
                d_x=d_x,
                d_y=d_y,
                d_1=d_1,
                d_xy=d_xy,
                b_strip=b_strip,
                I_1=I_1,
                I_2=I_2,
                I_3=I_3,
                I_4=I_4,
                I_5=I_5
            )

            kg_local[8 * i:8*i + 4, 8 * j:8*j + 4] = calc_gm_mp(
                u_i=u_i,
                u_j=u_j,
                b_strip=b_strip,
                length=length,
                Ty_1=Ty_1,
                Ty_2=Ty_2,
                I_4=I_4,
                I_5=I_5
            )
            kg_local[8*i + 4:8 * (i+1), 8*j + 4:8 * (j+1)] = calc_gf_mp(
                Ty_1=Ty_1, Ty_2=Ty_2, b_strip=b_strip, I_5=I_5
            )

    return k_local, kg_local


cpdef np.ndarray kglobal_transv(
    np.ndarray nodes, np.ndarray elements, np.ndarray el_props, np.ndarray props, 
    double length, str B_C, double m_i
):
    """this routine creates the global stiffness matrix for planar displacements
    basically the same way as in the main program, however:
      only one half-wave number m_i is considered,
      only w, teta terms are considered,
      plus E_y = nu_x = nu_y = 0 is assumed
      plus the longitudinal displacements. DOFs are explicitly eliminated
      the multiplication by 'length' (member length) is not done here, must be done
         outside of this routine

    Args:
        nodes (np.ndarray): standard parameter
        elements (np.ndarray): standard parameter
        props (np.ndarray): standard parameter
        m_i (float): number of half-wavelengths
        length (float): half-wavelength
        B_C (str): boundary condition
        el_props (np.ndarray): element propertise

    Returns:
        k_global_transv (np.ndarray): global stiffness matrix

    S. Adany, Feb 08, 2004
    Z. Li, Jul 10, 2009
    """
    cdef int n_nodes = len(nodes)
    cdef int n_elems = len(elements)
    cdef np.ndarray[np.double_t, ndim=2] k_global_transv = np.zeros((4 * n_nodes, 4 * n_nodes))

    # Declare looping variables
    cdef int i
    cdef double thick
    cdef double b_strip
    cdef int mat_num
    cdef int row
    cdef double stiff_x
    cdef double E_y
    cdef double nu_x
    cdef double nu_y
    cdef double G_bulk
    cdef int node_i
    cdef int node_j
    cdef double alpha
    cdef np.ndarray[np.double_t, ndim=2] k_l
    cdef np.ndarray[np.double_t, ndim=2] k_local

    for i in range(0, n_elems):
        thick = elements[i, 3]
        b_strip = el_props[i, 1]
        mat_num = int(elements[i, 4])
        row = int((np.argwhere(props[:, 0] == mat_num)).reshape(1))
        stiff_x = props[row, 1]
        E_y = props[row, 2]
        nu_x = props[row, 3]
        nu_y = props[row, 4]
        G_bulk = props[row, 5]
        k_l = klocal_transv(
            stiff_x=stiff_x,
            E_y=E_y,
            nu_x=nu_x,
            nu_y=nu_y,
            G_bulk=G_bulk,
            thick=thick,
            length=length,
            b_strip=b_strip,
            B_C=B_C,
            m_i=m_i
        )

        # Transform k_local and kg_local into global coordinates
        alpha = el_props[i, 2]
        gamma = trans(alpha=alpha, total_m=1)
        k_local = gamma @ k_l @ gamma.conj().T

        # Add element contribution of k_local to full matrix K_global and kg_local to Kg_global
        node_i = int(elements[i, 1])
        node_j = int(elements[i, 2])
        k_global_transv = assemble_single(
            K_global=k_global_transv,
            k_local=k_local,
            node_i=node_i,
            node_j=node_j,
            n_nodes=n_nodes
        )

    return k_global_transv


cdef np.ndarray klocal_transv(
    double stiff_x, double E_y, double nu_x, double nu_y, double G_bulk, double thick, 
    double length, double b_strip, str B_C, double m_i
):
    """this routine creates the local stiffness matrix for bending terms
    basically the same way as in the main program, however:
      only for single half-wave number m_i
      membrane strains practically zero, (membrane moduli are enlarged)
      for bending, only transverse terms are considered, (practically: only
      keeps the I_1 term, set I_2 through I_5 to be zero)
        also different from the main program, here only involves one single
        longitudinal term m_i.

    Args:
        stiff_x (float): material property
        E_y (float): material property
        nu_x (float): material property
        nu_y (float): material property
        G_bulk (float): material property
        thick (float): element thickness
        length (float): element length
        b_strip (float): element widget
        m_i (float): number of half-wavelengths
        B_C (str): boundary condition

    Returns:
        np.ndarray: _description_

    Z. Li, Jul 10, 2009
    """
    cdef double E_1 = stiff_x / (1 - nu_x*nu_y) * 100000000
    cdef double E_2 = E_y / (1 - nu_x*nu_y)
    cdef double d_x = stiff_x * thick**3 / (12 * (1 - nu_x*nu_y))
    cdef double d_y = E_y * thick**3 / (12 * (1 - nu_x*nu_y))
    cdef double d_1 = nu_x * E_y * thick**3 / (12 * (1 - nu_x*nu_y))
    cdef double d_xy = G_bulk * thick**3 / 12

    cdef np.ndarray[np.double_t, ndim=2] k_local = np.zeros((8, 8))
    cdef double u_m = m_i * np.pi
    cdef double u_p = m_i * np.pi
    cdef double c_1 = u_m / length
    cdef double c_2 = u_p / length

    [I_1, _, _, _, _] = bc_i1_5(B_C=B_C, m_i=m_i, m_j=m_i, length=length)
    cdef double I_2 = 0
    cdef double I_3 = 0
    cdef double I_4 = 0
    cdef double I_5 = 0

    k_local[0:4, 0:4] = calc_km_mp(
        E_1=E_1,
        E_2=E_2,
        c_1=c_1,
        c_2=c_2,
        b_strip=b_strip,
        G_bulk=G_bulk,
        nu_x=nu_x,
        thick=thick,
        I_1=I_1,
        I_2=I_2,
        I_3=I_3,
        I_4=I_4,
        I_5=I_5
    )
    k_local[4:8, 4:8] = calc_kf_mp(
        d_x=d_x,
        d_y=d_y,
        d_1=d_1,
        d_xy=d_xy,
        b_strip=b_strip,
        I_1=I_1,
        I_2=I_2,
        I_3=I_3,
        I_4=I_4,
        I_5=I_5
    )
    return k_local


cdef np.ndarray calc_km_mp(
    double E_1, double E_2, double c_1, double c_2, double b_strip, double G_bulk, 
    double nu_x, double thick, double I_1, double I_2, double I_3, double I_4, double I_5
):
    """Calculate the membrane stiffness sub-matrix, used in the assembly of local stiffness
    matrices

    Args:
        E_1 (float): _description_
        E_2 (float): _description_
        c_1 (float): _description_
        c_2 (float): _description_
        b_strip (float): _description_
        G_bulk (float): _description_
        nu_x (float): _description_
        thick (float): _description_
        I_1 (float): _description_
        I_2 (float): _description_
        I_3 (float): _description_
        I_4 (float): _description_
        I_5 (float): _description_

    Returns:
        km_mp (np.ndarray): membrane stiffness sub-matrix

    Z. Li June 2008
    modified by Z. Li, Aug. 09, 2009
    modified by Z. Li, June 2010
    Pulled out of klocal() function by B Smith, May 2023
    """
    cdef np.ndarray[np.double_t, ndim=2] km_mp = np.zeros((4, 4))

    # assemble the matrix of Km_mp (membrane stiffness matrix)
    km_mp[0, 0] = E_1*I_1/b_strip + G_bulk*b_strip*I_5/3
    km_mp[0, 1] = E_2 * nu_x * (-1 / 2 / c_2) * I_3 - G_bulk*I_5/2/c_2
    km_mp[0, 2] = -E_1 * I_1 / b_strip + G_bulk*b_strip*I_5/6
    km_mp[0, 3] = E_2 * nu_x * (-1 / 2 / c_2) * I_3 + G_bulk*I_5/2/c_2

    km_mp[1, 0] = E_2 * nu_x * (-1 / 2 / c_1) * I_2 - G_bulk*I_5/2/c_1
    km_mp[1, 1] = E_2*b_strip*I_4/3/c_1/c_2 + G_bulk*I_5/b_strip/c_1/c_2
    km_mp[1, 2] = E_2 * nu_x * (1/2/c_1) * I_2 - G_bulk*I_5/2/c_1
    km_mp[1, 3] = E_2*b_strip*I_4/6/c_1/c_2 - G_bulk*I_5/b_strip/c_1/c_2

    km_mp[2, 0] = -E_1 * I_1 / b_strip + G_bulk*b_strip*I_5/6
    km_mp[2, 1] = E_2 * nu_x * (1/2/c_2) * I_3 - G_bulk*I_5/2/c_2
    km_mp[2, 2] = E_1*I_1/b_strip + G_bulk*b_strip*I_5/3
    km_mp[2, 3] = E_2 * nu_x * (1/2/c_2) * I_3 + G_bulk*I_5/2/c_2

    km_mp[3, 0] = E_2 * nu_x * (-1 / 2 / c_1) * I_2 + G_bulk*I_5/2/c_1
    km_mp[3, 1] = E_2*b_strip*I_4/6/c_1/c_2 - G_bulk*I_5/b_strip/c_1/c_2
    km_mp[3, 2] = E_2 * nu_x * (1/2/c_1) * I_2 + G_bulk*I_5/2/c_1
    km_mp[3, 3] = E_2*b_strip*I_4/3/c_1/c_2 + G_bulk*I_5/b_strip/c_1/c_2

    return km_mp * thick

cdef np.ndarray calc_kf_mp(
    double d_x, double d_y, double d_1, double d_xy, double b_strip, double I_1, 
    double I_2, double I_3, double I_4, double I_5
):
    """Calculate the flexural stiffness sub-matrix, used in the assembly of local stiffness
    matrices

    Args:
        d_x (float): _description_
        d_y (float): _description_
        d_1 (float): _description_
        d_xy (float): _description_
        b_strip (float): _description_
        I_1 (float): _description_
        I_2 (float): _description_
        I_3 (float): _description_
        I_4 (float): _description_
        I_5 (float): _description_

    Returns:
        np.ndarray: _description_

    Z. Li June 2008
    modified by Z. Li, Aug. 09, 2009
    modified by Z. Li, June 2010
    Pulled out of klocal() function by B Smith, May 2023
    """
    cdef np.ndarray[np.double_t, ndim=2] kf_mp = np.zeros((4, 4))

    # assemble the matrix of Kf_mp (flexural stiffness matrix)
    kf_mp[0, 0] = (5040*d_x*I_1 - 504*b_strip**2*d_1*I_2 - 504*b_strip**2*d_1*I_3 \
        + 156*b_strip**4*d_y*I_4 + 2016*b_strip**2*d_xy*I_5)/420/b_strip**3
    kf_mp[0, 1] = (2520*b_strip*d_x*I_1 - 462*b_strip**3*d_1*I_2 - 42*b_strip**3*d_1*I_3 \
        + 22*b_strip**5*d_y*I_4 + 168*b_strip**3*d_xy*I_5)/420/b_strip**3
    kf_mp[0, 2] = (-5040*d_x*I_1 + 504*b_strip**2*d_1*I_2 + 504*b_strip**2*d_1*I_3 \
        + 54*b_strip**4*d_y*I_4 - 2016*b_strip**2*d_xy*I_5)/420/b_strip**3
    kf_mp[0, 3] = (2520*b_strip*d_x*I_1 - 42*b_strip**3*d_1*I_2 - 42*b_strip**3*d_1*I_3 \
        - 13*b_strip**5*d_y*I_4 + 168*b_strip**3*d_xy*I_5)/420/b_strip**3

    kf_mp[1, 0] = (2520*b_strip*d_x*I_1 - 462*b_strip**3*d_1*I_3 - 42*b_strip**3*d_1*I_2 \
        + 22*b_strip**5*d_y*I_4 + 168*b_strip**3*d_xy*I_5)/420/b_strip**3
    kf_mp[1, 1] = (1680*b_strip**2*d_x*I_1 - 56*b_strip**4*d_1*I_2 - 56*b_strip**4*d_1*I_3 \
        + 4*b_strip**6*d_y*I_4 + 224*b_strip**4*d_xy*I_5)/420/b_strip**3
    kf_mp[1, 2] = (-2520*b_strip*d_x*I_1 + 42*b_strip**3*d_1*I_2 + 42*b_strip**3*d_1*I_3 \
        + 13*b_strip**5*d_y*I_4 - 168*b_strip**3*d_xy*I_5)/420/b_strip**3
    kf_mp[1, 3] = (840*b_strip**2*d_x*I_1 + 14*b_strip**4*d_1*I_2 + 14*b_strip**4*d_1*I_3 \
        - 3*b_strip**6*d_y*I_4 - 56*b_strip**4*d_xy*I_5)/420/b_strip**3

    kf_mp[2, 0] = kf_mp[0, 2]
    kf_mp[2, 1] = kf_mp[1, 2]
    kf_mp[2, 2] = (5040*d_x*I_1 - 504*b_strip**2*d_1*I_2 - 504*b_strip**2*d_1*I_3 \
        + 156*b_strip**4*d_y*I_4 + 2016*b_strip**2*d_xy*I_5)/420/b_strip**3
    kf_mp[2, 3] = (-2520*b_strip*d_x*I_1 + 462*b_strip**3*d_1*I_2 + 42*b_strip**3*d_1*I_3 \
        - 22*b_strip**5*d_y*I_4 - 168*b_strip**3*d_xy*I_5)/420/b_strip**3

    kf_mp[3, 0] = kf_mp[0, 3]
    kf_mp[3, 1] = kf_mp[1, 3]
    kf_mp[3, 2] = (-2520*b_strip*d_x*I_1 + 462*b_strip**3*d_1*I_3 + 42*b_strip**3*d_1*I_2 \
        - 22*b_strip**5*d_y*I_4 - 168*b_strip**3*d_xy*I_5)/420/b_strip**3 # not symmetric
    kf_mp[3, 3] = (1680*b_strip**2*d_x*I_1 - 56*b_strip**4*d_1*I_2 - 56*b_strip**4*d_1*I_3 \
        + 4*b_strip**6*d_y*I_4 + 224*b_strip**4*d_xy*I_5)/420/b_strip**3

    return kf_mp

cdef np.ndarray calc_gm_mp(
    double u_i, double u_j, double b_strip, double length, double Ty_1, double Ty_2, 
    double I_4, double I_5
):
    """Calculate the membrane geometric stiffness sub-matrix, used in the assembly of local 
    geometric stiffness matrices

    Args:
        u_i (float): _description_
        u_j (float): _description_
        b_strip (float): _description_
        length (float): _description_
        Ty_1 (float): _description_
        Ty_2 (float): _description_
        I_4 (float): _description_
        I_5 (float): _description_

    Returns:
        np.ndarray: _description_

    Z. Li June 2008
    modified by Z. Li, Aug. 09, 2009
    modified by Z. Li, June 2010
    Pulled out of kglocal() function by B Smith, May 2023
    """
    cdef np.ndarray[np.double_t, ndim=2] gm_mp = np.zeros((4, 4))

    # assemble the matrix of gm_mp (symmetric membrane stability matrix)
    gm_mp[0, 0] = b_strip * (3*Ty_1 + Ty_2) * I_5 / 12
    gm_mp[0, 2] = b_strip * (Ty_1+Ty_2) * I_5 / 12
    gm_mp[2, 0] = gm_mp[0, 2]
    gm_mp[1, 1] = b_strip * length**2 * (3*Ty_1 + Ty_2) * I_4 / 12 / u_i / u_j
    gm_mp[1, 3] = b_strip * length**2 * (Ty_1+Ty_2) * I_4 / 12 / u_i / u_j
    gm_mp[3, 1] = gm_mp[1, 3]
    gm_mp[2, 2] = b_strip * (Ty_1 + 3*Ty_2) * I_5 / 12
    gm_mp[3, 3] = b_strip * length**2 * (Ty_1 + 3*Ty_2) * I_4 / 12 / u_i / u_j

    return gm_mp


cdef np.ndarray calc_gf_mp(double Ty_1, double Ty_2, double b_strip, double I_5):
    """Calculate the flexural geometric stiffness sub-matrix, used in the assembly of local 
    geometric stiffness matrices

    Args:
        Ty_1 (float): _description_
        Ty_2 (float): _description_
        b_strip (float): _description_
        I_5 (float): _description_

    Returns:
        np.ndarray: _description_

    Z. Li June 2008
    modified by Z. Li, Aug. 09, 2009
    modified by Z. Li, June 2010
    Pulled out of kglocal() function by B Smith, May 2023
    """
    cdef np.ndarray[np.double_t, ndim=2] gf_mp = np.zeros((4, 4))

    # assemble the matrix of gf_mp (symmetric flexural stability matrix)
    gf_mp[0, 0] = (10*Ty_1 + 3*Ty_2) * b_strip * I_5 / 35
    gf_mp[0, 1] = (15*Ty_1 + 7*Ty_2) * b_strip**2 * I_5 / 210 / 2
    gf_mp[1, 0] = gf_mp[0, 1]
    gf_mp[0, 2] = 9 * (Ty_1+Ty_2) * b_strip * I_5 / 140
    gf_mp[2, 0] = gf_mp[0, 2]
    gf_mp[0, 3] = -(7*Ty_1 + 6*Ty_2) * b_strip**2 * I_5 / 420
    gf_mp[3, 0] = gf_mp[0, 3]
    gf_mp[1, 1] = (5*Ty_1 + 3*Ty_2) * b_strip**3 * I_5 / 2 / 420
    gf_mp[1, 2] = (6*Ty_1 + 7*Ty_2) * b_strip**2 * I_5 / 420
    gf_mp[2, 1] = gf_mp[1, 2]
    gf_mp[1, 3] = -(Ty_1 + Ty_2) * b_strip**3 * I_5 / 140 / 2
    gf_mp[3, 1] = gf_mp[1, 3]
    gf_mp[2, 2] = (3*Ty_1 + 10*Ty_2) * b_strip * I_5 / 35
    gf_mp[2, 3] = -(7*Ty_1 + 15*Ty_2) * b_strip**2 * I_5 / 420
    gf_mp[3, 2] = gf_mp[2, 3]
    gf_mp[3, 3] = (3*Ty_1 + 5*Ty_2) * b_strip**3 * I_5 / 420 / 2

    return gf_mp


cdef bc_i1_5(str B_C, double m_i, double m_j, double length):
    """Calculate the 5 undetermined parameters I_1,I_2,I_3,I_4,I_5 for local elastic
    and geometric stiffness matrices.

    Args:
        B_C (str): a string specifying boundary conditions to be analysed:
            'S-S' simply-pimply supported boundary condition at loaded edges
            'C-C' clamped-clamped boundary condition at loaded edges
            'S-C' simply-clamped supported boundary condition at loaded edges
            'C-F' clamped-free supported boundary condition at loaded edges
            'C-G' clamped-guided supported boundary condition at loaded edges
        m_i (float): number of half-wavelengths for node i
        m_j (float): number of half-wavelengths for node j
        length (float): length of element

    Returns:
        I_1 though 5 (list): 5 undetermined parameters I_1,I_2,I_3,I_4,I_5
            calculation of I_1 is the integration of y_m*Yn from 0 to length
            calculation of I_2 is the integration of y_m''*Yn from 0 to length
            calculation of I_3 is the integration of y_m*Yn'' from 0 to length
            calculation of I_3 is the integration of y_m*Yn'' from 0 to length
            calculation of I_4 is the integration of y_m''*Yn'' from 0 to length
            calculation of I_5 is the integration of y_m'*Yn' from 0 to length
    """
    cdef double I_1 = 0
    cdef double I_2 = 0
    cdef double I_3 = 0
    cdef double I_4 = 0
    cdef double I_5 = 0

    if B_C == 'S-S':
        # For simply-pimply supported boundary condition at loaded edges
        if m_i == m_j:
            I_1 = length / 2
            I_2 = -m_i**2 * np.pi**2 / length / 2
            I_3 = -m_j**2 * np.pi**2 / length / 2
            I_4 = np.pi**4 * m_i**4 / 2 / length**3
            I_5 = np.pi**2 * m_i**2 / 2 / length

    elif B_C == 'C-C':
        # For Clamped-clamped boundary condition at loaded edges
        # calculation of I_1 is the integration of y_m*Yn from 0 to length
        if m_i == m_j:
            if m_i == 1:
                I_1 = 3 * length / 8
            else:
                I_1 = length / 4
            I_2 = -(m_i**2 + 1) * np.pi**2 / 4 / length
            I_3 = -(m_j**2 + 1) * np.pi**2 / 4 / length
            I_4 = np.pi**4 * ((m_i**2 + 1)**2 + 4 * m_i**2) / 4 / length**3
            I_5 = (1 + m_i**2) * np.pi**2 / 4 / length
        else:
            if m_i - m_j == 2:
                I_1 = -length / 8
                I_2 = (m_i**2 + 1) * np.pi**2 / 8 / length - m_i * np.pi**2 / 4 / length
                I_3 = (m_j**2 + 1) * np.pi**2 / 8 / length + m_j * np.pi**2 / 4 / length
                I_4 = -(m_i - 1)**2 * (m_j + 1)**2 * np.pi**4 / 8 / length**3
                I_5 = -(1 + m_i*m_j) * np.pi**2 / 8 / length
            elif m_i - m_j == -2:
                I_1 = -length / 8
                I_2 = (m_i**2 + 1) * np.pi**2 / 8 / length + m_i * np.pi**2 / 4 / length
                I_3 = (m_j**2 + 1) * np.pi**2 / 8 / length - m_j * np.pi**2 / 4 / length
                I_4 = -(m_i + 1)**2 * (m_j - 1)**2 * np.pi**4 / 8 / length**3
                I_5 = -(1 + m_i*m_j) * np.pi**2 / 8 / length

    elif B_C == 'S-C' or B_C == 'C-S':
        # For simply-clamped supported boundary condition at loaded edges
        # calculation of I_1 is the integration of y_m*Yn from 0 to length
        if m_i == m_j:
            I_1 = (1 + (m_i + 1)**2 / m_i**2) * length / 2
            I_2 = -(m_i + 1)**2 * np.pi**2 / length
            I_3 = -(m_i + 1)**2 * np.pi**2 / length
            I_4 = (m_i + 1)**2 * np.pi**4 * ((m_i + 1)**2 + m_i**2) / 2 / length**3
            I_5 = (1 + m_i)**2 * np.pi**2 / length
        else:
            if m_i - m_j == 1:
                I_1 = (m_i+1) * length / 2 / m_i
                I_2 = -(m_i + 1) * m_i * np.pi**2 / 2 / length
                I_3 = -(m_j + 1)**2 * np.pi**2 * (m_i+1) / 2 / length / m_i
                I_4 = (m_i+1) * m_i * (m_j + 1)**2 * np.pi**4 / 2 / length**3
                I_5 = (1+m_i) * (1+m_j) * np.pi**2 / 2 / length
            elif m_i - m_j == -1:
                I_1 = (m_j+1) * length / 2 / m_j
                I_2 = -(m_i + 1)**2 * np.pi**2 * (m_j+1) / 2 / length / m_j
                I_3 = -(m_j + 1) * m_j * np.pi**2 / 2 / length
                I_4 = (m_i + 1)**2 * m_j * (m_j+1) * np.pi**4 / 2 / length**3
                I_5 = (1+m_i) * (1+m_j) * np.pi**2 / 2 / length

    elif B_C == 'C-F' or B_C == 'F-C':
        # For clamped-free supported boundary condition at loaded edges
        # calculation of I_1 is the integration of y_m*Yn from 0 to length
        if m_i == m_j:
            I_1 = 3*length/2 - 2 * length * (-1)**(m_i - 1) / (m_i - 1/2) / np.pi
            I_2 = (m_i - 1/2)**2 * np.pi**2 * ((-1)**(m_i - 1) / (m_i - 1/2) / np.pi - 1/2) / length
            I_3 = (m_j - 1/2)**2 * np.pi**2 * ((-1)**(m_j - 1) / (m_j - 1/2) / np.pi - 1/2) / length
            I_4 = (m_i - 1/2)**4 * np.pi**4 / 2 / length**3
            I_5 = (m_i - 1/2)**2 * np.pi**2 / 2 / length
        else:
            I_1 = length - length*(-1)**(m_i - 1) / (m_i - 1/2) / np.pi \
                - length*(-1)**(m_j - 1) / (m_j - 1/2) / np.pi
            I_2 = (m_i - 1/2)**2 * np.pi**2 * ((-1)**(m_i - 1) / (m_i - 1/2) / np.pi) / length
            I_3 = (m_j - 1/2)**2 * np.pi**2 * ((-1)**(m_j - 1) / (m_j - 1/2) / np.pi) / length

    elif B_C == 'C-G' or B_C == 'G-C':
        # For clamped-guided supported boundary condition at loaded edges
        # calculation of I_1 is the integration of y_m*Yn from 0 to length
        if m_i == m_j:
            if m_i == 1:
                I_1 = 3 * length / 8
            else:
                I_1 = length / 4
            I_2 = -((m_i - 1/2)**2 + 1/4) * np.pi**2 / length / 4
            I_3 = -((m_i - 1/2)**2 + 1/4) * np.pi**2 / length / 4
            I_4 = ((m_i - 1/2)**2
                   + 1/4)**2 * np.pi**4 / 4 / length**3 + (m_i - 1/2)**2 * np.pi**4 / 4 / length**3
            I_5 = (m_i - 1/2)**2 * np.pi**2 / length / 4 + np.pi**2 / 16 / length
        else:
            if m_i - m_j == 1:
                I_1 = -length / 8
                I_2 = ((m_i - 1/2)**2
                       + 1/4) * np.pi**2 / length / 8 - (m_i - 1/2) * np.pi**2 / length / 8
                I_3 = ((m_j - 1/2)**2
                       + 1/4) * np.pi**2 / length / 8 + (m_j - 1/2) * np.pi**2 / length / 8
                I_4 = -m_j**4 * np.pi**4 / 8 / length**3
                I_5 = -m_j**2 * np.pi**2 / 8 / length
            elif m_i - m_j == -1:
                I_1 = -length / 8
                I_2 = ((m_i - 1/2)**2
                       + 1/4) * np.pi**2 / length / 8 + (m_i - 1/2) * np.pi**2 / length / 8
                I_3 = ((m_j - 1/2)**2
                       + 1/4) * np.pi**2 / length / 8 - (m_j - 1/2) * np.pi**2 / length / 8
                I_4 = -m_i**4 * np.pi**4 / 8 / length**3
                I_5 = -m_i**2 * np.pi**2 / 8 / length

    return [I_1, I_2, I_3, I_4, I_5]


cpdef np.ndarray trans(float alpha, int total_m):
    """Transform local stiffness into global stiffness

    Args:
        alpha (float): element angle
        total_m (int): number of half-wavelengths

    Returns:
        gamma (np.ndarray): transformation matrix
    
    Zhanjie 2008
    modified by Z. Li, Aug. 09, 2009
    """
    cdef np.ndarray[np.double_t, ndim=2] gam = np.array([[np.cos(alpha), 0, 0, 0, -np.sin(alpha), 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, np.cos(alpha), 0, 0, 0, -np.sin(alpha), 0], [0, 0, 0, 1, 0, 0, 0, 0],
                    [np.sin(alpha), 0, 0, 0, np.cos(alpha), 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, np.sin(alpha), 0, 0, 0, np.cos(alpha), 0], [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.double)

    if total_m == 1:
        return gam

    cdef np.ndarray[np.double_t, ndim=2] gamma = np.zeros((8 * total_m, 8 * total_m), dtype=np.double)
    # extend to multi-m
    cdef int i
    for i in range(0, total_m):
        gamma[8 * i:8 * (i+1), 8 * i:8 * (i+1)] = gam

    return gamma


cdef assemble(
    np.ndarray K_global, np.ndarray Kg_global, np.ndarray k_local, np.ndarray kg_local, 
    int node_i, int node_j, int n_nodes, np.ndarray m_a
):
    """Add the element contribution to the global stiffness matrix

    Args:
        K_global (np.ndarray): global elastic stiffness matrix
        Kg_global (np.ndarray): global geometric stiffness matrix
        k_local (np.ndarray): local elastic stiffness matrix. Each submatrix is similar to the
            one used in original CUFSM for single longitudinal term m in the DOF order
            [u1 v1...un vn w1 01...wn 0n]m'.
        kg_local (np.ndarray): local geometric stiffness matrix. Each submatrix is similar to the
            one used in original CUFSM for single longitudinal term m in the DOF order 
            [u1 v1...un vn w1 01...wn 0n]m'.
        node_i (int): node number
        node_j (int): node number
        n_nodes (int): total number of nodes in section
        m_a (np.ndarray): numbers of half-wavelengths

    Returns:
        K_global (np.ndarray): global elastic stiffness matrix
        Kg_global (np.ndarray): global geometric stiffness matrix

    Z. Li, June 2008
    modified by Z. Li, Aug. 09, 2009
    Z. Li, June 2010
    """
    cdef int total_m = len(m_a)  # Total number of longitudinal terms m
    cdef int skip = 2 * n_nodes

    # declare loop variables
    cdef int i
    cdef int j
    cdef np.ndarray[np.double_t, ndim=2] k11
    cdef np.ndarray[np.double_t, ndim=2] k12
    cdef np.ndarray[np.double_t, ndim=2] k13
    cdef np.ndarray[np.double_t, ndim=2] k14
    cdef np.ndarray[np.double_t, ndim=2] k21
    cdef np.ndarray[np.double_t, ndim=2] k22
    cdef np.ndarray[np.double_t, ndim=2] k23
    cdef np.ndarray[np.double_t, ndim=2] k24
    cdef np.ndarray[np.double_t, ndim=2] k31
    cdef np.ndarray[np.double_t, ndim=2] k32
    cdef np.ndarray[np.double_t, ndim=2] k33
    cdef np.ndarray[np.double_t, ndim=2] k34
    cdef np.ndarray[np.double_t, ndim=2] k41
    cdef np.ndarray[np.double_t, ndim=2] k42
    cdef np.ndarray[np.double_t, ndim=2] k43
    cdef np.ndarray[np.double_t, ndim=2] k44
    cdef np.ndarray[np.double_t, ndim=2] kg11
    cdef np.ndarray[np.double_t, ndim=2] kg12
    cdef np.ndarray[np.double_t, ndim=2] kg13
    cdef np.ndarray[np.double_t, ndim=2] kg14
    cdef np.ndarray[np.double_t, ndim=2] kg21
    cdef np.ndarray[np.double_t, ndim=2] kg22
    cdef np.ndarray[np.double_t, ndim=2] kg23
    cdef np.ndarray[np.double_t, ndim=2] kg24
    cdef np.ndarray[np.double_t, ndim=2] kg31
    cdef np.ndarray[np.double_t, ndim=2] kg32
    cdef np.ndarray[np.double_t, ndim=2] kg33
    cdef np.ndarray[np.double_t, ndim=2] kg34
    cdef np.ndarray[np.double_t, ndim=2] kg41
    cdef np.ndarray[np.double_t, ndim=2] kg42
    cdef np.ndarray[np.double_t, ndim=2] kg43
    cdef np.ndarray[np.double_t, ndim=2] kg44

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

            K_global[4*n_nodes*i+(node_i+1)*2-2:4*n_nodes*i+(node_i+1)*2, \
                4*n_nodes*j+(node_i+1)*2-2:4*n_nodes*j+(node_i+1)*2] += k11
            K_global[4*n_nodes*i+(node_i+1)*2-2:4*n_nodes*i+(node_i+1)*2, \
                4*n_nodes*j+(node_j+1)*2-2:4*n_nodes*j+(node_j+1)*2] += k12
            K_global[4*n_nodes*i+(node_j+1)*2-2:4*n_nodes*i+(node_j+1)*2, \
                4*n_nodes*j+(node_i+1)*2-2:4*n_nodes*j+(node_i+1)*2] += k21
            K_global[4*n_nodes*i+(node_j+1)*2-2:4*n_nodes*i+(node_j+1)*2, \
                4*n_nodes*j+(node_j+1)*2-2:4*n_nodes*j+(node_j+1)*2] += k22

            K_global[4*n_nodes*i+skip+(node_i+1)*2-2:4*n_nodes*i+skip+(node_i+1)*2, \
                4*n_nodes*j+skip+(node_i+1)*2-2:4*n_nodes*j+skip+(node_i+1)*2] += k33
            K_global[4*n_nodes*i+skip+(node_i+1)*2-2:4*n_nodes*i+skip+(node_i+1)*2, \
                4*n_nodes*j+skip+(node_j+1)*2-2:4*n_nodes*j+skip+(node_j+1)*2] += k34
            K_global[4*n_nodes*i+skip+(node_j+1)*2-2:4*n_nodes*i+skip+(node_j+1)*2, \
                4*n_nodes*j+skip+(node_i+1)*2-2:4*n_nodes*j+skip+(node_i+1)*2] += k43
            K_global[4*n_nodes*i+skip+(node_j+1)*2-2:4*n_nodes*i+skip+(node_j+1)*2, \
                4*n_nodes*j+skip+(node_j+1)*2-2:4*n_nodes*j+skip+(node_j+1)*2] += k44

            K_global[4*n_nodes*i+(node_i+1)*2-2:4*n_nodes*i+(node_i+1)*2, \
                4*n_nodes*j+skip+(node_i+1)*2-2:4*n_nodes*j+skip+(node_i+1)*2] += k13
            K_global[4*n_nodes*i+(node_i+1)*2-2:4*n_nodes*i+(node_i+1)*2, \
                4*n_nodes*j+skip+(node_j+1)*2-2:4*n_nodes*j+skip+(node_j+1)*2] += k14
            K_global[4*n_nodes*i+(node_j+1)*2-2:4*n_nodes*i+(node_j+1)*2, \
                4*n_nodes*j+skip+(node_i+1)*2-2:4*n_nodes*j+skip+(node_i+1)*2] += k23
            K_global[4*n_nodes*i+(node_j+1)*2-2:4*n_nodes*i+(node_j+1)*2, \
                4*n_nodes*j+skip+(node_j+1)*2-2:4*n_nodes*j+skip+(node_j+1)*2] += k24

            K_global[4*n_nodes*i+skip+(node_i+1)*2-2:4*n_nodes*i+skip+(node_i+1)*2, \
                4*n_nodes*j+(node_i+1)*2-2:4*n_nodes*j+(node_i+1)*2] += k31
            K_global[4*n_nodes*i+skip+(node_i+1)*2-2:4*n_nodes*i+skip+(node_i+1)*2, \
                4*n_nodes*j+(node_j+1)*2-2:4*n_nodes*j+(node_j+1)*2] += k32
            K_global[4*n_nodes*i+skip+(node_j+1)*2-2:4*n_nodes*i+skip+(node_j+1)*2, \
                4*n_nodes*j+(node_i+1)*2-2:4*n_nodes*j+(node_i+1)*2] += k41
            K_global[4*n_nodes*i+skip+(node_j+1)*2-2:4*n_nodes*i+skip+(node_j+1)*2, \
                4*n_nodes*j+(node_j+1)*2-2:4*n_nodes*j+(node_j+1)*2] += k42

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

            Kg_global[4*n_nodes*i + (node_i+1) * 2 - 2:4*n_nodes*i + (node_i+1) * 2,
                      4*n_nodes*j + (node_i+1) * 2 - 2:4*n_nodes*j + (node_i+1) * 2] += kg11
            Kg_global[4*n_nodes*i + (node_i+1) * 2 - 2:4*n_nodes*i + (node_i+1) * 2,
                      4*n_nodes*j + (node_j+1) * 2 - 2:4*n_nodes*j + (node_j+1) * 2] += kg12
            Kg_global[4*n_nodes*i + (node_j+1) * 2 - 2:4*n_nodes*i + (node_j+1) * 2,
                      4*n_nodes*j + (node_i+1) * 2 - 2:4*n_nodes*j + (node_i+1) * 2] += kg21
            Kg_global[4*n_nodes*i + (node_j+1) * 2 - 2:4*n_nodes*i + (node_j+1) * 2,
                      4*n_nodes*j + (node_j+1) * 2 - 2:4*n_nodes*j + (node_j+1) * 2] += kg22

            Kg_global[4*n_nodes*i + skip + (node_i+1) * 2 - 2:4*n_nodes*i + skip + (node_i+1) * 2,
                      4*n_nodes*j + skip + (node_i+1) * 2 - 2:4*n_nodes*j + skip
                      + (node_i+1) * 2] += kg33
            Kg_global[4*n_nodes*i + skip + (node_i+1) * 2 - 2:4*n_nodes*i + skip + (node_i+1) * 2,
                      4*n_nodes*j + skip + (node_j+1) * 2 - 2:4*n_nodes*j + skip
                      + (node_j+1) * 2] += kg34
            Kg_global[4*n_nodes*i + skip + (node_j+1) * 2 - 2:4*n_nodes*i + skip + (node_j+1) * 2,
                      4*n_nodes*j + skip + (node_i+1) * 2 - 2:4*n_nodes*j + skip
                      + (node_i+1) * 2] += kg43
            Kg_global[4*n_nodes*i + skip + (node_j+1) * 2 - 2:4*n_nodes*i + skip + (node_j+1) * 2,
                      4*n_nodes*j + skip + (node_j+1) * 2 - 2:4*n_nodes*j + skip
                      + (node_j+1) * 2] += kg44

            Kg_global[4*n_nodes*i + (node_i+1) * 2 - 2:4*n_nodes*i + (node_i+1) * 2, 4*n_nodes*j
                      + skip + (node_i+1) * 2 - 2:4*n_nodes*j + skip + (node_i+1) * 2] += kg13
            Kg_global[4*n_nodes*i + (node_i+1) * 2 - 2:4*n_nodes*i + (node_i+1) * 2, 4*n_nodes*j
                      + skip + (node_j+1) * 2 - 2:4*n_nodes*j + skip + (node_j+1) * 2] += kg14
            Kg_global[4*n_nodes*i + (node_j+1) * 2 - 2:4*n_nodes*i + (node_j+1) * 2, 4*n_nodes*j
                      + skip + (node_i+1) * 2 - 2:4*n_nodes*j + skip + (node_i+1) * 2] += kg23
            Kg_global[4*n_nodes*i + (node_j+1) * 2 - 2:4*n_nodes*i + (node_j+1) * 2, 4*n_nodes*j
                      + skip + (node_j+1) * 2 - 2:4*n_nodes*j + skip + (node_j+1) * 2] += kg24

            Kg_global[4*n_nodes*i + skip + (node_i+1) * 2 - 2:4*n_nodes*i + skip + (node_i+1) * 2,
                      4*n_nodes*j + (node_i+1) * 2 - 2:4*n_nodes*j + (node_i+1) * 2] += kg31
            Kg_global[4*n_nodes*i + skip + (node_i+1) * 2 - 2:4*n_nodes*i + skip + (node_i+1) * 2,
                      4*n_nodes*j + (node_j+1) * 2 - 2:4*n_nodes*j + (node_j+1) * 2] += kg32
            Kg_global[4*n_nodes*i + skip + (node_j+1) * 2 - 2:4*n_nodes*i + skip + (node_j+1) * 2,
                      4*n_nodes*j + (node_i+1) * 2 - 2:4*n_nodes*j + (node_i+1) * 2] += kg41
            Kg_global[4*n_nodes*i + skip + (node_j+1) * 2 - 2:4*n_nodes*i + skip + (node_j+1) * 2,
                      4*n_nodes*j + (node_j+1) * 2 - 2:4*n_nodes*j + (node_j+1) * 2] += kg42

    return K_global, Kg_global


cdef assemble_single(
    np.ndarray K_global, np.ndarray k_local, int node_i, int node_j, int n_nodes
):
    """this routine adds the element contribution to the global stiffness matrix
    basically it does the same as routine 'assemble', however:
    it does not care about Kg_global (geom stiff matrix)
    only involves single half-wave number m_i

    Args:
        K_global (np.ndarray): global elastic stiffness matrix
        k_local (np.ndarray): local elastic stiffness matrix
        node_i (int): node number
        node_j (int): node number
        n_nodes (int): total number of nodes in section

    Returns:
        _type_: _description_

    S. Adany, Feb 06, 2004
    Z. Li, Jul 10, 2009
    """
    # submatrices for the initial stiffness
    cdef np.ndarray[np.double_t, ndim=2] k11 = k_local[0:2, 0:2]
    cdef np.ndarray[np.double_t, ndim=2] k12 = k_local[0:2, 2:4]
    cdef np.ndarray[np.double_t, ndim=2] k13 = k_local[0:2, 4:6]
    cdef np.ndarray[np.double_t, ndim=2] k14 = k_local[0:2, 6:8]
    cdef np.ndarray[np.double_t, ndim=2] k21 = k_local[2:4, 0:2]
    cdef np.ndarray[np.double_t, ndim=2] k22 = k_local[2:4, 2:4]
    cdef np.ndarray[np.double_t, ndim=2] k23 = k_local[2:4, 4:6]
    cdef np.ndarray[np.double_t, ndim=2] k24 = k_local[2:4, 6:8]
    cdef np.ndarray[np.double_t, ndim=2] k31 = k_local[4:6, 0:2]
    cdef np.ndarray[np.double_t, ndim=2] k32 = k_local[4:6, 2:4]
    cdef np.ndarray[np.double_t, ndim=2] k33 = k_local[4:6, 4:6]
    cdef np.ndarray[np.double_t, ndim=2] k34 = k_local[4:6, 6:8]
    cdef np.ndarray[np.double_t, ndim=2] k41 = k_local[6:8, 0:2]
    cdef np.ndarray[np.double_t, ndim=2] k42 = k_local[6:8, 2:4]
    cdef np.ndarray[np.double_t, ndim=2] k43 = k_local[6:8, 4:6]
    cdef np.ndarray[np.double_t, ndim=2] k44 = k_local[6:8, 6:8]

    # the additional terms for K_global are stored in k_2_matrix
    cdef int skip = 2 * n_nodes
    K_global[node_i * 2:node_i*2 + 2, node_i * 2:node_i*2 + 2] += k11
    K_global[node_i * 2:node_i*2 + 2, node_j * 2:node_j*2 + 2] += k12
    K_global[node_j * 2:node_j*2 + 2, node_i * 2:node_i*2 + 2] += k21
    K_global[node_j * 2:node_j*2 + 2, node_j * 2:node_j*2 + 2] += k22

    K_global[skip + node_i*2:skip + node_i*2 + 2, skip + node_i*2:skip + node_i*2 + 2] += k33
    K_global[skip + node_i*2:skip + node_i*2 + 2, skip + node_j*2:skip + node_j*2 + 2] += k34
    K_global[skip + node_j*2:skip + node_j*2 + 2, skip + node_i*2:skip + node_i*2 + 2] += k43
    K_global[skip + node_j*2:skip + node_j*2 + 2, skip + node_j*2:skip + node_j*2 + 2] += k44

    K_global[node_i * 2:node_i*2 + 2, skip + node_i*2:skip + node_i*2 + 2] += k13
    K_global[node_i * 2:node_i*2 + 2, skip + node_j*2:skip + node_j*2 + 2] += k14
    K_global[node_j * 2:node_j*2 + 2, skip + node_i*2:skip + node_i*2 + 2] += k23
    K_global[node_j * 2:node_j*2 + 2, skip + node_j*2:skip + node_j*2 + 2] += k24

    K_global[skip + node_i*2:skip + node_i*2 + 2, node_i * 2:node_i*2 + 2] += k31
    K_global[skip + node_i*2:skip + node_i*2 + 2, node_j * 2:node_j*2 + 2] += k32
    K_global[skip + node_j*2:skip + node_j*2 + 2, node_i * 2:node_i*2 + 2] += k41
    K_global[skip + node_j*2:skip + node_j*2 + 2, node_j * 2:node_j*2 + 2] += k42

    return K_global


cpdef np.ndarray spring_klocal(
    double k_u, double k_v, double k_w, double k_q, double length, 
    str B_C, np.ndarray m_a, int discrete, double y_s
):
    """Generate spring stiffness matrix (k_local) in local coordinates, modified from
    klocal

    Args:
        k_u (float): spring stiffness values
        k_v (float): spring stiffness values
        k_w (float): spring stiffness values
        k_q (float): spring stiffness values
        length (float): length of the strip in longitudinal direction
        B_C (str): ['S-S'] a string specifying boundary conditions to be analyzed:
            'S-S' simply-pimply supported boundary condition at loaded edges
            'C-C' clamped-clamped boundary condition at loaded edges
            'S-C' simply-clamped supported boundary condition at loaded edges
            'C-F' clamped-free supported boundary condition at loaded edges
            'C-G' clamped-guided supported boundary condition at loaded edges
        m_a (np.ndarray): longitudinal terms (or half-wave numbers) for this length
        discrete (int): discrete == 1 if discrete spring
        y_s (float): location of discrete spring

    Returns:
        klocal (np.ndarray): local stiffness matrix, a total_m x total_m matrix of 8 by 8 
            submatrices. k_local=[k_mp]total_m x total_m block matrix
            each k_mp is the 8 x 8 submatrix in the DOF order [u1 v1 u2 v2 w1 theta1 w2 theta2]'

    BWS DEC 2015
    """
    cdef int total_m = len(m_a)  # Total number of longitudinal terms m
    cdef np.ndarray[np.double_t, ndim=2] k_local = np.zeros((8 * total_m, 8 * total_m))

    # Declare looping variables
    cdef int i
    cdef int j
    cdef np.ndarray[np.double_t, ndim=2] km_mp
    cdef np.ndarray[np.double_t, ndim=2] kf_mp
    cdef double u_i
    cdef double u_j

    for i in range(0, total_m):
        for j in range(0, total_m):
            km_mp = np.zeros((4, 4))
            kf_mp = np.zeros((4, 4))
            u_i = m_a[i] * np.pi
            u_j = m_a[j] * np.pi

            if discrete:
                [I_1, I_5] = bc_i1_5_atpoint(
                    B_C=B_C, m_i=m_a[i], m_j=m_a[j], length=length, y_s=y_s
                )
            else:  # foundation spring
                [I_1, _, _, _, I_5] = bc_i1_5(B_C=B_C, m_i=m_a[i], m_j=m_a[j], length=length)
            # assemble the matrix of km_mp (membrane stiffness)
            km_mp = np.array(
                [[k_u * I_1, 0, -k_u * I_1, 0],
                 [0, k_v * I_5 * length**2 / (u_i*u_j), 0, -k_v * I_5 * length**2 / (u_i*u_j)],
                 [-k_u * I_1, 0, k_u * I_1, 0],
                 [0, -k_v * I_5 * length**2 / (u_i*u_j), 0, k_v * I_5 * length**2 / (u_i*u_j)]]
            )
            # assemble the matrix of kf_mp (flexural stiffness)
            kf_mp = np.array([[k_w * I_1, 0, -k_w * I_1, 0], [0, k_q * I_1, 0, -k_q * I_1],
                              [-k_w * I_1, 0, k_w * I_1, 0], [0, -k_q * I_1, 0, k_q * I_1]])

            k_local[8 * i:8*i + 4, 8 * j:8*j + 4] = km_mp
            k_local[8*i + 4:8 * (i+1), 8*j + 4:8 * (j+1)] = kf_mp

    return k_local


cdef bc_i1_5_atpoint(str B_C, double m_i, double m_j, double length, double y_s):
    """Calculate the value of the longitudinal shape functions for discrete springs


    Args:
        B_C (str): a string specifying boundary conditions to be analyzed:
            'S-S' simply-pimply supported boundary condition at loaded edges
            'C-C' clamped-clamped boundary condition at loaded edges
            'S-C' simply-clamped supported boundary condition at loaded edges
            'C-F' clamped-free supported boundary condition at loaded edges
            'C-G' clamped-guided supported boundary condition at loaded edges
        m_i (float): number of half-wavelengths
        m_j (float): number of half-wavelengths
        length (float): length of element
        y_s (float): location of discrete spring

    Returns:
        I_1 (float): calculation of I_1 is the value of y_m(y/L)*Yn(y/L)
        I_5 (float): calculation of I_5 is the value of y_m'(y/L)*Yn'(y/L)_description_
    """
    cdef double y_i = ym_at_ys(B_C=B_C, m_i=m_i, y_s=y_s, length=length)
    cdef double y_j = ym_at_ys(B_C=B_C, m_i=m_j, y_s=y_s, length=length)
    cdef double y_i_prime = ymprime_at_ys(B_C=B_C, m_i=m_i, y_s=y_s, length=length)
    cdef double y_j_prime = ymprime_at_ys(B_C=B_C, m_i=m_j, y_s=y_s, length=length)
    cdef double I_1 = y_i * y_j
    cdef double I_5 = y_i_prime * y_j_prime
    return I_1, I_5


cpdef np.ndarray spring_assemble(
    np.ndarray K_global, np.ndarray k_local, int node_i, int node_j, int n_nodes, 
    np.ndarray m_a
):
    """Add the (spring) contribution to the global stiffness matrix

    Args:
        K_global (np.ndarray): global elastic stiffness matrix 
            total_m x total_m submatrices. Each submatrix is similar to the
            one used in original CUFSM for single longitudinal term m in the DOF order
            [u1 v1...un vn w1 01...wn 0n]m'.
        k_local (np.ndarray): local elastic stiffness matrix
        node_i (int): node number
        node_j (int): node number
        n_nodes (int): total number of nodes 
        m_a (np.ndarray): number of half-wavelengths

    Returns:
        K_global (np.ndarray): global elastic stiffness matrix

    Z. Li, June 2008
    modified by Z. Li, Aug. 09, 2009
    Z. Li, June 2010
    adapted for springs BWS Dec 2015
    """
    cdef int total_m = len(m_a)  # Total number of longitudinal terms m
    cdef int skip = 2 * n_nodes

    # Declare looping variables
    cdef int i
    cdef int j
    cdef np.ndarray[np.double_t, ndim=2] k11 = k_local[0:2, 0:2]
    cdef np.ndarray[np.double_t, ndim=2] k12 = k_local[0:2, 2:4]
    cdef np.ndarray[np.double_t, ndim=2] k13 = k_local[0:2, 4:6]
    cdef np.ndarray[np.double_t, ndim=2] k14 = k_local[0:2, 6:8]
    cdef np.ndarray[np.double_t, ndim=2] k21 = k_local[2:4, 0:2]
    cdef np.ndarray[np.double_t, ndim=2] k22 = k_local[2:4, 2:4]
    cdef np.ndarray[np.double_t, ndim=2] k23 = k_local[2:4, 4:6]
    cdef np.ndarray[np.double_t, ndim=2] k24 = k_local[2:4, 6:8]
    cdef np.ndarray[np.double_t, ndim=2] k31 = k_local[4:6, 0:2]
    cdef np.ndarray[np.double_t, ndim=2] k32 = k_local[4:6, 2:4]
    cdef np.ndarray[np.double_t, ndim=2] k33 = k_local[4:6, 4:6]
    cdef np.ndarray[np.double_t, ndim=2] k34 = k_local[4:6, 6:8]
    cdef np.ndarray[np.double_t, ndim=2] k41 = k_local[6:8, 0:2]
    cdef np.ndarray[np.double_t, ndim=2] k42 = k_local[6:8, 2:4]
    cdef np.ndarray[np.double_t, ndim=2] k43 = k_local[6:8, 4:6]
    cdef np.ndarray[np.double_t, ndim=2] k44 = k_local[6:8, 6:8]

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

            K_global[4*n_nodes*i + (node_i+1) * 2 - 1:4*n_nodes*i + (node_i+1) * 2,
                     4*n_nodes*j + (node_i+1) * 2 - 1:4*n_nodes*j + (node_i+1) * 2] += k11
            if node_j != -1:
                K_global[4*n_nodes*i + (node_i+1) * 2 - 1:4*n_nodes*i + (node_i+1) * 2,
                         4*n_nodes*j + (node_j+1) * 2 - 1:4*n_nodes*j + (node_j+1) * 2] += k12
                K_global[4*n_nodes*i + (node_j+1) * 2 - 1:4*n_nodes*i + (node_j+1) * 2,
                         4*n_nodes*j + (node_i+1) * 2 - 1:4*n_nodes*j + (node_i+1) * 2] += k21
                K_global[4*n_nodes*i + (node_j+1) * 2 - 1:4*n_nodes*i + (node_j+1) * 2,
                         4*n_nodes*j + (node_j+1) * 2 - 1:4*n_nodes*j + (node_j+1) * 2] += k22

            K_global[4*n_nodes*i + skip + (node_i+1) * 2 - 1:4*n_nodes*i + skip + (node_i+1) * 2,
                     4*n_nodes*j + skip + (node_i+1) * 2 - 1:4*n_nodes*j + skip
                     + (node_i+1) * 2] += k33
            if node_j != -1:
                K_global[4*n_nodes*i + skip + (node_i+1) * 2 - 1:4*n_nodes*i + skip
                         + (node_i+1) * 2, 4*n_nodes*j + skip + (node_j+1) * 2 - 1:4*n_nodes*j
                         + skip + (node_j+1) * 2] += k34
                K_global[4*n_nodes*i + skip + (node_j+1) * 2 - 1:4*n_nodes*i + skip
                         + (node_j+1) * 2, 4*n_nodes*j + skip + (node_i+1) * 2 - 1:4*n_nodes*j
                         + skip + (node_i+1) * 2] += k43
                K_global[4*n_nodes*i + skip + (node_j+1) * 2 - 1:4*n_nodes*i + skip
                         + (node_j+1) * 2, 4*n_nodes*j + skip + (node_j+1) * 2 - 1:4*n_nodes*j
                         + skip + (node_j+1) * 2] += k44

            K_global[4*n_nodes*i + (node_i+1) * 2 - 1:4*n_nodes*i + (node_i+1) * 2, 4*n_nodes*j
                     + skip + (node_i+1) * 2 - 1:4*n_nodes*j + skip + (node_i+1) * 2] += k13
            if node_j != -1:
                K_global[4*n_nodes*i + (node_i+1) * 2 - 1:4*n_nodes*i + (node_i+1) * 2, 4*n_nodes*j
                         + skip + (node_j+1) * 2 - 1:4*n_nodes*j + skip + (node_j+1) * 2] += k14
                K_global[4*n_nodes*i + (node_j+1) * 2 - 1:4*n_nodes*i + (node_j+1) * 2, 4*n_nodes*j
                         + skip + (node_i+1) * 2 - 1:4*n_nodes*j + skip + (node_i+1) * 2] += k23
                K_global[4*n_nodes*i + (node_j+1) * 2 - 1:4*n_nodes*i + (node_j+1) * 2, 4*n_nodes*j
                         + skip + (node_j+1) * 2 - 1:4*n_nodes*j + skip + (node_j+1) * 2] += k24

            K_global[4*n_nodes*i + skip + (node_i+1) * 2 - 1:4*n_nodes*i + skip + (node_i+1) * 2,
                     4*n_nodes*j + (node_i+1) * 2 - 1:4*n_nodes*j + (node_i+1) * 2] += k31
            if node_j != -1:
                K_global[4*n_nodes*i + skip + (node_i+1) * 2 - 1:4*n_nodes*i + skip
                         + (node_i+1) * 2,
                         4*n_nodes*j + (node_j+1) * 2 - 1:4*n_nodes*j + (node_j+1) * 2] += k32
                K_global[4*n_nodes*i + skip + (node_j+1) * 2 - 1:4*n_nodes*i + skip
                         + (node_j+1) * 2,
                         4*n_nodes*j + (node_i+1) * 2 - 1:4*n_nodes*j + (node_i+1) * 2] += k41
                K_global[4*n_nodes*i + skip + (node_j+1) * 2 - 1:4*n_nodes*i + skip
                         + (node_j+1) * 2,
                         4*n_nodes*j + (node_j+1) * 2 - 1:4*n_nodes*j + (node_j+1) * 2] += k42

    return K_global


cpdef double ym_at_ys(str B_C, double m_i, double y_s, double length):
    """Longitudinal shape function values
    could be called in lots of places,  but now (2015) is hardcoded by Zhanjie
    in several places in the interface
    written in 2015 because wanted it for a new idea on discrete springs

    Args:
        B_C (str): boundary condition
        m_i (float): number of half-wavelengths
        y_s (float): location of discrete spring
        length (float): element length

    Returns:
        y_m (float): longitudinal shape function value
    
    BWS in 2015
    """
    cdef double y_m
    if B_C == 'S-S':
        y_m = np.sin(m_i * np.pi * y_s / length)
    elif B_C == 'C-C':
        y_m = np.sin(m_i * np.pi * y_s / length) * np.sin(np.pi * y_s / length)
    elif B_C == 'S-C' or B_C == 'C-S':
        y_m = np.sin((m_i+1) * np.pi * y_s / length
                     ) + (m_i+1) / m_i * np.sin(m_i * np.pi * y_s / length)
    elif B_C == 'C-F' or B_C == 'F-C':
        y_m = 1 - np.cos((m_i-0.5) * np.pi * y_s / length)
    elif B_C == 'C-G' or B_C == 'G-C':
        y_m = np.sin((m_i-0.5) * np.pi * y_s / length) * np.sin(np.pi * y_s / length / 2)
    else:
        raise ValueError(f"Unrecognised boundary condition '{B_C}'")

    return y_m


cpdef double ymprime_at_ys(str B_C, double m_i, double y_s, double length):
    """First Derivative of Longitudinal shape function values
    could be called in lots of places,  but now (2015) is hardcoded by Zhanjie
    in several places in the interface
    written in 2015 because wanted it for a new idea on discrete springs

    Args:
        B_C (str): boundary condition
        m_i (float): number of half-wavelengths
        y_s (float): location of discrete spring
        length (float): element length

    Returns:
        y_m_prime (float): first derivative of longitudinal shape function

    BWS in 2015
    """
    cdef double y_m_prime
    if B_C == 'S-S':
        y_m_prime = (np.pi * m_i * np.cos((np.pi * m_i * y_s) / length)) / length
    elif B_C == 'C-C':
        y_m_prime = (np.pi * np.cos((np.pi*y_s) / length) \
            * np.sin((np.pi*m_i*y_s) / length)) / length \
            + (np.pi*m_i * np.sin((np.pi*y_s)/length) \
                * np.cos((np.pi*m_i*y_s)/length)) / length
    elif B_C == 'S-C' or B_C == 'C-S':
        y_m_prime = (np.pi * np.cos((np.pi*y_s * (m_i + 1))/length) * (m_i + 1)) / length \
            + (np.pi * np.cos((np.pi*m_i*y_s)/length)*(m_i + 1)) / length
    elif B_C == 'C-F' or B_C == 'F-C':
        y_m_prime = (np.pi * np.sin((np.pi * y_s * (m_i - 1/2)) / length) * (m_i - 1/2)) / length
    elif B_C == 'C-G' or B_C == 'G-C':
        y_m_prime = (np.pi*np.sin((np.pi*y_s * (m_i - 1/2))/length) \
            * np.cos((np.pi*y_s)/(2*length)))/(2*length) \
            + (np.pi*np.cos((np.pi*y_s*(m_i - 1/2))/length) \
            * np.sin((np.pi*y_s)/(2*length))*(m_i - 1/2))/length
    else:
        raise ValueError(f"Unrecognised boundary condition '{B_C}'")

    return y_m_prime
