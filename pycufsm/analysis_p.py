from typing import Tuple

import numpy as np

# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE, CPEng
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


def m_sort(m_all: np.ndarray) -> np.ndarray:
    """Cleans up longitudinal terms by removing any zero terms, removing any duplicates, and sorting

    Args:
        m_all (np.ndarray): list of all longitudinal terms for each half-wavelength

    Returns:
        m_all (np.ndarray): cleaned list of all longitudinal terms for each half-wavelength
    """
    for i, m_a in enumerate(m_all):
        # return all the nonzero longitudinal terms in m_a as a column vector
        m_a = m_a[np.nonzero(m_a)]
        m_a = np.lib.unique(m_a)  # remove repetitive longitudinal terms
        m_all[i] = m_a
    return m_all


def constr_bc_flag(nodes: np.ndarray, constraints: np.ndarray) -> int:
    """this subroutine is to determine flags for user constraints and internal (at node) B.C.'s

    Args:
        nodes (np.ndarray): [node# x y dof_x dof_y dof_z dof_r stress] n_nodes x 8
        constraints (np.ndarray): [node# e dof_e coeff node# k dof_k] e=dof to be eliminated
            k=kept dof dofe_node = coeff*dofk_nodek

    Returns:
        bc_flag (int): 1 if there are user constraints or node fixities
            0 if there is no user constraints and node fixities
    
    Z. Li, June 2010
    """
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


def elem_prop(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """creates a matrix of element properties for each element (width and slope)

    Args:
        nodes (np.ndarray): standard nodes array
        elements (np.ndarray): standard elements array

    Returns:
        el_props (np.ndarray): [id, width, alpha]
    """
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


def k_kg_global(
    nodes: np.ndarray, elements: np.ndarray, el_props: np.ndarray, props: np.ndarray, length: float,
    b_c: str, m_a: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates element stiffness matrix (k_global) in global coordinates
    Generate geometric stiffness matrix (kg_global) in global coordinates

    Args:
        nodes (np.ndarray): 
        elements (np.ndarray): 
        el_props (np.ndarray): 
        props (np.ndarray): 
        length (np.ndarray): 
        b_c (str): 
        m_a (np.ndarray): 

    Returns:
        k_global (np.ndarray): global stiffness matrix
        kg_global (np.ndarray): global geometric stiffness matrix

    B Smith, May 2023
    """
    total_m = len(m_a)
    n_nodes = len(nodes)
    n_elems = len(elements)
    # ZERO OUT THE GLOBAL MATRICES
    k_global = np.zeros((4 * n_nodes * total_m, 4 * n_nodes * total_m))
    kg_global = np.zeros((4 * n_nodes * total_m, 4 * n_nodes * total_m))

    # ASSEMBLE THE GLOBAL STIFFNESS MATRICES
    for i in range(0, n_elems):
        # Generate element stiffness matrix (k_local) in local coordinates
        # Generate geometric stiffness matrix (kg_local) in local coordinates
        thick = elements[i, 3]
        b_strip = el_props[i, 1]
        mat_num = int(elements[i, 4])
        row = int((np.argwhere(props[:, 0] == mat_num)).reshape(1))
        stiff_x = props[row, 1]
        stiff_y = props[row, 2]
        nu_x = props[row, 3]
        nu_y = props[row, 4]
        bulk = props[row, 5]

        node_i = int(elements[i, 1])
        node_j = int(elements[i, 2])
        ty_1 = nodes[node_i, 7] * thick
        ty_2 = nodes[node_j, 7] * thick

        k_l, kg_l = k_kg_local(
            stiff_x=stiff_x,
            stiff_y=stiff_y,
            nu_x=nu_x,
            nu_y=nu_y,
            bulk=bulk,
            thick=thick,
            length=length,
            ty_1=ty_1,
            ty_2=ty_2,
            b_strip=b_strip,
            b_c=b_c,
            m_a=m_a
        )

        # Transform k_local and kg_local into global coordinates
        alpha = el_props[i, 2]
        gamma = trans(alpha=alpha, total_m=total_m)
        k_local = gamma @ k_l @ gamma.conj().T
        kg_local = gamma @ kg_l @ gamma.conj().T

        # Add element contribution of k_local to full matrix k_global and kg_local to kg_global
        k_global, kg_global = assemble(
            k_global=k_global,
            kg_global=kg_global,
            k_local=k_local,
            kg_local=kg_local,
            node_i=node_i,
            node_j=node_j,
            n_nodes=n_nodes,
            m_a=m_a
        )

    return k_global, kg_global


def k_kg_local(
    stiff_x: float, stiff_y: float, nu_x: float, nu_y: float, bulk: float, thick: float,
    length: float, ty_1: float, ty_2: float, b_strip: float, b_c: str, m_a: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate element stiffness matrix (k_local) in local coordinates
    Generate geometric stiffness matrix (kg_local) in local coordinates

    Args:
        stiff_x (float): material property
        stiff_y (float): material property
        nu_x (float): material property
        nu_y (float): material property
        bulk (float): material property
        thick (float): thickness of the strip (element)
        length (np.ndarray): length of the strip in longitudinal direction
        ty_1 (float): node stresses
        ty_2 (float): node stresses
        b_strip (float): width of the strip in transverse direction
        b_c (str): ['S-S'] a string specifying boundary conditions to be analyzed:
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
    e_1 = stiff_x / (1 - nu_x*nu_y)
    e_2 = stiff_y / (1 - nu_x*nu_y)
    d_x = stiff_x * thick**3 / (12 * (1 - nu_x*nu_y))
    d_y = stiff_y * thick**3 / (12 * (1 - nu_x*nu_y))
    d_1 = nu_x * stiff_y * thick**3 / (12 * (1 - nu_x*nu_y))
    d_xy = bulk * thick**3 / 12

    total_m = len(m_a)  # Total number of longitudinal terms m

    k_local = np.zeros((8 * total_m, 8 * total_m))
    kg_local = np.zeros((8 * total_m, 8 * total_m))

    for i in range(0, total_m):
        for j in range(0, total_m):
            u_i = m_a[i] * np.pi
            u_j = m_a[j] * np.pi
            c_1 = u_i / length
            c_2 = u_j / length

            [i_1, i_2, i_3, i_4, i_5] = bc_i1_5(b_c=b_c, m_i=m_a[i], m_j=m_a[j], length=length)

            k_local[8 * i:8*i + 4, 8 * j:8*j + 4] = calc_km_mp(
                e_1=e_1,
                e_2=e_2,
                c_1=c_1,
                c_2=c_2,
                b_strip=b_strip,
                bulk=bulk,
                nu_x=nu_x,
                thick=thick,
                i_1=i_1,
                i_2=i_2,
                i_3=i_3,
                i_4=i_4,
                i_5=i_5
            )
            k_local[8*i + 4:8 * (i+1), 8*j + 4:8 * (j+1)] = calc_kf_mp(
                d_x=d_x,
                d_y=d_y,
                d_1=d_1,
                d_xy=d_xy,
                b_strip=b_strip,
                i_1=i_1,
                i_2=i_2,
                i_3=i_3,
                i_4=i_4,
                i_5=i_5
            )

            kg_local[8 * i:8*i + 4, 8 * j:8*j + 4] = calc_gm_mp(
                u_i=u_i,
                u_j=u_j,
                b_strip=b_strip,
                length=length,
                ty_1=ty_1,
                ty_2=ty_2,
                i_4=i_4,
                i_5=i_5
            )
            kg_local[8*i + 4:8 * (i+1), 8*j + 4:8 * (j+1)] = calc_gf_mp(
                ty_1=ty_1, ty_2=ty_2, b_strip=b_strip, i_5=i_5
            )

    return k_local, kg_local


def kglobal_transv(
    nodes: np.ndarray, elements: np.ndarray, props: np.ndarray, m_i: float, length: float, b_c: str,
    el_props: np.ndarray
) -> np.ndarray:
    """this routine creates the global stiffness matrix for planar displacements
    basically the same way as in the main program, however:
      only one half-wave number m_i is considered,
      only w, teta terms are considered,
      plus stiff_y = nu_x = nu_y = 0 is assumed
      plus the longitudinal displacements. DOFs are explicitly eliminated
      the multiplication by 'length' (member length) is not done here, must be done
         outside of this routine

    Args:
        nodes (np.ndarray): standard parameter
        elements (np.ndarray): standard parameter
        props (np.ndarray): standard parameter
        m_i (float): number of half-wavelengths
        length (float): half-wavelength
        b_c (str): boundary condition
        el_props (np.ndarray): element propertise

    Returns:
        k_global_transv (np.ndarray): global stiffness matrix

    S. Adany, Feb 08, 2004
    Z. Li, Jul 10, 2009
    """
    n_nodes = len(nodes)
    n_elems = len(elements)
    k_global_transv = np.zeros((4 * n_nodes, 4 * n_nodes))

    for i in range(0, n_elems):
        thick = elements[i, 3]
        b_strip = el_props[i, 1]
        mat_num = int(elements[i, 4])
        row = int((np.argwhere(props[:, 0] == mat_num)).reshape(1))
        stiff_x = props[row, 1]
        stiff_y = props[row, 2]
        nu_x = props[row, 3]
        nu_y = props[row, 4]
        bulk = props[row, 5]
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
        gamma = trans(alpha=alpha, total_m=1)
        k_local = gamma @ k_l @ gamma.conj().T

        # Add element contribution of k_local to full matrix k_global and kg_local to kg_global
        node_i = int(elements[i, 1])
        node_j = int(elements[i, 2])
        k_global_transv = assemble_single(
            k_global=k_global_transv,
            k_local=k_local,
            node_i=node_i,
            node_j=node_j,
            n_nodes=n_nodes
        )

    return k_global_transv


def klocal_transv(
    stiff_x: float, stiff_y: float, nu_x: float, nu_y: float, bulk: float, thick: float,
    length: float, b_strip: float, m_i: float, b_c: str
) -> np.ndarray:
    """this routine creates the local stiffness matrix for bending terms
    basically the same way as in the main program, however:
      only for single half-wave number m_i
      membrane strains practically zero, (membrane moduli are enlarged)
      for bending, only transverse terms are considered, (practically: only
      keeps the i_1 term, set i_2 through i_5 to be zero)
        also different from the main program, here only involves one single
        longitudinal term m_i.

    Args:
        stiff_x (float): material property
        stiff_y (float): material property
        nu_x (float): material property
        nu_y (float): material property
        bulk (float): material property
        thick (float): element thickness
        length (float): element length
        b_strip (float): element widget
        m_i (float): number of half-wavelengths
        b_c (str): boundary condition

    Returns:
        np.ndarray: _description_

    Z. Li, Jul 10, 2009
    """
    e_1 = stiff_x / (1 - nu_x*nu_y) * 100000000
    e_2 = stiff_y / (1 - nu_x*nu_y)
    d_x = stiff_x * thick**3 / (12 * (1 - nu_x*nu_y))
    d_y = stiff_y * thick**3 / (12 * (1 - nu_x*nu_y))
    d_1 = nu_x * stiff_y * thick**3 / (12 * (1 - nu_x*nu_y))
    d_xy = bulk * thick**3 / 12

    k_local = np.zeros((8, 8))
    u_m = m_i * np.pi
    u_p = m_i * np.pi
    c_1 = u_m / length
    c_2 = u_p / length

    [i_1, _, _, _, _] = bc_i1_5(b_c=b_c, m_i=m_i, m_j=m_i, length=length)
    i_2 = 0
    i_3 = 0
    i_4 = 0
    i_5 = 0

    k_local[0:4, 0:4] = calc_km_mp(
        e_1=e_1,
        e_2=e_2,
        c_1=c_1,
        c_2=c_2,
        b_strip=b_strip,
        bulk=bulk,
        nu_x=nu_x,
        thick=thick,
        i_1=i_1,
        i_2=i_2,
        i_3=i_3,
        i_4=i_4,
        i_5=i_5
    )
    k_local[4:8, 4:8] = calc_kf_mp(
        d_x=d_x,
        d_y=d_y,
        d_1=d_1,
        d_xy=d_xy,
        b_strip=b_strip,
        i_1=i_1,
        i_2=i_2,
        i_3=i_3,
        i_4=i_4,
        i_5=i_5
    )
    return k_local


def calc_km_mp(
    e_1: float, e_2: float, c_1: float, c_2: float, b_strip: float, bulk: float, nu_x: float,
    thick: float, i_1: float, i_2: float, i_3: float, i_4: float, i_5: float
) -> np.ndarray:
    """Calculate the membrane stiffness sub-matrix, used in the assembly of local stiffness
    matrices

    Args:
        e_1 (float): _description_
        e_2 (float): _description_
        c_1 (float): _description_
        c_2 (float): _description_
        b_strip (float): _description_
        bulk (float): _description_
        nu_x (float): _description_
        thick (float): _description_
        i_1 (float): _description_
        i_2 (float): _description_
        i_3 (float): _description_
        i_4 (float): _description_
        i_5 (float): _description_

    Returns:
        km_mp (np.ndarray): membrane stiffness sub-matrix

    Z. Li June 2008
    modified by Z. Li, Aug. 09, 2009
    modified by Z. Li, June 2010
    Pulled out of klocal() function by B Smith, May 2023
    """
    km_mp = np.zeros((4, 4))

    # assemble the matrix of Km_mp (membrane stiffness matrix)
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

    return km_mp * thick


def calc_kf_mp(
    d_x: float, d_y: float, d_1: float, d_xy: float, b_strip: float, i_1: float, i_2: float,
    i_3: float, i_4: float, i_5: float
) -> np.ndarray:
    """Calculate the flexural stiffness sub-matrix, used in the assembly of local stiffness
    matrices

    Args:
        d_x (float): _description_
        d_y (float): _description_
        d_1 (float): _description_
        d_xy (float): _description_
        b_strip (float): _description_
        i_1 (float): _description_
        i_2 (float): _description_
        i_3 (float): _description_
        i_4 (float): _description_
        i_5 (float): _description_

    Returns:
        np.ndarray: _description_

    Z. Li June 2008
    modified by Z. Li, Aug. 09, 2009
    modified by Z. Li, June 2010
    Pulled out of klocal() function by B Smith, May 2023
    """
    kf_mp = np.zeros((4, 4))

    # assemble the matrix of Kf_mp (flexural stiffness matrix)
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

    return kf_mp


def calc_gm_mp(
    u_i: float, u_j: float, b_strip: float, length: float, ty_1: float, ty_2: float, i_4: float,
    i_5: float
) -> np.ndarray:
    """Calculate the membrane geometric stiffness sub-matrix, used in the assembly of local 
    geometric stiffness matrices

    Args:
        u_i (float): _description_
        u_j (float): _description_
        b_strip (float): _description_
        length (float): _description_
        ty_1 (float): _description_
        ty_2 (float): _description_
        i_4 (float): _description_
        i_5 (float): _description_

    Returns:
        np.ndarray: _description_

    Z. Li June 2008
    modified by Z. Li, Aug. 09, 2009
    modified by Z. Li, June 2010
    Pulled out of kglocal() function by B Smith, May 2023
    """
    gm_mp = np.zeros((4, 4))

    # assemble the matrix of gm_mp (symmetric membrane stability matrix)
    gm_mp[0, 0] = b_strip * (3*ty_1 + ty_2) * i_5 / 12
    gm_mp[0, 2] = b_strip * (ty_1+ty_2) * i_5 / 12
    gm_mp[2, 0] = gm_mp[0, 2]
    gm_mp[1, 1] = b_strip * length**2 * (3*ty_1 + ty_2) * i_4 / 12 / u_i / u_j
    gm_mp[1, 3] = b_strip * length**2 * (ty_1+ty_2) * i_4 / 12 / u_i / u_j
    gm_mp[3, 1] = gm_mp[1, 3]
    gm_mp[2, 2] = b_strip * (ty_1 + 3*ty_2) * i_5 / 12
    gm_mp[3, 3] = b_strip * length**2 * (ty_1 + 3*ty_2) * i_4 / 12 / u_i / u_j

    return gm_mp


def calc_gf_mp(ty_1: float, ty_2: float, b_strip: float, i_5: float) -> np.ndarray:
    """Calculate the flexural geometric stiffness sub-matrix, used in the assembly of local 
    geometric stiffness matrices

    Args:
        ty_1 (float): _description_
        ty_2 (float): _description_
        b_strip (float): _description_
        i_5 (float): _description_

    Returns:
        np.ndarray: _description_

    Z. Li June 2008
    modified by Z. Li, Aug. 09, 2009
    modified by Z. Li, June 2010
    Pulled out of kglocal() function by B Smith, May 2023
    """
    gf_mp = np.zeros((4, 4))

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

    return gf_mp


def bc_i1_5(b_c: str, m_i: float, m_j: float, length: float) -> list:
    """Calculate the 5 undetermined parameters i_1,i_2,i_3,i_4,i_5 for local elastic
    and geometric stiffness matrices.

    Args:
        b_c (str): a string specifying boundary conditions to be analysed:
            'S-S' simply-pimply supported boundary condition at loaded edges
            'C-C' clamped-clamped boundary condition at loaded edges
            'S-C' simply-clamped supported boundary condition at loaded edges
            'C-F' clamped-free supported boundary condition at loaded edges
            'C-G' clamped-guided supported boundary condition at loaded edges
        m_i (float): number of half-wavelengths for node i
        m_j (float): number of half-wavelengths for node j
        length (float): length of element

    Returns:
        i_1 though 5 (list): 5 undetermined parameters i_1,i_2,i_3,i_4,i_5
            calculation of i_1 is the integration of y_m*Yn from 0 to length
            calculation of i_2 is the integration of y_m''*Yn from 0 to length
            calculation of i_3 is the integration of y_m*Yn'' from 0 to length
            calculation of i_3 is the integration of y_m*Yn'' from 0 to length
            calculation of i_4 is the integration of y_m''*Yn'' from 0 to length
            calculation of i_5 is the integration of y_m'*Yn' from 0 to length
    """
    i_1: float = 0
    i_2: float = 0
    i_3: float = 0
    i_4: float = 0
    i_5: float = 0

    if b_c == 'S-S':
        # For simply-pimply supported boundary condition at loaded edges
        if m_i == m_j:
            i_1 = length / 2
            i_2 = -m_i**2 * np.pi**2 / length / 2
            i_3 = -m_j**2 * np.pi**2 / length / 2
            i_4 = np.pi**4 * m_i**4 / 2 / length**3
            i_5 = np.pi**2 * m_i**2 / 2 / length

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

    return [i_1, i_2, i_3, i_4, i_5]


def trans(alpha: float, total_m: int) -> np.ndarray:
    """Transform local stiffness into global stiffness

    Args:
        alpha (float): element angle
        total_m (int): number of half-wavelengths

    Returns:
        gamma (np.ndarray): transformation matrix
    
    Zhanjie 2008
    modified by Z. Li, Aug. 09, 2009
    """
    gam = np.array([[np.cos(alpha), 0, 0, 0, -np.sin(alpha), 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, np.cos(alpha), 0, 0, 0, -np.sin(alpha), 0], [0, 0, 0, 1, 0, 0, 0, 0],
                    [np.sin(alpha), 0, 0, 0, np.cos(alpha), 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, np.sin(alpha), 0, 0, 0, np.cos(alpha), 0], [0, 0, 0, 0, 0, 0, 0, 1]])

    if total_m == 1:
        return gam

    # extend to multi-m
    gamma = np.zeros((8 * total_m, 8 * total_m))
    for i in range(0, total_m):
        gamma[8 * i:8 * (i+1), 8 * i:8 * (i+1)] = gam

    return gamma


def assemble(
    k_global: np.ndarray, kg_global: np.ndarray, k_local: np.ndarray, kg_local: np.ndarray,
    node_i: int, node_j: int, n_nodes: int, m_a: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Add the element contribution to the global stiffness matrix

    Args:
        k_global (np.ndarray): global elastic stiffness matrix
        kg_global (np.ndarray): global geometric stiffness matrix
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
        k_global (np.ndarray): global elastic stiffness matrix
        kg_global (np.ndarray): global geometric stiffness matrix

    Z. Li, June 2008
    modified by Z. Li, Aug. 09, 2009
    Z. Li, June 2010
    """
    total_m = len(m_a)  # Total number of longitudinal terms m
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

            k_global[4*n_nodes*i+(node_i+1)*2-2:4*n_nodes*i+(node_i+1)*2, \
                4*n_nodes*j+(node_i+1)*2-2:4*n_nodes*j+(node_i+1)*2] += k11
            k_global[4*n_nodes*i+(node_i+1)*2-2:4*n_nodes*i+(node_i+1)*2, \
                4*n_nodes*j+(node_j+1)*2-2:4*n_nodes*j+(node_j+1)*2] += k12
            k_global[4*n_nodes*i+(node_j+1)*2-2:4*n_nodes*i+(node_j+1)*2, \
                4*n_nodes*j+(node_i+1)*2-2:4*n_nodes*j+(node_i+1)*2] += k21
            k_global[4*n_nodes*i+(node_j+1)*2-2:4*n_nodes*i+(node_j+1)*2, \
                4*n_nodes*j+(node_j+1)*2-2:4*n_nodes*j+(node_j+1)*2] += k22

            k_global[4*n_nodes*i+skip+(node_i+1)*2-2:4*n_nodes*i+skip+(node_i+1)*2, \
                4*n_nodes*j+skip+(node_i+1)*2-2:4*n_nodes*j+skip+(node_i+1)*2] += k33
            k_global[4*n_nodes*i+skip+(node_i+1)*2-2:4*n_nodes*i+skip+(node_i+1)*2, \
                4*n_nodes*j+skip+(node_j+1)*2-2:4*n_nodes*j+skip+(node_j+1)*2] += k34
            k_global[4*n_nodes*i+skip+(node_j+1)*2-2:4*n_nodes*i+skip+(node_j+1)*2, \
                4*n_nodes*j+skip+(node_i+1)*2-2:4*n_nodes*j+skip+(node_i+1)*2] += k43
            k_global[4*n_nodes*i+skip+(node_j+1)*2-2:4*n_nodes*i+skip+(node_j+1)*2, \
                4*n_nodes*j+skip+(node_j+1)*2-2:4*n_nodes*j+skip+(node_j+1)*2] += k44

            k_global[4*n_nodes*i+(node_i+1)*2-2:4*n_nodes*i+(node_i+1)*2, \
                4*n_nodes*j+skip+(node_i+1)*2-2:4*n_nodes*j+skip+(node_i+1)*2] += k13
            k_global[4*n_nodes*i+(node_i+1)*2-2:4*n_nodes*i+(node_i+1)*2, \
                4*n_nodes*j+skip+(node_j+1)*2-2:4*n_nodes*j+skip+(node_j+1)*2] += k14
            k_global[4*n_nodes*i+(node_j+1)*2-2:4*n_nodes*i+(node_j+1)*2, \
                4*n_nodes*j+skip+(node_i+1)*2-2:4*n_nodes*j+skip+(node_i+1)*2] += k23
            k_global[4*n_nodes*i+(node_j+1)*2-2:4*n_nodes*i+(node_j+1)*2, \
                4*n_nodes*j+skip+(node_j+1)*2-2:4*n_nodes*j+skip+(node_j+1)*2] += k24

            k_global[4*n_nodes*i+skip+(node_i+1)*2-2:4*n_nodes*i+skip+(node_i+1)*2, \
                4*n_nodes*j+(node_i+1)*2-2:4*n_nodes*j+(node_i+1)*2] += k31
            k_global[4*n_nodes*i+skip+(node_i+1)*2-2:4*n_nodes*i+skip+(node_i+1)*2, \
                4*n_nodes*j+(node_j+1)*2-2:4*n_nodes*j+(node_j+1)*2] += k32
            k_global[4*n_nodes*i+skip+(node_j+1)*2-2:4*n_nodes*i+skip+(node_j+1)*2, \
                4*n_nodes*j+(node_i+1)*2-2:4*n_nodes*j+(node_i+1)*2] += k41
            k_global[4*n_nodes*i+skip+(node_j+1)*2-2:4*n_nodes*i+skip+(node_j+1)*2, \
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

            kg_global[4*n_nodes*i + (node_i+1) * 2 - 2:4*n_nodes*i + (node_i+1) * 2,
                      4*n_nodes*j + (node_i+1) * 2 - 2:4*n_nodes*j + (node_i+1) * 2] += kg11
            kg_global[4*n_nodes*i + (node_i+1) * 2 - 2:4*n_nodes*i + (node_i+1) * 2,
                      4*n_nodes*j + (node_j+1) * 2 - 2:4*n_nodes*j + (node_j+1) * 2] += kg12
            kg_global[4*n_nodes*i + (node_j+1) * 2 - 2:4*n_nodes*i + (node_j+1) * 2,
                      4*n_nodes*j + (node_i+1) * 2 - 2:4*n_nodes*j + (node_i+1) * 2] += kg21
            kg_global[4*n_nodes*i + (node_j+1) * 2 - 2:4*n_nodes*i + (node_j+1) * 2,
                      4*n_nodes*j + (node_j+1) * 2 - 2:4*n_nodes*j + (node_j+1) * 2] += kg22

            kg_global[4*n_nodes*i + skip + (node_i+1) * 2 - 2:4*n_nodes*i + skip + (node_i+1) * 2,
                      4*n_nodes*j + skip + (node_i+1) * 2 - 2:4*n_nodes*j + skip
                      + (node_i+1) * 2] += kg33
            kg_global[4*n_nodes*i + skip + (node_i+1) * 2 - 2:4*n_nodes*i + skip + (node_i+1) * 2,
                      4*n_nodes*j + skip + (node_j+1) * 2 - 2:4*n_nodes*j + skip
                      + (node_j+1) * 2] += kg34
            kg_global[4*n_nodes*i + skip + (node_j+1) * 2 - 2:4*n_nodes*i + skip + (node_j+1) * 2,
                      4*n_nodes*j + skip + (node_i+1) * 2 - 2:4*n_nodes*j + skip
                      + (node_i+1) * 2] += kg43
            kg_global[4*n_nodes*i + skip + (node_j+1) * 2 - 2:4*n_nodes*i + skip + (node_j+1) * 2,
                      4*n_nodes*j + skip + (node_j+1) * 2 - 2:4*n_nodes*j + skip
                      + (node_j+1) * 2] += kg44

            kg_global[4*n_nodes*i + (node_i+1) * 2 - 2:4*n_nodes*i + (node_i+1) * 2, 4*n_nodes*j
                      + skip + (node_i+1) * 2 - 2:4*n_nodes*j + skip + (node_i+1) * 2] += kg13
            kg_global[4*n_nodes*i + (node_i+1) * 2 - 2:4*n_nodes*i + (node_i+1) * 2, 4*n_nodes*j
                      + skip + (node_j+1) * 2 - 2:4*n_nodes*j + skip + (node_j+1) * 2] += kg14
            kg_global[4*n_nodes*i + (node_j+1) * 2 - 2:4*n_nodes*i + (node_j+1) * 2, 4*n_nodes*j
                      + skip + (node_i+1) * 2 - 2:4*n_nodes*j + skip + (node_i+1) * 2] += kg23
            kg_global[4*n_nodes*i + (node_j+1) * 2 - 2:4*n_nodes*i + (node_j+1) * 2, 4*n_nodes*j
                      + skip + (node_j+1) * 2 - 2:4*n_nodes*j + skip + (node_j+1) * 2] += kg24

            kg_global[4*n_nodes*i + skip + (node_i+1) * 2 - 2:4*n_nodes*i + skip + (node_i+1) * 2,
                      4*n_nodes*j + (node_i+1) * 2 - 2:4*n_nodes*j + (node_i+1) * 2] += kg31
            kg_global[4*n_nodes*i + skip + (node_i+1) * 2 - 2:4*n_nodes*i + skip + (node_i+1) * 2,
                      4*n_nodes*j + (node_j+1) * 2 - 2:4*n_nodes*j + (node_j+1) * 2] += kg32
            kg_global[4*n_nodes*i + skip + (node_j+1) * 2 - 2:4*n_nodes*i + skip + (node_j+1) * 2,
                      4*n_nodes*j + (node_i+1) * 2 - 2:4*n_nodes*j + (node_i+1) * 2] += kg41
            kg_global[4*n_nodes*i + skip + (node_j+1) * 2 - 2:4*n_nodes*i + skip + (node_j+1) * 2,
                      4*n_nodes*j + (node_j+1) * 2 - 2:4*n_nodes*j + (node_j+1) * 2] += kg42

    return k_global, kg_global


def assemble_single(
    k_global: np.ndarray, k_local: np.ndarray, node_i: int, node_j: int, n_nodes: int
) -> np.ndarray:
    """this routine adds the element contribution to the global stiffness matrix
    basically it does the same as routine 'assemble', however:
    it does not care about kg_global (geom stiff matrix)
    only involves single half-wave number m_i

    Args:
        k_global (np.ndarray): global elastic stiffness matrix
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

    # the additional terms for k_global are stored in k_2_matrix
    skip = 2 * n_nodes
    k_global[node_i * 2:node_i*2 + 2, node_i * 2:node_i*2 + 2] += k11
    k_global[node_i * 2:node_i*2 + 2, node_j * 2:node_j*2 + 2] += k12
    k_global[node_j * 2:node_j*2 + 2, node_i * 2:node_i*2 + 2] += k21
    k_global[node_j * 2:node_j*2 + 2, node_j * 2:node_j*2 + 2] += k22

    k_global[skip + node_i*2:skip + node_i*2 + 2, skip + node_i*2:skip + node_i*2 + 2] += k33
    k_global[skip + node_i*2:skip + node_i*2 + 2, skip + node_j*2:skip + node_j*2 + 2] += k34
    k_global[skip + node_j*2:skip + node_j*2 + 2, skip + node_i*2:skip + node_i*2 + 2] += k43
    k_global[skip + node_j*2:skip + node_j*2 + 2, skip + node_j*2:skip + node_j*2 + 2] += k44

    k_global[node_i * 2:node_i*2 + 2, skip + node_i*2:skip + node_i*2 + 2] += k13
    k_global[node_i * 2:node_i*2 + 2, skip + node_j*2:skip + node_j*2 + 2] += k14
    k_global[node_j * 2:node_j*2 + 2, skip + node_i*2:skip + node_i*2 + 2] += k23
    k_global[node_j * 2:node_j*2 + 2, skip + node_j*2:skip + node_j*2 + 2] += k24

    k_global[skip + node_i*2:skip + node_i*2 + 2, node_i * 2:node_i*2 + 2] += k31
    k_global[skip + node_i*2:skip + node_i*2 + 2, node_j * 2:node_j*2 + 2] += k32
    k_global[skip + node_j*2:skip + node_j*2 + 2, node_i * 2:node_i*2 + 2] += k41
    k_global[skip + node_j*2:skip + node_j*2 + 2, node_j * 2:node_j*2 + 2] += k42

    return k_global


def spring_klocal(
    k_u: float, k_v: float, k_w: float, k_q: float, length: float, b_c: str, m_a: np.ndarray,
    discrete: int, y_s: float
) -> np.ndarray:
    """Generate spring stiffness matrix (k_local) in local coordinates, modified from
    klocal

    Args:
        k_u (float): spring stiffness values
        k_v (float): spring stiffness values
        k_w (float): spring stiffness values
        k_q (float): spring stiffness values
        length (float): length of the strip in longitudinal direction
        b_c (str): ['S-S'] a string specifying boundary conditions to be analyzed:
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
    total_m = len(m_a)  # Total number of longitudinal terms m
    k_local = np.zeros((8 * total_m, 8 * total_m))
    for i in range(0, total_m):
        for j in range(0, total_m):
            km_mp = np.zeros((4, 4))
            kf_mp = np.zeros((4, 4))
            u_i = m_a[i] * np.pi
            u_j = m_a[j] * np.pi

            if discrete:
                [i_1, i_5] = bc_i1_5_atpoint(
                    b_c=b_c, m_i=m_a[i], m_j=m_a[j], length=length, y_s=y_s
                )
            else:  # foundation spring
                [i_1, _, _, _, i_5] = bc_i1_5(b_c=b_c, m_i=m_a[i], m_j=m_a[j], length=length)
            # assemble the matrix of km_mp (membrane stiffness)
            km_mp = np.array(
                [[k_u * i_1, 0, -k_u * i_1, 0],
                 [0, k_v * i_5 * length**2 / (u_i*u_j), 0, -k_v * i_5 * length**2 / (u_i*u_j)],
                 [-k_u * i_1, 0, k_u * i_1, 0],
                 [0, -k_v * i_5 * length**2 / (u_i*u_j), 0, k_v * i_5 * length**2 / (u_i*u_j)]]
            )
            # assemble the matrix of kf_mp (flexural stiffness)
            kf_mp = np.array([[k_w * i_1, 0, -k_w * i_1, 0], [0, k_q * i_1, 0, -k_q * i_1],
                              [-k_w * i_1, 0, k_w * i_1, 0], [0, -k_q * i_1, 0, k_q * i_1]])

            k_local[8 * i:8*i + 4, 8 * j:8*j + 4] = km_mp
            k_local[8*i + 4:8 * (i+1), 8*j + 4:8 * (j+1)] = kf_mp

    return k_local


def bc_i1_5_atpoint(b_c: str, m_i: float, m_j: float, length: float,
                    y_s: float) -> Tuple[float, float]:
    """Calculate the value of the longitudinal shape functions for discrete springs


    Args:
        b_c (str): a string specifying boundary conditions to be analyzed:
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
        i_1 (float): calculation of i_1 is the value of y_m(y/L)*Yn(y/L)
        i_5 (float): calculation of i_5 is the value of y_m'(y/L)*Yn'(y/L)_description_
    """
    y_i = ym_at_ys(b_c=b_c, m_i=m_i, y_s=y_s, length=length)
    y_j = ym_at_ys(b_c=b_c, m_i=m_j, y_s=y_s, length=length)
    y_i_prime = ymprime_at_ys(b_c=b_c, m_i=m_i, y_s=y_s, length=length)
    y_j_prime = ymprime_at_ys(b_c=b_c, m_i=m_j, y_s=y_s, length=length)
    i_1 = y_i * y_j
    i_5 = y_i_prime * y_j_prime
    return i_1, i_5


def spring_assemble(
    k_global: np.ndarray, k_local: np.ndarray, node_i: int, node_j: int, n_nodes: int,
    m_a: np.ndarray
) -> np.ndarray:
    """Add the (spring) contribution to the global stiffness matrix

    Args:
        k_global (np.ndarray): global elastic stiffness matrix 
            total_m x total_m submatrices. Each submatrix is similar to the
            one used in original CUFSM for single longitudinal term m in the DOF order
            [u1 v1...un vn w1 01...wn 0n]m'.
        k_local (np.ndarray): local elastic stiffness matrix
        node_i (int): node number
        node_j (int): node number
        n_nodes (int): total number of nodes 
        m_a (np.ndarray): number of half-wavelengths

    Returns:
        k_global (np.ndarray): global elastic stiffness matrix

    Z. Li, June 2008
    modified by Z. Li, Aug. 09, 2009
    Z. Li, June 2010
    adapted for springs BWS Dec 2015
    """
    total_m = len(m_a)  # Total number of longitudinal terms m
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

            k_global[4*n_nodes*i + (node_i+1) * 2 - 1:4*n_nodes*i + (node_i+1) * 2,
                     4*n_nodes*j + (node_i+1) * 2 - 1:4*n_nodes*j + (node_i+1) * 2] += k11
            if node_j != -1:
                k_global[4*n_nodes*i + (node_i+1) * 2 - 1:4*n_nodes*i + (node_i+1) * 2,
                         4*n_nodes*j + (node_j+1) * 2 - 1:4*n_nodes*j + (node_j+1) * 2] += k12
                k_global[4*n_nodes*i + (node_j+1) * 2 - 1:4*n_nodes*i + (node_j+1) * 2,
                         4*n_nodes*j + (node_i+1) * 2 - 1:4*n_nodes*j + (node_i+1) * 2] += k21
                k_global[4*n_nodes*i + (node_j+1) * 2 - 1:4*n_nodes*i + (node_j+1) * 2,
                         4*n_nodes*j + (node_j+1) * 2 - 1:4*n_nodes*j + (node_j+1) * 2] += k22

            k_global[4*n_nodes*i + skip + (node_i+1) * 2 - 1:4*n_nodes*i + skip + (node_i+1) * 2,
                     4*n_nodes*j + skip + (node_i+1) * 2 - 1:4*n_nodes*j + skip
                     + (node_i+1) * 2] += k33
            if node_j != -1:
                k_global[4*n_nodes*i + skip + (node_i+1) * 2 - 1:4*n_nodes*i + skip
                         + (node_i+1) * 2, 4*n_nodes*j + skip + (node_j+1) * 2 - 1:4*n_nodes*j
                         + skip + (node_j+1) * 2] += k34
                k_global[4*n_nodes*i + skip + (node_j+1) * 2 - 1:4*n_nodes*i + skip
                         + (node_j+1) * 2, 4*n_nodes*j + skip + (node_i+1) * 2 - 1:4*n_nodes*j
                         + skip + (node_i+1) * 2] += k43
                k_global[4*n_nodes*i + skip + (node_j+1) * 2 - 1:4*n_nodes*i + skip
                         + (node_j+1) * 2, 4*n_nodes*j + skip + (node_j+1) * 2 - 1:4*n_nodes*j
                         + skip + (node_j+1) * 2] += k44

            k_global[4*n_nodes*i + (node_i+1) * 2 - 1:4*n_nodes*i + (node_i+1) * 2, 4*n_nodes*j
                     + skip + (node_i+1) * 2 - 1:4*n_nodes*j + skip + (node_i+1) * 2] += k13
            if node_j != -1:
                k_global[4*n_nodes*i + (node_i+1) * 2 - 1:4*n_nodes*i + (node_i+1) * 2, 4*n_nodes*j
                         + skip + (node_j+1) * 2 - 1:4*n_nodes*j + skip + (node_j+1) * 2] += k14
                k_global[4*n_nodes*i + (node_j+1) * 2 - 1:4*n_nodes*i + (node_j+1) * 2, 4*n_nodes*j
                         + skip + (node_i+1) * 2 - 1:4*n_nodes*j + skip + (node_i+1) * 2] += k23
                k_global[4*n_nodes*i + (node_j+1) * 2 - 1:4*n_nodes*i + (node_j+1) * 2, 4*n_nodes*j
                         + skip + (node_j+1) * 2 - 1:4*n_nodes*j + skip + (node_j+1) * 2] += k24

            k_global[4*n_nodes*i + skip + (node_i+1) * 2 - 1:4*n_nodes*i + skip + (node_i+1) * 2,
                     4*n_nodes*j + (node_i+1) * 2 - 1:4*n_nodes*j + (node_i+1) * 2] += k31
            if node_j != -1:
                k_global[4*n_nodes*i + skip + (node_i+1) * 2 - 1:4*n_nodes*i + skip
                         + (node_i+1) * 2,
                         4*n_nodes*j + (node_j+1) * 2 - 1:4*n_nodes*j + (node_j+1) * 2] += k32
                k_global[4*n_nodes*i + skip + (node_j+1) * 2 - 1:4*n_nodes*i + skip
                         + (node_j+1) * 2,
                         4*n_nodes*j + (node_i+1) * 2 - 1:4*n_nodes*j + (node_i+1) * 2] += k41
                k_global[4*n_nodes*i + skip + (node_j+1) * 2 - 1:4*n_nodes*i + skip
                         + (node_j+1) * 2,
                         4*n_nodes*j + (node_j+1) * 2 - 1:4*n_nodes*j + (node_j+1) * 2] += k42

    return k_global


def ym_at_ys(b_c: str, m_i: float, y_s: float, length: float) -> float:
    """Longitudinal shape function values
    could be called in lots of places,  but now (2015) is hardcoded by Zhanjie
    in several places in the interface
    written in 2015 because wanted it for a new idea on discrete springs

    Args:
        b_c (str): boundary condition
        m_i (float): number of half-wavelengths
        y_s (float): location of discrete spring
        length (float): element length

    Returns:
        y_m (float): longitudinal shape function value
    
    BWS in 2015
    """
    y_m: float
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
    else:
        raise ValueError(f"Unrecognised boundary condition '{b_c}'")

    return y_m


def ymprime_at_ys(b_c: str, m_i: float, y_s: float, length: float) -> float:
    """First Derivative of Longitudinal shape function values
    could be called in lots of places,  but now (2015) is hardcoded by Zhanjie
    in several places in the interface
    written in 2015 because wanted it for a new idea on discrete springs

    Args:
        b_c (str): boundary condition
        m_i (float): number of half-wavelengths
        y_s (float): location of discrete spring
        length (float): element length

    Returns:
        y_m_prime (float): first derivative of longitudinal shape function

    BWS in 2015
    """
    y_m_prime: float
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
    else:
        raise ValueError(f"Unrecognised boundary condition '{b_c}'")

    return y_m_prime
