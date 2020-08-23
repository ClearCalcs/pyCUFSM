import numpy as np
import pycufsm.fsm
import pycufsm.cfsm

# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE, CPEng
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


#Helper Function
def gammait(phi, dbar):
    p = phi
    gamma = np.array([[np.cos(p), 0, 0, 0, -np.sin(p), 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, np.cos(p), 0, 0, 0, -np.sin(p), 0], [0, 0, 0, 1, 0, 0, 0, 0],
                      [np.sin(p), 0, 0, 0, np.cos(p), 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, np.sin(p), 0, 0, 0, np.cos(p), 0], [0, 0, 0, 0, 0, 0, 0, 1]])
    d = np.dot(gamma, dbar)
    return d


#Helper Fucntion
def gammait2(phi, dl):
    p = phi
    gamma = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
    dlbar = np.dot(np.linalg.inv(gamma), dl)
    return dlbar


#Helper function
def shapef(links, d, b):
    inc = 1/(links)
    xb = np.linspace(inc, 1 - inc, links - 1)
    dl = np.zeros((3, len(xb)))
    for i in range(len(xb)):
        N1 = 1 - 3*xb[i]*xb[i] + 2*xb[i]*xb[i]*xb[i]
        N2 = xb[i]*b*(1 - 2*xb[i] + xb[i]**2)
        N3 = 3*xb[i]**2 - 2*xb[i]**3
        N4 = xb[i]*b*(xb[i]**2 - xb[i])
        N = np.array([[(1 - xb[i]), 0, xb[i], 0, 0, 0, 0, 0],
                      [0, (1 - xb[i]), 0, xb[i], 0, 0, 0, 0], [0, 0, 0, 0, N1, N2, N3, N4]])
        dl[:, i] = np.dot(N, d).reshape(3)
    return dl


def lengths_recommend(nodes, elements, length_append=None, n_lengths=50):
    #
    #Z. Li, July 2010 (last modified)
    #generate the signature curve solution
    # Function split by Brooks Smith; this part only finds recommended lengths
    #
    min_el_length = 1000000  #Minimum element length
    max_el_length = 0  #Maximum element length
    for elem in elements:
        hh1 = abs(
            np.sqrt((nodes[int(elem[1]), 1] - nodes[int(elem[2]), 1])**2
                    + (nodes[int(elem[1]), 2] - nodes[int(elem[2]), 2])**2)
        )
        min_el_length = min(hh1, min_el_length)
        max_el_length = max(hh1, max_el_length)

    lengths = np.logspace(np.log10(min_el_length), np.log10(1000*max_el_length), num=n_lengths)

    if length_append is not None:
        lengths = np.sort(np.concatenate((lengths, [length_append])))

    return lengths


def signature_ss(props, nodes, elements, i_gbt_con, sect_props, lengths):
    #
    #Z. Li, July 2010 (last modified)
    #generate the signature curve solution
    #
    i_springs = []
    i_constraints = []
    i_b_c = 'S-S'
    i_m_all = np.ones((len(lengths), 1))

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


def m_recommend(props, nodes, elements, sect_props, length_append=None):
    # Z. Li, Oct. 2010
    #Suggested longitudinal terms are calculated based on the characteristic
    #half-wave lengths of local, distortional, and global buckling from the
    #signature curve.

    #the inputs for signature curve
    i_gbt_con = {
        "glob": [0],
        "dist": [0],
        "local": [0],
        "other": [0],
        "o_space": 1,
        "couple": 1,
        "orth": 1,
        "norm": 1,
    }
    lengths = lengths_recommend(nodes=nodes, elements=elements, length_append=length_append)

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
    curve_signature[:, 0] = lengths
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
    n_other_modes = 2*(len(nodes) - 1)

    i_gbt_con["local"] = np.ones((n_local_modes, 1))
    i_gbt_con["dist"] = np.zeros((n_dist_modes, 1))
    i_gbt_con["glob"] = np.zeros((n_global_modes, 1))
    i_gbt_con["other"] = np.zeros((n_other_modes, 1))

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
    i_gbt_con["local"] = np.zeros((n_local_modes, 1))
    i_gbt_con["dist"] = np.ones((n_dist_modes, 1))
    i_gbt_con["glob"] = np.zeros((n_global_modes, 1))
    i_gbt_con["other"] = np.zeros((n_other_modes, 1))
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

    #cFSM dist half-wavelength
    local_minima_dist = []
    for i, c_sign in enumerate(curve_signature_dist[:-2]):
        load1 = c_sign[1]
        load2 = curve_signature_dist[i + 1, 1]
        load3 = curve_signature_dist[i + 2, 1]
        if load2 < load1 and load2 <= load3:
            local_minima_dist.append(curve_signature_dist[i + 1, 0])

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

        if np.ceil(im_p_len/length_crl) > 4:
            im_pm_all_temp = [
                np.ceil(im_p_len/length_crl) - 3,
                np.ceil(im_p_len/length_crl) - 2,
                np.ceil(im_p_len/length_crl) - 1,
                np.ceil(im_p_len/length_crl),
                np.ceil(im_p_len/length_crl) + 1,
                np.ceil(im_p_len/length_crl) + 2,
                np.ceil(im_p_len/length_crl) + 3,
            ]
        else:
            im_pm_all_temp = [1, 2, 3, 4, 5, 6, 7]

        if np.ceil(im_p_len/length_crd) > 4:
            im_pm_all_temp.extend([
                np.ceil(im_p_len/length_crd) - 3,
                np.ceil(im_p_len/length_crd) - 2,
                np.ceil(im_p_len/length_crd) - 1,
                np.ceil(im_p_len/length_crd),
                np.ceil(im_p_len/length_crd) + 1,
                np.ceil(im_p_len/length_crd) + 2,
                np.ceil(im_p_len/length_crl) + 3,
            ])
        else:
            im_pm_all_temp.extend([1, 2, 3, 4, 5, 6, 7])

        im_pm_all_temp.extend([1, 2, 3])

        im_pm_all.append(im_pm_all_temp)

    #m_a_recommend = analysis.m_sort(im_pm_all)
    m_a_recommend = im_pm_all

    return m_a_recommend, lengths, isignature, icurve, ishapes, length_crl, length_crd, \
        isignature_local, icurve_local, ishapes_local, \
            isignature_dist, icurve_dist, ishapes_dist


def load_mat(mat):
    nodes = np.array(mat['node'], dtype=np.dtype(np.double))
    for i in range(len(nodes)):
        nodes[i, 0] = int(np.double(nodes[i, 0])) - 1
        nodes[i, 3] = int(np.double(nodes[i, 3]))
        nodes[i, 4] = int(np.double(nodes[i, 4]))
        nodes[i, 5] = int(np.double(nodes[i, 5]))
        nodes[i, 6] = int(np.double(nodes[i, 6]))
        #nodes[i, 7] = 0
    elements = np.array(mat['elem'])
    for i in range(len(elements)):
        elements[i, 0] = int(np.double(elements[i, 0])) - 1
        elements[i, 1] = int(np.double(elements[i, 1])) - 1
        elements[i, 2] = int(np.double(elements[i, 2])) - 1
    lengths = np.array(mat['lengths']).T
    props = np.array(mat['prop'])
    constraints = np.array(mat['constraints'])
    if (len(constraints[0]) > 5):
        for i in range(len(constraints)):
            for j in range(len(constraints[i])):
                if j < 5 and j != 2:
                    constraints[i, j] = int(constraints[i, j]) - 1
    if (len(constraints[0]) < 5):
        constraints = []
    springs = np.array(mat['springs'])
    return nodes, elements, lengths, props, constraints, springs
