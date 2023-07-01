from typing import Sequence

import numpy as np

from pycufsm.types import ArrayLike, New_Element, Sect_Props


def prop2_new(nodes: ArrayLike, elements: Sequence[New_Element]) -> Sect_Props:
    """Run a CUTWP analysis using the new format for nodes and elements

    Args:
        nodes (ArrayLike): Nodal coordinates
            | `[[x, y], ...]`
            Note that any extra columns (such as a third column for stresses),
            will simply be ignored.
        elements (Sequence[New_Element]): Element definition
            | [{
            |    nodes: "all"|[node1, ...],
            |    t: thickness,
            |    mat: mat_name
            | }]
            How the nodes are connected. The material name will be ignored.

    Returns:
        sect_props(Sect_Props): Dictionary of section properties
    """
    # The new nodes format is structurally identical to the original `coord` format
    coord = np.array(nodes)

    ends: list = []
    for elem in elements:
        if isinstance(elem["nodes"], str) and elem["nodes"] == "all":
            elem["nodes"] = list(range(len(nodes)))
        for node1, node2 in zip(elem["nodes"][0:], elem["nodes"][1:]):
            ends.append([node1, node2, elem["t"]])

    return prop2(coord=coord, ends=np.array(ends))


def prop2(coord: np.ndarray, ends: np.ndarray) -> Sect_Props:
    """Function modified for use in CUFSM by Ben Schafer in 2004 with permission
    of Sarawit. removed elastic buckling calcs and kept only section
    properties.
    
    Compute cross section properties
        This function computes the cross section properties: area, centroid,
        moment of inertia, torsional constant, shear center, warping constant,
        b1_vals, b2_vals, elastic critical buckling load and the deformed buckling shape

    Args:
        coord (np.ndarray): node i's coordinates
            coord(i,1) = X coordinate
            coord(i,2) = Y coordinate
        ends (np.ndarray): subelement i's nodal information
            ends(i,1) = start node #
            ends(i,2) = finish node #
            ends(i,3) = element's thicknesses

    Returns:
        sect_props (Sect_Props): Dictionary of section properties:
            A              ==  cross section area
            xc             ==  X coordinate of the centroid from origin
            yc             ==  Y coordinate of the centroid from origin
            Ix             ==  moment of inertia about centroid X axes
            Iy             ==  moment of inertia about centroid Y axes
            Ixy            ==  product of inertia about centroid
            Iz             ==  polar moment of inertia about centroid
            theta          ==  rotation angle for the principal axes
            I1             ==  principal moment of inertia about centroid 1 axes
            I2             ==  principal moment of inertia about centroid 2 axes
            J              ==  torsional constant
            xo             ==  X coordinate of the shear center from origin
            yo             ==  Y coordinate of the shear center from origin
            Cw             ==  warping constant
            B1             ==  int(y*(x^2+y^2),s,0,lengths)   *BWS, x,y=prin. crd.
            B2             ==  int(x*(x^2+y^2),s,0,lengths)
                                where: x = x_1+s/lengths*(x_2-x_1)
                                    y = y_1+s/lengths*(y_2-y_1)
                                    lengths = length of the element

        Note:
        J,xo,yo,Cw,B1,B2,Pe,dcoord is not computed for close-section

    Andrew T. Sarawit, Wed 10/25/01
    BWS, 2004
    BWS, Aug 2005: additional modifications, program only handles
        singly-branched open sections, or single cell closed sections, arbitrary
        section designation added for other types.
    BWS, Dec 2006 bug fixes to b1_vals b2_vals
    BWS, Dec 2015 extended to not crash on disconnected and arbitrary sections
    """
    # find n_elements  == total number of elements
    #      n_nodes == total number of nodes
    #      j     == total number of 2 element joints

    ###Calculates Section Properties
    n_elements = len(ends)
    node = ends[:, 0:2]
    # node = node(:)
    # n_nodes = 0
    # j = 0
    # while len(node)>0:
    #     i =
    nodes = np.append(node[:, 0], node[:, 1])
    nodes = np.lib.unique(nodes)
    # j = len(nodes)-1
    # if j == n_elements:
    #     section = 'close'
    # elif j == n_elements - 1:
    #     section = 'open'
    # else:
    #     section = 'arbitrary'
    section = "open"  # only open sections are supported in CUFSM

    # #if the section is closed re-order the elements
    # if (section == 'close'):
    #     xnele = n_elements - 1
    #     for i in range(xnele):
    #         en = ends
    #         en[i, 1] = 0

    # Find the element properties
    thicknesses = np.zeros(n_elements)
    x_means = np.zeros(n_elements)
    y_means = np.zeros(n_elements)
    x_diffs = np.zeros(n_elements)
    y_diffs = np.zeros(n_elements)
    lengths = np.zeros(n_elements)
    for i, elem in enumerate(ends):
        start_node = int(elem[0])
        end_node = int(elem[1])
        thicknesses[i] = elem[2]
        # Compute coordinate of midpoint of the element
        x_means[i] = np.mean([coord[start_node, 0], coord[end_node, 0]])
        y_means[i] = np.mean([coord[start_node, 1], coord[end_node, 1]])
        # Compute the dimension of the element
        x_diffs[i] = np.diff([coord[start_node, 0], coord[end_node, 0]])
        y_diffs[i] = np.diff([coord[start_node, 1], coord[end_node, 1]])
        # Compute length
        lengths[i] = np.sqrt(x_diffs[i]**2 + y_diffs[i]**2)

    # Compute Area
    area = np.sum(lengths * thicknesses)
    # Compute centroid
    x_centroid = np.sum(lengths * thicknesses * x_means) / area
    y_centroid = np.sum(lengths * thicknesses * y_means) / area
    if np.abs(x_centroid / np.sqrt(area)) < 1e-12:
        x_centroid = 0
    if np.abs(y_centroid / np.sqrt(area)) < 1e-12:
        y_centroid = 0

    # Compute moment of inertia
    i_xx = np.sum((y_diffs**2 / 12 + (y_means - y_centroid)**2) * lengths * thicknesses)
    i_yy = np.sum((x_diffs**2 / 12 + (x_means - x_centroid)**2) * lengths * thicknesses)
    i_xy = np.sum(
        (x_diffs*y_diffs/12 + (x_means-x_centroid) * (y_means-y_centroid) * lengths * thicknesses)
    )
    if np.abs(i_xy / area**2) < 1e-12:
        i_xy = 0

    # Compute rotation angle for the principal axes
    theta = (np.angle([(i_xx-i_yy) - 2*i_xy*1j]) / 2)[0]

    # Transfer section coordinates to the centroid principal coordinates
    coord12 = np.array([coord[:, 0] - x_centroid, coord[:, 1] - y_centroid]).T
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    coord12 = rotation_matrix @ coord12.T
    coord12 = coord12.T

    # Find the element properties
    x_means12 = np.zeros(n_elements)
    y_means12 = np.zeros(n_elements)
    x_diffs12 = np.zeros(n_elements)
    y_diffs12 = np.zeros(n_elements)
    for i, elem in enumerate(ends):
        start_node = int(elem[0])
        end_node = int(elem[1])
        # Compute coordinate of midpoint of the element
        x_means12[i] = np.mean([coord12[start_node, 0], coord12[end_node, 0]])
        y_means12[i] = np.mean([coord12[start_node, 1], coord12[end_node, 1]])
        # Compute the dimension of the element
        x_diffs12[i] = np.diff([coord12[start_node, 0], coord12[end_node, 0]])
        y_diffs12[i] = np.diff([coord12[start_node, 1], coord12[end_node, 1]])

    # Compute the principal moment of inertia
    i_11 = np.sum((y_diffs12**2 / 12 + (y_means12)**2) * lengths * thicknesses)
    i_22 = np.sum((x_diffs12**2 / 12 + (x_means12)**2) * lengths * thicknesses)

    if section == "open":
        # Compute torsional constant
        j_torsion = np.sum(lengths * thicknesses**3) / 3
        # Compute shear center and initialize variables
        n_nodes = len(coord)
        w_vals = np.zeros((n_nodes, 2))
        w_vals[int(ends[0, 0]), 0] = int(ends[0, 0]) + 1
        wo_vals = np.zeros((n_nodes, 2))
        wo_vals[int(ends[0, 0]), 0] = int(ends[0, 0]) + 1
        i_wx = 0
        i_wy = 0
        w_no = 0
        c_warping = 0
        ends[:, 0:2] = (ends[:, 0:2]) + 1
        for _ in range(n_elements):
            i = 0
            while i < len(ends) - 1 and (
                (np.any(w_vals[:, 0] == ends[i, 0]) and np.any(w_vals[:, 0] == ends[i, 1])) or
                (not (np.any(w_vals[:, 0] == ends[i, 0])) and
                 (not np.any(w_vals[:, 0] == ends[i, 1])))):
                i = i + 1

            start_node = int(ends[i, 0]) - 1
            end_node = int(ends[i, 1]) - 1
            p_vals = ((coord[start_node, 0] - x_centroid) * (coord[end_node, 1] - y_centroid) -
                      (coord[end_node, 0] - x_centroid) *
                      (coord[start_node, 1] - y_centroid)) / lengths[i]
            if w_vals[start_node, 0] == 0:
                w_vals[start_node, 0] = start_node + 1
                w_vals[start_node, 1] = w_vals[end_node, 1] - p_vals * lengths[i]
            elif w_vals[end_node, 0] == 0:
                w_vals[end_node, 0] = end_node + 1
                w_vals[end_node, 1] = w_vals[start_node, 1] + p_vals * lengths[i]
            i_wx = (
                i_wx + (
                    1 / 3 * (
                        w_vals[start_node, 1] *
                        (coord[start_node, 0] - x_centroid) + w_vals[end_node, 1] *
                        (coord[end_node, 0] - x_centroid)
                    ) + 1 / 6 * (
                        w_vals[start_node, 1] *
                        (coord[end_node, 0] - x_centroid) + w_vals[end_node, 1] *
                        (coord[start_node, 0] - x_centroid)
                    )
                ) * thicknesses[i] * lengths[i]
            )
            i_wy = (
                i_wy + (
                    1 / 3 * (
                        w_vals[start_node, 1] *
                        (coord[start_node, 1] - y_centroid) + w_vals[end_node, 1] *
                        (coord[end_node, 1] - y_centroid)
                    ) + 1 / 6 * (
                        w_vals[start_node, 1] *
                        (coord[end_node, 1] - y_centroid) + w_vals[end_node, 1] *
                        (coord[start_node, 1] - y_centroid)
                    )
                ) * thicknesses[i] * lengths[i]
            )
        if (i_xx*i_yy - i_xy**2) != 0:
            x_shearcentre = (i_yy*i_wy - i_xy*i_wx) / (i_xx*i_yy - i_xy**2) + x_centroid
            y_shearcentre = -(i_xx*i_wx - i_xy*i_wy) / (i_xx*i_yy - i_xy**2) + y_centroid
        else:
            x_shearcentre = x_centroid
            y_shearcentre = y_centroid
        if np.abs(x_shearcentre / np.sqrt(area)) < 1e-12:
            x_shearcentre = 0
        if np.abs(y_shearcentre / np.sqrt(area)) < 1e-12:
            y_shearcentre = 0
        # Compute unit warping
        for _ in range(n_elements):
            i = 0
            while i < len(ends) - 1 and (
                (np.any(w_vals[:, 0] == ends[i, 0]) and np.any(w_vals[:, 0] == ends[i, 1])) or
                (not (np.any(w_vals[:, 0] == ends[i, 0])) and
                 (not np.any(w_vals[:, 0] == ends[i, 1])))):
                i = i + 1
            start_node = int(ends[i, 0]) - 1
            end_node = int(ends[i, 0]) - 1
            po_vals = ((coord[start_node, 0] - x_shearcentre) *
                       (coord[end_node, 1] - y_shearcentre) - (coord[end_node, 0] - x_shearcentre) *
                       (coord[start_node, 1] - y_shearcentre)) / lengths[i]
            if w_vals[start_node, 0] == 0:
                w_vals[start_node, 0] = start_node + 1
                w_vals[start_node, 1] = w_vals[end_node, 1] - po_vals * lengths[i]
            elif w_vals[end_node, 0] == 0:
                w_vals[end_node, 0] = end_node + 1
                w_vals[end_node, 1] = w_vals[start_node, 1] + po_vals * lengths[i]
            w_no = (
                w_no + 1 / (2*area) *
                (wo_vals[start_node, 1] + wo_vals[end_node, 1]) * thicknesses[i] * lengths[i]
            )
        wn_vals = np.zeros((len(wo_vals), 2))
        wn_vals = w_no - wo_vals[:, 1]
        # Compute the warping constant
        for i in range(n_elements):
            start_node = int(ends[i, 0]) - 1
            end_node = int(ends[i, 1]) - 1
            c_warping = (
                c_warping + 1 / 3 * (
                    wn_vals[start_node]**2 + wn_vals[start_node] * wn_vals[end_node]
                    + wn_vals[end_node]**2
                ) * thicknesses[i] * lengths[i]
            )
        # transfer the shear center coordinates to the centroid principal coordinates
        s12 = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]
               ] @ (np.array([x_shearcentre - x_centroid, y_shearcentre - y_centroid]).T)
        # compute the polar radius of gyration of cross section about shear center
        # ro = np.sqrt((i_11 + i_22) / area + s12[0] ** 2 + s12[1] ** 2)

        # Compute b1_vals and b2_vals
        b1_vals = 0
        b2_vals = b1_vals
        for i in range(n_elements):
            start_node = int(ends[i, 0]) - 1
            end_node = int(ends[i, 1]) - 1
            x_1 = coord12[start_node, 0]
            y_1 = coord12[start_node, 1]
            x_2 = coord12[end_node, 0]
            y_2 = coord12[end_node, 1]
            b1_vals = (
                b1_vals +
                ((y_1+y_2) * (y_1**2 + y_2**2) / 4 +
                 (y_1 * (2 * x_1**2 + (x_1 + x_2)**2) + y_2 *
                  (2 * x_2**2 + (x_1 + x_2)**2)) / 12) * lengths[i] * thicknesses[i]
            )
            b2_vals = (
                b2_vals +
                ((x_1+x_2) * (x_1**2 + x_2**2) / 4 +
                 (x_1 * (2 * y_1**2 + (y_1 + y_2)**2) + x_2 *
                  (2 * y_2**2 + (y_1 + y_2)**2)) / 12) * lengths[i] * thicknesses[i]
            )
        b1_vals = b1_vals/i_11 - 2 * s12[1]
        b2_vals = b2_vals/i_22 - 2 * s12[0]

        if np.abs(b1_vals / np.sqrt(area) < 1e-12):
            b1_vals = 0
        if np.abs(b2_vals / np.sqrt(area) < 1e-12):
            b2_vals = 0
    ends[:, 0:2] = (ends[:, 0:2]) - 1
    return {
        "A": area,
        "cx": x_centroid,
        "cy": y_centroid,
        "Ixx": i_xx,
        "Iyy": i_yy,
        "Ixy": i_xy,
        "phi": theta,
        "I11": i_11,
        "I22": i_22,
        "J": j_torsion,
        "x0": x_shearcentre,
        "y0": y_shearcentre,
        "Cw": c_warping,
        "B1": b1_vals,
        "B2": b2_vals,
        "wn": wn_vals,
    }
