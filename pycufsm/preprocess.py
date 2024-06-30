from typing import Optional, Tuple, Union

import numpy as np
from scipy import linalg as spla  # type: ignore

from pycufsm.types import Forces, Sect_Geom, Sect_Props

# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


def template_path(
    draw_table: list, thick: float, n_r: int = 4, shift: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Assuming a uniform thickness, draws a section according to a path definition

    Args:
        draw_table (list): matrix of the form [[theta, dist, rad, n_s]], where:
            theta = starting angle, dist = length of straight segment,
            rad = radius of curved segment, n_s = number of mesh elements in straight
        thick (float): thickness
        n_r (int, optional): number of mesh elements in curved segments. Defaults to 4.
        shift (Optional[list], optional): amount to shift cross-section. Defaults to None.

    Returns:
        nodes (np.ndarray): standard nodes matrix
        elements (np.ndarray): standard elements matrix

    B Smith, Jun 2020
    """
    if shift is None:
        shift = [0, 0]

    nodes: list = []
    elements: list = []
    n_r_table = len(draw_table[0]) == 5

    # Set initial point
    if draw_table[0][1] != 0:
        nodes.append(np.array([len(nodes), shift[0], shift[1], 1, 1, 1, 1, 1.0]))

    # Progress through drawing the section
    for i, row in enumerate(draw_table[:-1]):
        theta = row[0]
        dist = row[1]
        rad = row[2]
        n_s = row[3]
        next_theta = draw_table[i + 1][0]
        phi = np.mod(next_theta - theta, 2 * np.pi)
        if phi > np.pi:
            phi = phi - 2 * np.pi

        # Add elements in straight segment (if n_s > 1)
        for j in range(1, int(n_s)):
            x_loc = nodes[-1][1] + dist / n_s * np.cos(theta)
            y_loc = nodes[-1][2] + dist / n_s * np.sin(theta)
            nodes.append(np.array([len(nodes), x_loc, y_loc, 1, 1, 1, 1, 1.0]))

        # Add elements in curved segment
        centre = [
            nodes[-1][1] + dist / n_s * np.cos(theta) - np.sign(phi) * rad * np.sin(theta),
            nodes[-1][2] + dist / n_s * np.sin(theta) + np.sign(phi) * rad * np.cos(theta),
        ]
        if n_r_table:
            n_r = row[4]

        if rad == 0:
            nodes.append(np.array([len(nodes), centre[0], centre[1], 1, 1, 1, 1, 1.0]))
        else:
            for j in range(int(n_r)):
                theta_j = theta - np.sign(phi) * np.pi / 2 + j * 1.0 / max(1, n_r - 1) * phi
                x_loc = centre[0] + rad * np.cos(theta_j)
                y_loc = centre[1] + rad * np.sin(theta_j)
                nodes.append(np.array([len(nodes), x_loc, y_loc, 1, 1, 1, 1, 1.0]))

    # Draw the last straight line
    theta = draw_table[-1][0]
    dist = draw_table[-1][1]
    n_s = draw_table[-1][3]
    if dist > 0:
        for j in range(1, int(n_s) + 1):
            x_loc = nodes[-1][1] + dist / n_s * np.cos(theta)
            y_loc = nodes[-1][2] + dist / n_s * np.sin(theta)
            nodes.append(np.array([len(nodes), x_loc, y_loc, 1, 1, 1, 1, 1.0]))

    # build the elements list
    for i in range(0, len(nodes) - 1):
        elements.append(np.array([i, i, i + 1, thick, 0]))

    return np.array(nodes), np.array(elements)


def template_calc(sect: Sect_Geom) -> Tuple[np.ndarray, np.ndarray]:
    """Converts overall geometry parameters for C or Z sections into valid
    nodes and elements matrices. Facilitates easier geometry creation

    Args:
        sect (Sect_Geom): overall section geometry of C or Z section

    Returns:
        nodes (np.ndarray): standard nodes matrix
        elements (np.ndarray): standard elements matrix

    BWS Aug 2000
    BWS, 2015 modification to allow for l_1=l_2=0 and creation of a track with same template
    BWS, 2015 addition to allow outer dimensions and inner radii to be used
    BWS, 2015 addition to control element discretization
    """
    n_d = sect["n_d"]
    n_b1 = sect["n_b1"]
    n_b2 = sect["n_b2"]
    n_l1 = sect["n_l1"]
    n_l2 = sect["n_l2"]
    n_r = sect["n_r"]

    nodes: list = []
    elements: list = []

    # CorZ=determines sign conventions for flange 1=C 2=Z
    if sect["type"] == "Z":
        flip_b2 = -1
    else:
        flip_b2 = 1

    # channel template
    # convert angles to radians
    q_1 = 90 * np.pi / 180
    q_2 = 90 * np.pi / 180

    # outer dimensions and inner radii came in and these
    # need to be corrected to all centerline for the use of this template
    [depth, b_1, l_1, b_2, l_2, rad, thick] = template_out_to_in(sect)

    # rest of the dimensions are "flat dimensions" and acceptable for modeling
    if rad == 0:
        if l_1 == 0 and l_2 == 0:
            # track or unlipped Z with sharp corners
            geom = [
                {"x": b_1, "y": 0, "n": n_b1, "r": False},
                {"x": 0, "y": 0, "n": n_d, "r": False},
                {"x": 0, "y": depth, "n": n_b2, "r": False},
                {"x": flip_b2 * (b_2), "y": depth, "n": 0, "r": False},
            ]
        else:
            # lipped C or Z with sharp corners
            geom = [
                {"x": b_1 + l_1 * np.cos(q_1), "y": l_1 * np.sin(q_1), "n": n_l1, "r": False},
                {"x": b_1, "y": 0, "n": n_b1, "r": False},
                {"x": 0, "y": 0, "n": n_d, "r": False},
                {"x": 0, "y": depth, "n": n_b2, "r": False},
                {"x": flip_b2 * (b_2), "y": depth, "n": n_l2, "r": False},
                {"x": flip_b2 * (b_2 + l_2 * np.cos(q_2)), "y": depth - l_2 * np.sin(q_2), "n": 0, "r": False},
            ]

    else:
        # Unlipped C or Z with round corners
        if l_1 == 0 and l_2 == 0:
            geom = [
                {"x": rad + b_1, "y": 0, "n": n_b1, "r": False},
                {"x": rad, "y": 0, "n": n_r, "r": True},
                {"x": 0, "y": rad, "n": n_d, "r": False},
                {"x": 0, "y": rad + depth, "n": n_r, "r": True},
                {"x": flip_b2 * rad, "y": rad + depth + rad, "n": n_b2, "r": False},
                {"x": flip_b2 * (rad + b_2), "y": rad + depth + rad, "n": 0, "r": False},
            ]
        # lipped C or Z with round corners
        else:
            geom = [
                {
                    "x": rad + b_1 + rad * np.cos(np.pi / 2 - q_1) + l_1 * np.cos(q_1),
                    "y": rad - rad * np.sin(np.pi / 2 - q_1) + l_1 * np.sin(q_1),
                    "n": n_l1,
                    "r": False,
                },
                {
                    "x": rad + b_1 + rad * np.cos(np.pi / 2 - q_1),
                    "y": rad - rad * np.sin(np.pi / 2 - q_1),
                    "n": n_r,
                    "r": True,
                },
                {"x": rad + b_1, "y": 0, "n": n_b1, "r": False},
                {"x": rad, "y": 0, "n": n_r, "r": True},
                {"x": 0, "y": rad, "n": n_d, "r": False},
                {"x": 0, "y": rad + depth, "n": n_r, "r": True},
                {"x": flip_b2 * rad, "y": rad + depth + rad, "n": n_b2, "r": False},
                {"x": flip_b2 * (rad + b_2), "y": rad + depth + rad, "n": n_r, "r": True},
                {
                    "x": flip_b2 * (rad + b_2 + rad * np.cos(np.pi / 2 - q_2)),
                    "y": rad + depth + rad - rad + rad * np.sin(np.pi / 2 - q_2),
                    "n": n_l2,
                    "r": False,
                },
                {
                    "x": flip_b2 * (rad + b_2 + rad * np.cos(np.pi / 2 - q_2) + l_2 * np.cos(q_2)),
                    "y": rad + depth + rad - rad + rad * np.sin(np.pi / 2 - q_2) - l_2 * np.sin(q_2),
                    "n": 0,
                    "r": False,
                },
            ]

    # mesh it
    # number of elements between the geom coordinates
    for i, (g_1, g_2) in enumerate(zip(geom[:-1], geom[1:])):
        d_x = g_2["x"] - g_1["x"]
        d_y = g_2["y"] - g_1["y"]
        # nodes.append(
        #    np.array([len(nodes), g_1['x'], g_1['y'], 1, 1, 1, 1, 1.0]))

        if g_1["r"]:
            # ROUND CORNER MODEL
            if l_1 == 0 and l_2 == 0:
                # ------------------------
                # UNLIPPED C OR Z SECTION
                for j in range(0, g_1["n"]):
                    if i == 1:
                        x_c = rad
                        y_c = rad
                        q_start = np.pi / 2
                        d_q = np.pi / 2 * j / g_1["n"]
                    elif i == 3:
                        x_c = flip_b2 * rad
                        y_c = rad + depth
                        q_start = np.pi if flip_b2 == 1 else 0
                        d_q = flip_b2 * np.pi / 2 * j / g_1["n"]
                    else:
                        raise ValueError("Invalid geometry")
                    x_2 = x_c + rad * np.cos(q_start + d_q)
                    # note sign on 2nd term is negative due to y sign convention (down positive)
                    y_2 = y_c - rad * np.sin(q_start + d_q)
                    nodes.append(np.array([len(nodes), x_2, y_2, 1, 1, 1, 1, 1.0]))
                # ------------------------
            else:
                # ------------------------
                # LIPPED C OR Z SECTION
                # we are in a corner and must be fancier
                for j in range(0, g_1["n"]):
                    if i == 1:
                        x_c = rad + b_1
                        y_c = rad
                        q_start = np.pi / 2 - q_1
                        d_q = q_1 * j / g_1["n"]
                    if i == 3:
                        x_c = rad
                        y_c = rad
                        q_start = np.pi / 2
                        d_q = np.pi / 2 * j / g_1["n"]
                    if i == 5:
                        x_c = flip_b2 * rad
                        y_c = rad + depth
                        q_start = np.pi if flip_b2 == 1 else 0
                        d_q = flip_b2 * np.pi / 2 * j / g_1["n"]
                    if i == 7:
                        x_c = flip_b2 * (rad + b_2)
                        y_c = rad + depth + rad - rad
                        q_start = 3 * np.pi / 2
                        d_q = flip_b2 * q_2 * j / g_1["n"]
                    x_2 = x_c + rad * np.cos(q_start + d_q)
                    # note sign on 2nd term is negative due to y sign convention (down positive)
                    y_2 = y_c - rad * np.sin(q_start + d_q)
                    nodes.append(np.array([len(nodes), x_2, y_2, 1, 1, 1, 1, 1.0]))
                # ------------------------
        else:
            # ------------------------
            # FLAT SECTION
            for j in range(0, g_1["n"]):
                nodes.append(
                    np.array(
                        [len(nodes), g_1["x"] + d_x * j / g_1["n"], g_1["y"] + d_y * j / g_1["n"], 1, 1, 1, 1, 1.0]
                    )
                )
            # ------------------------

    # GET THE LAST NODE ASSIGNED
    if rad == 0:
        if l_1 == 0 and l_2 == 0:
            i = 3
            nodes.append(np.array([len(nodes), geom[-1]["x"], geom[-1]["y"], 1, 1, 1, 1, 1.0]))
        else:
            i = 6
            nodes.append(np.array([len(nodes), geom[-1]["x"], geom[-1]["y"], 1, 1, 1, 1, 1.0]))
    else:
        if l_1 == 0 and l_2 == 0:
            i = 6
            nodes.append(np.array([len(nodes), geom[-1]["x"], geom[-1]["y"], 1, 1, 1, 1, 1.0]))
        else:
            i = 10
            nodes.append(np.array([len(nodes), geom[-1]["x"], geom[-1]["y"], 1, 1, 1, 1, 1.0]))

    for i in range(0, len(nodes) - 1):
        elements.append(np.array([i, i, i + 1, thick, 0]))

    return np.array(nodes), np.array(elements)


def template_out_to_in(sect: Sect_Geom) -> list:
    """For template calc, convert outer dimensions and inside radii to centerline
    dimensions throughout convert the inner radii to centerline if nonzero.
    Reference AISI Design Manual for the lovely corner radius calcs.

    Args:
        sect (Sect_Geom): _description_

    Returns:
        list: _description_

    BWS, 2015

    """
    #
    if sect["type"] == "C":
        b_1 = sect["b_1"]
        b_2 = sect["b_2"]
    else:
        b_1 = sect["b_1"]
        b_2 = sect["b_2"]

    thick: float = sect["t"]
    if sect["r_out"] == 0:
        rad: float = 0
    else:
        rad = sect["r_out"] - thick / 2
    depth = sect["d"] - thick / 2 - rad - rad - thick / 2

    if sect["l_1"] == 0:
        b_1 = b_1 - rad - thick / 2
        l_1 = 0
    else:
        b_1 = b_1 - rad - thick / 2 - (rad + thick / 2) * np.tan(np.pi / 4)
        l_1 = sect["l_1"] - (rad + thick / 2) * np.tan(np.pi / 4)
    if sect["l_2"] == 0:
        b_2 = b_2 - rad - thick / 2
        l_2 = 0
    else:
        b_2 = b_2 - rad - thick / 2 - (rad + thick / 2) * np.tan(np.pi / 4)
        l_2 = sect["l_2"] - (rad + thick / 2) * np.tan(np.pi / 4)
    return [depth, b_1, l_1, b_2, l_2, rad, thick]


def yield_mp(nodes: np.ndarray, f_y: float, sect_props: Sect_Props, restrained: bool = False) -> Forces:
    """Determine yield strengths in bending and axial loading

    Args:
        nodes (np.ndarray): _description_
        f_y (float): _description_
        sect_props (Sect_Props): _description_
        restrained (bool, optional): _description_. Defaults to False.

    Returns:
        forces (Forces): Yield bending and axial strengths
            {Py,Mxx_y,Mzz_y,M11_y,M22_y}

    BWS, Aug 2000
    BWS, May 2019 trap nan when flat plate or other properites are zero
    """
    f_yield: Forces = {"P": 0, "Mxx": 0, "Myy": 0, "M11": 0, "M22": 0, "restrain": restrained, "offset": [0, 0]}

    f_yield["P"] = f_y * sect_props["A"]

    # account for the possibility of restrained bending vs. unrestrained bending
    if restrained is False:
        sect_props["Ixy"] = 0
    # Calculate stress at every point based on m_xx=1
    m_xx = 1
    m_yy = 0
    stress1 = np.zeros((1, len(nodes)))
    stress1 = stress1 - (
        (m_yy * sect_props["Ixx"] + m_xx * sect_props["Ixy"]) * (nodes[:, 1] - sect_props["cx"])
        - (m_yy * sect_props["Ixy"] + m_xx * sect_props["Iyy"]) * (nodes[:, 2] - sect_props["cy"])
    ) / (sect_props["Iyy"] * sect_props["Ixx"] - sect_props["Ixy"] ** 2)
    if np.max(abs(stress1)) == 0:
        f_yield["Mxx"] = 0
    else:
        f_yield["Mxx"] = f_y / np.max(abs(stress1))
    # Calculate stress at every point based on m_yy=1
    m_xx = 0
    m_yy = 1
    stress1 = np.zeros((1, len(nodes)))
    stress1 = stress1 - (
        (m_yy * sect_props["Ixx"] + m_xx * sect_props["Ixy"]) * (nodes[:, 1] - sect_props["cx"])
        - (m_yy * sect_props["Ixy"] + m_xx * sect_props["Iyy"]) * (nodes[:, 2] - sect_props["cy"])
    ) / (sect_props["Iyy"] * sect_props["Ixx"] - sect_props["Ixy"] ** 2)
    if np.max(abs(stress1)) == 0:
        f_yield["Myy"] = 0
    else:
        f_yield["Myy"] = f_y / np.max(abs(stress1))
    # %M11_y, M22_y
    # %transform coordinates of nodes into principal coordinates
    phi = sect_props["phi"]
    transform = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    cent_coord = np.array([nodes[:, 1] - sect_props["cx"], nodes[:, 2] - sect_props["cy"]])
    prin_coord = np.transpose(spla.inv(transform) @ cent_coord)
    f_yield["M11"] = 1
    stress1 = np.zeros((1, len(nodes)))
    stress1 = stress1 - f_yield["M11"] * prin_coord[:, 1] / sect_props["I11"]
    if np.max(abs(stress1)) == 0:
        f_yield["M11"] = 0
    else:
        f_yield["M11"] = f_y / np.max(abs(stress1)) * f_yield["M11"]

    f_yield["M22"] = 1
    stress1 = np.zeros((1, len(nodes)))
    stress1 = stress1 - f_yield["M22"] * prin_coord[:, 0] / sect_props["I22"]
    if np.max(abs(stress1)) == 0:
        f_yield["M22"] = 0
    else:
        f_yield["M22"] = f_y / np.max(abs(stress1)) * f_yield["M22"]
    return f_yield


def stress_gen(
    nodes: np.ndarray,
    forces: Forces,
    sect_props: Sect_Props,
    restrained: bool = False,
    offset_basis: Union[int, list] = 0,
) -> np.ndarray:
    """Generates stresses on nodes based upon applied loadings

    Args:
        nodes (np.ndarray): _description_
        forces (Forces): _description_
        sect_props (Sect_Props): _description_
        restrained (bool, optional): _description_. Defaults to False.
        offset_basis (Union[int, list], optional): offset_basis compensates for section properties
            that are based upon coordinate
            [0, 0] being something other than the centreline of elements. For example,
            if section properties are based upon the outer perimeter, then
            offset_basis=[-thickness/2, -thickness/2]. Defaults to 0.

    Returns:
        np.ndarray: _description_

    BWS, 1998
    B Smith, Aug 2020
    """
    if "restrain" in forces:
        restrained = forces["restrain"]
    if "offset" in forces and forces["offset"] is not None:
        offset_basis = list(forces["offset"])
    if isinstance(offset_basis, (float, int)):
        offset_basis = [offset_basis, offset_basis]

    stress = np.zeros((1, len(nodes)))
    stress = stress + forces["P"] / sect_props["A"]
    if restrained:
        stress = stress - (
            (forces["Myy"] * sect_props["Ixx"]) * (nodes[:, 1] - sect_props["cx"] - offset_basis[0])
            - (forces["Mxx"] * sect_props["Iyy"]) * (nodes[:, 2] - sect_props["cy"] - offset_basis[1])
        ) / (sect_props["Iyy"] * sect_props["Ixx"])
    else:
        stress = stress - (
            (forces["Myy"] * sect_props["Ixx"] + forces["Mxx"] * sect_props["Ixy"])
            * (nodes[:, 1] - sect_props["cx"] - offset_basis[0])
            - (forces["Myy"] * sect_props["Ixy"] + forces["Mxx"] * sect_props["Iyy"])
            * (nodes[:, 2] - sect_props["cy"] - offset_basis[1])
        ) / (sect_props["Iyy"] * sect_props["Ixx"] - sect_props["Ixy"] ** 2)
    phi = sect_props["phi"] * np.pi / 180
    transform = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    cent_coord = np.array(
        [nodes[:, 1] - sect_props["cx"] - offset_basis[0], nodes[:, 2] - sect_props["cy"] - offset_basis[1]]
    )
    prin_coord = np.transpose(spla.inv(transform) @ cent_coord)
    stress = stress - forces["M11"] * prin_coord[:, 1] / sect_props["I11"]

    stress = stress - forces["M22"] * prin_coord[:, 0] / sect_props["I22"]
    nodes[:, 7] = stress.flatten()
    return nodes


def doubler(nodes: np.ndarray, elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """A function to double the number of elements to help
    out the discretization of the member somewhat.

    Args:
        nodes (np.ndarray): [node# x z dofx dofz dofy doftheta stress]
        elements (np.ndarray):[elem# nodei nodej thickness]

    Returns:
        _type_: _description_

    BWS, 1998 (last modified)
    """
    old_num_elem = len(elements)
    old_num_node = len(nodes)
    elem_out = np.zeros((2 * old_num_elem, 5))
    node_out = np.zeros((old_num_elem + old_num_node, 8))
    # %For node_out set all the old numbers to odd numbers and fill in the
    # %new ones with even numbers.
    for i in range(old_num_node):
        node_out[2 * i, 0] = 2 * nodes[i, 0]
        node_out[2 * i, 1:8] = nodes[i, 1:8]

    for i in range(old_num_elem):
        elem_out[2 * i, :] = [2 * elements[i, 0], 2 * elements[i, 1], 2 * i + 1, elements[i, 3], elements[i, 4]]
        elem_out[2 * i + 1, :] = [2 * i + 1, 2 * i + 1, 2 * elements[i, 2], elements[i, 3], elements[i, 4]]
        nnumi = int(elements[i, 1])
        nnumj = int(elements[i, 2])
        xcoord = np.mean([nodes[nnumi, 1], nodes[nnumj, 1]])
        zcoord = np.mean([nodes[nnumi, 2], nodes[nnumj, 2]])
        stress = np.mean([nodes[nnumi, 7], nodes[nnumj, 7]])
        node_out[2 * i + 1, :] = [
            2 * i + 1,
            xcoord,
            zcoord,
            nodes[nnumi, 3],
            nodes[nnumi, 4],
            nodes[nnumi, 5],
            nodes[nnumi, 6],
            stress,
        ]
    return node_out, elem_out
