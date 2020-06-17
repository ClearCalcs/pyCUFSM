import numpy as np
from scipy import linalg as spla

# Originally developed for MATLAB by Benjamin Schafer PhD et al
# Ported to Python by Brooks Smith MEng, PE
#
# Each function within this file was originally its own separate file.
# Original MATLAB comments, especially those retaining to authorship or
# change history, have been generally retained unaltered


def template_path(draw_table, thick, n_r=4):
    # Brooks H. Smith
    # 17 June 2020
    # Assuming a uniform thickness, draws a section according to a path definition
    # draw_table = matrix of the form [[theta, dist, rad, n_s]], where:
    #              theta = starting angle, dist = length of straight segment,
    #              rad = radius of curved segment, n_s = number of mesh elements in straight
    # thick = thickness
    # n_r = number of mesh elements in curved segments

    nodes = []
    elements = []

    # Set initial point
    if draw_table[0][1] != 0:
        nodes.append(np.array([len(nodes), 0, 0, 1, 1, 1, 1, 1.0]))

    # Progress through drawing the section
    for i, row in enumerate(draw_table[:-1]):
        theta = row[0]
        dist = row[1]
        rad = row[2]
        n_s = row[3]
        next_theta = draw_table[i + 1][0]
        phi = np.mod(next_theta - theta, 2*np.pi)
        if phi > np.pi:
            phi = phi - 2*np.pi

        # Add elements in straight segment (if n_s > 1)
        for i in range(1, int(n_s)):
            x_loc = nodes[-1][1] + dist/n_s*i*np.cos(theta)
            y_loc = nodes[-1][2] + dist/n_s*i*np.sin(theta)
            nodes.append(np.array([len(nodes), x_loc, y_loc, 1, 1, 1, 1, 1.0]))

        # Add elements in curved segment
        centre = [
            nodes[-1][1] + dist*np.cos(theta) - np.sign(phi)*rad*np.sin(theta),
            nodes[-1][2] + dist*np.sin(theta) + np.sign(phi)*rad*np.cos(theta),
        ]
        if rad == 0:
            nodes.append(np.array([len(nodes), centre[0], centre[1], 1, 1, 1, 1, 1.0]))
        else:
            for i in range(int(n_r)):
                theta_i = theta + i*1.0/max(1, n_r - 1)*phi
                x_loc = centre[0] + rad*np.cos(theta_i)
                y_loc = centre[1] + rad*np.sin(theta_i)
                nodes.append(np.array([len(nodes), x_loc, y_loc, 1, 1, 1, 1, 1.0]))

    # Draw the last straight line
    theta = draw_table[-1][0]
    dist = draw_table[-1][1]
    if dist > 0:
        x_loc = nodes[-1][1] + dist*np.cos(theta)
        y_loc = nodes[-1][2] + dist*np.sin(theta)
        nodes.append(np.array([len(nodes), x_loc, y_loc, 1, 1, 1, 1, 1.0]))

    # build the elements list
    for i in range(len(nodes)):
        elements.append(np.array([i, i, i + 1, thick, 0]))

    return [np.array(nodes), np.array(elements)]


def template_calc(sect):
    n_d = 4
    n_b1 = 4
    n_b2 = 4
    n_l1 = 4
    n_l2 = 4
    n_r = 4

    # BWS
    # August 23, 2000
    # 2015 modification to allow for l_1=l_2=0 and creation of a track with same template
    # 2015 addition to allow outer dimensions and inner radii to be used
    # 2015 addition to control element discretization

    nodes = []
    elements = []

    # CorZ=determines sign conventions for flange 1=C 2=Z
    if sect['type'] == 'Z':
        flip_b2 = -1
    else:
        flip_b2 = 1

    # channel template
    # convert angles to radians
    q_1 = 90*np.pi/180
    q_2 = 90*np.pi/180

    # outer dimensions and inner radii came in and these
    # need to be corrected to all centerline for the use of this template
    [depth, b_1, l_1, b_2, l_2, rad, thick] = template_out_to_in(sect)

    # rest of the dimensions are "flat dimensions" and acceptable for modeling
    if rad == 0:
        if l_1 == 0 and l_2 == 0:
            # track or unlipped Z with sharp corners
            geom = [{
                'x': b_1,
                'y': 0,
                'n': n_b1,
                'r': False
            }, {
                'x': 0,
                'y': 0,
                'n': n_d,
                'r': False
            }, {
                'x': 0,
                'y': depth,
                'n': n_b2,
                'r': False
            }, {
                'x': flip_b2*(b_2),
                'y': depth,
                'n': 0,
                'r': False
            }]
        else:
            # lipped C or Z with sharp corners
            geom = [{
                'x': b_1 + l_1*np.cos(q_1),
                'y': l_1*np.sin(q_1),
                'n': n_l1,
                'r': False
            }, {
                'x': b_1,
                'y': 0,
                'n': n_b1,
                'r': False
            }, {
                'x': 0,
                'y': 0,
                'n': n_d,
                'r': False
            }, {
                'x': 0,
                'y': depth,
                'n': n_b2,
                'r': False
            }, {
                'x': flip_b2*(b_2),
                'y': depth,
                'n': n_l2,
                'r': False
            }, {
                'x': flip_b2*(b_2 + l_2*np.cos(q_2)),
                'y': depth - l_2*np.sin(q_2),
                'n': 0,
                'r': False
            }]

    else:
        # Unlipped C or Z with round corners
        if l_1 == 0 and l_2 == 0:
            geom = [{
                'x': rad + b_1,
                'y': 0,
                'n': n_b1,
                'r': False
            }, {
                'x': rad,
                'y': 0,
                'n': n_r,
                'r': True
            }, {
                'x': 0,
                'y': rad,
                'n': n_d,
                'r': False
            }, {
                'x': 0,
                'y': rad + depth,
                'n': n_r,
                'r': True
            }, {
                'x': flip_b2*rad,
                'y': rad + depth + rad,
                'n': n_b2,
                'r': False
            }, {
                'x': flip_b2*(rad + b_2),
                'y': rad + depth + rad,
                'n': 0,
                'r': False
            }]
        # lipped C or Z with round corners
        else:
            geom = [{
                'x': rad + b_1 + rad*np.cos(np.pi/2 - q_1) + l_1*np.cos(q_1),
                'y': rad - rad*np.sin(np.pi/2 - q_1) + l_1*np.sin(q_1),
                'n': n_l1,
                'r': False
            }, {
                'x': rad + b_1 + rad*np.cos(np.pi/2 - q_1),
                'y': rad - rad*np.sin(np.pi/2 - q_1),
                'n': n_r,
                'r': True
            }, {
                'x': rad + b_1,
                'y': 0,
                'n': n_b1,
                'r': False
            }, {
                'x': rad,
                'y': 0,
                'n': n_r,
                'r': True
            }, {
                'x': 0,
                'y': rad,
                'n': n_d,
                'r': False
            }, {
                'x': 0,
                'y': rad + depth,
                'n': n_r,
                'r': True
            }, {
                'x': flip_b2*rad,
                'y': rad + depth + rad,
                'n': n_b2,
                'r': False
            }, {
                'x': flip_b2*(rad + b_2),
                'y': rad + depth + rad,
                'n': n_r,
                'r': True
            }, {
                'x': flip_b2*(rad + b_2 + rad*np.cos(np.pi/2 - q_2)),
                'y': rad + depth + rad - rad + rad*np.sin(np.pi/2 - q_2),
                'n': n_l2,
                'r': False
            }, {
                'x': flip_b2*(rad + b_2 + rad*np.cos(np.pi/2 - q_2) + l_2*np.cos(q_2)),
                'y': rad + depth + rad - rad + rad*np.sin(np.pi/2 - q_2) - l_2*np.sin(q_2),
                'n': 0,
                'r': False
            }]

    # mesh it
    # number of elements between the geom coordinates
    for i, (g_1, g_2) in enumerate(zip(geom[:-1], geom[1:])):
        d_x = g_2['x'] - g_1['x']
        d_y = g_2['y'] - g_1['y']
        # nodes.append(
        #    np.array([len(nodes), g_1['x'], g_1['y'], 1, 1, 1, 1, 1.0]))

        if g_1['r']:
            # ROUND CORNER MODEL
            if l_1 == 0 and l_2 == 0:
                # ------------------------
                # UNLIPPED C OR Z SECTION
                for j in range(0, g_1['n']):
                    if i == 1:
                        x_c = rad
                        y_c = rad
                        q_start = np.pi/2
                        d_q = np.pi/2*j/g_1['n']
                    if i == 3:
                        x_c = flip_b2*rad
                        y_c = rad + depth
                        q_start = np.pi if flip_b2 == 1 else 0
                        d_q = flip_b2*np.pi/2*j/g_1['n']
                    x_2 = x_c + rad*np.cos(q_start + d_q)
                    # note sign on 2nd term is negative due to y sign convention (down positive)
                    y_2 = y_c - rad*np.sin(q_start + d_q)
                    nodes.append(np.array([len(nodes), x_2, y_2, 1, 1, 1, 1, 1.0]))
                # ------------------------
            else:
                # ------------------------
                # LIPPED C OR Z SECTION
                # we are in a corner and must be fancier
                for j in range(0, g_1['n']):
                    if i == 1:
                        x_c = rad + b_1
                        y_c = rad
                        q_start = np.pi/2 - q_1
                        d_q = q_1*j/g_1['n']
                    if i == 3:
                        x_c = rad
                        y_c = rad
                        q_start = np.pi/2
                        d_q = np.pi/2*j/g_1['n']
                    if i == 5:
                        x_c = flip_b2*rad
                        y_c = rad + depth
                        q_start = np.pi if flip_b2 == 1 else 0
                        d_q = flip_b2*np.pi/2*j/g_1['n']
                    if i == 7:
                        x_c = flip_b2*(rad + b_2)
                        y_c = rad + depth + rad - rad
                        q_start = 3*np.pi/2
                        d_q = flip_b2*q_2*j/g_1['n']
                    x_2 = x_c + rad*np.cos(q_start + d_q)
                    # note sign on 2nd term is negative due to y sign convention (down positive)
                    y_2 = y_c - rad*np.sin(q_start + d_q)
                    nodes.append(np.array([len(nodes), x_2, y_2, 1, 1, 1, 1, 1.0]))
                # ------------------------
        else:
            # ------------------------
            # FLAT SECTION
            for j in range(0, g_1['n']):
                nodes.append(
                    np.array([
                        len(nodes), g_1['x'] + d_x*j/g_1['n'], g_1['y'] + d_y*j/g_1['n'], 1, 1, 1,
                        1, 1.0
                    ])
                )
            # ------------------------

    # GET THE LAST NODE ASSIGNED
    if rad == 0:
        if l_1 == 0 and l_2 == 0:
            i = 3
            nodes.append(np.array([len(nodes), geom[-1]['x'], geom[-1]['y'], 1, 1, 1, 1, 1.0]))
        else:
            i = 6
            nodes.append(np.array([len(nodes), geom[-1]['x'], geom[-1]['y'], 1, 1, 1, 1, 1.0]))
    else:
        if l_1 == 0 and l_2 == 0:
            i = 6
            nodes.append(np.array([len(nodes), geom[-1]['x'], geom[-1]['y'], 1, 1, 1, 1, 1.0]))
        else:
            i = 10
            nodes.append(np.array([len(nodes), geom[-1]['x'], geom[-1]['y'], 1, 1, 1, 1, 1.0]))

    for i in range(0, len(nodes) - 1):
        elements.append(np.array([i, i, i + 1, thick, 0]))

    return [np.array(nodes), np.array(elements)]


def template_out_to_in(sect):
    # BWS 2015
    # reference AISI Design Manual for the lovely corner radius calcs.
    # For template calc, convert outer dimensions and inside radii to centerline
    # dimensions throughout
    # convert the inner radii to centerline if nonzero
    if sect['type'] == 'C':
        b_1 = sect['b']
        b_2 = sect['b']
    else:
        b_1 = sect['b_l']
        b_2 = sect['b_r']

    thick = sect['t']
    if sect['r_out'] == 0:
        rad = 0
    else:
        rad = sect['r_out'] - thick/2
    depth = sect['d'] - thick/2 - rad - rad - thick/2

    if sect['l'] == 0:
        b_1 = b_1 - rad - thick/2
        l_1 = 0
    else:
        b_1 = b_1 - rad - thick/2 - (rad + thick/2)*np.tan(np.pi/4)
        l_1 = sect['l'] - (rad + thick/2)*np.tan(np.pi/4)
    if sect['l'] == 0:
        b_2 = b_2 - rad - thick/2
        l_2 = 0
    else:
        b_2 = b_2 - rad - thick/2 - (rad + thick/2)*np.tan(np.pi/4)
        l_2 = sect['l'] - (rad + thick/2)*np.tan(np.pi/4)
    return [depth, b_1, l_1, b_2, l_2, rad, thick]


def stress_gen(nodes, forces, sect_props, restrained=False, offset_basis=0):
    # BWS
    # 1998
    # offset_basis compensates for section properties that are based upon coordinate
    # [0, 0] being something other than the centreline of elements. For example,
    # if section properties are based upon the outer perimeter, then
    # offset_basis=[-thickness/2, -thickness/2]
    if isinstance(offset_basis, float) or isinstance(offset_basis, int):
        offset_basis = [offset_basis, offset_basis]

    stress = np.zeros((1, len(nodes)))
    stress = stress + forces['P']/sect_props['A']
    if restrained:
        stress = stress - ((forces['Myy']*sect_props['Ixx'])*
                           (nodes[:, 1] - sect_props['cx'] - offset_basis[0]) -
                           (forces['Mxx']*sect_props['Iyy'])*
                           (nodes[:, 2] - sect_props['cy'] - offset_basis[1])
                           )/(sect_props['Iyy']*sect_props['Ixx'])
    else:
        stress = stress - ((forces['Myy']*sect_props['Ixx'] + forces['Mxx']*sect_props['Ixy'])*
                           (nodes[:, 1] - sect_props['cx'] - offset_basis[0]) -
                           (forces['Myy']*sect_props['Ixy'] + forces['Mxx']*sect_props['Iyy'])*
                           (nodes[:, 2] - sect_props['cy'] - offset_basis[1])
                           )/(sect_props['Iyy']*sect_props['Ixx'] - sect_props['Ixy']**2)
    phi = sect_props['phi']*np.pi/180
    transform = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    cent_coord = np.array([
        nodes[:, 1] - sect_props['cx'] - offset_basis[0],
        nodes[:, 2] - sect_props['cy'] - offset_basis[1]
    ])
    prin_coord = np.transpose(spla.inv(transform) @ cent_coord)
    stress = stress - \
        forces['M11'] * prin_coord[:, 1] / sect_props['I11']
    stress = stress - \
        forces['M22'] * prin_coord[:, 0] / sect_props['I22']
    nodes[:, 7] = stress.flatten()
    return nodes
