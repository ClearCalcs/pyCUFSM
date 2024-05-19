from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from pycufsm import helpers
from pycufsm.analysis import analysis


def crossect(
    nodes: np.ndarray,
    elements: np.ndarray,
    springs: np.ndarray,  # pylint: disable=unused-argument
    constraints: np.ndarray,
    flags: List[int],
) -> None:
    """plot cross-section

    Args:
        nodes (np.ndarray): [node# x z dofx dofz dofy dofrot stress] nnodes x 8;
        elements (np.ndarray): [elem# nodei nodej t mat#] nelems x 4;
        springs (np.ndarray): _description_
        flags (List[int]): [node# element# mat# stress# stresspic coord constraints
            springs origin propaxis] 1 means show

    BWS
    October 2001 (last modified)
    December 2015 springs updated
    December 2015 origin updated to also show axes
    December 2015 added stress distribution into this plot instead of separate
    December 2015 added an additional flag for section property axes
    plots the cross-section
    """
    nodeflag = flags[0]
    elemflag = flags[1]
    matflag = flags[2]
    stressflag = flags[3]
    stresspicflag = flags[4]
    # coordflag = flags[5]
    constraintsflag = flags[6]
    # springsflag = flags[7]
    originflag = flags[8]
    patches: list = []
    # if len(flags) > 10:
    #     propaxisflag = flags[9]
    # else:
    #     propaxisflag = 0
    if stresspicflag == 1:
        scale = 1
        maxstress = max(np.abs(nodes[:, 7]))
        stress = np.append(
            nodes[:, 0].reshape((len(nodes), 1)),
            (nodes[:, 7] / maxstress).reshape((len(nodes), 1)),
            axis=1,
        )
        maxi = np.max(np.abs(nodes[:, 1:3]))
        maxoffset = scale * np.max(maxi) / 10
        stresscord = np.zeros((len(nodes), 3))
        for i in range(len(stress)):
            stresscord[i, 0:3] = [
                nodes[i, 0],
                nodes[i, 1] + maxoffset * stress[i, 1],
                nodes[i, 2] - maxoffset * stress[i, 1],
            ]
    # Plot the nodes
    _, ax1 = plt.subplots(constrained_layout=True, figsize=(6, 6))
    plt.plot(nodes[:, 1], nodes[:, 2], "bo", markersize=2)
    # Plot the elements
    for i in range((len(elements))):
        nodei = int(elements[i, 1])
        nodej = int(elements[i, 2])
        x_i = nodes[nodei, 1]
        z_i = nodes[nodei, 2]
        x_j = nodes[nodej, 1]
        z_j = nodes[nodej, 2]
        theta = np.arctan2((z_j - z_i), (x_j - x_i))
        thick = elements[i, 3] * 1
        points = np.array(
            [
                [x_i - np.sin(theta) * thick / 2, z_i + np.cos(theta) * thick / 2],
                [x_j - np.sin(theta) * thick / 2, z_j + np.cos(theta) * thick / 2],
                [x_j + np.sin(theta) * thick / 2, z_j - np.cos(theta) * thick / 2],
                [x_i + np.sin(theta) * thick / 2, z_i - np.cos(theta) * thick / 2],
            ]
        )
        plt.plot([x_i, x_j], [z_i, z_j], "bo", markersize=0.5)
        polygon = Polygon(xy=points, closed=True, ec="b", fc=(1, 1, 0, 1), lw=0.5)
        ax1.add_artist(polygon)
        # patches.append(polygon)
        if stresspicflag == 1:
            # get the stresses
            sxi = stresscord[nodei, 1]
            szi = stresscord[nodei, 2]
            sxj = stresscord[nodej, 1]
            szj = stresscord[nodej, 2]
            # plot the stress in pseudo 3D
            if nodes[nodei, 7] >= 0:
                plt.plot([x_i, sxi], [z_i, szi], "r")
            else:
                plt.plot([x_i, sxi], [z_i, szi], "b")
            if nodes[nodej, 7] >= 0:
                plt.plot([x_j, sxj], [z_j, szj], "r")
            else:
                plt.plot([x_j, sxj], [z_j, szj], "b")
            plt.plot([sxi, sxj], [szi, szj], "k")
            if stressflag == 1:
                plt.text(sxi, szi, str(round(nodes[nodei, 7], 2)))
                plt.text(sxj, szj, str(round(nodes[nodej, 7], 2)))
        # plot the element labels if wanted
        if elemflag == 1:
            plt.text((x_i + x_j) / 2, (z_i + z_j) / 2, str(elements[i, 0] + 1), fontsize=8)
        # plot the materials labels if wanted
        if matflag == 1:
            plt.text(
                (x_i + x_j) / 2 + 10,
                (z_i + z_j) / 2 + 10,
                str(elements[i, 4]),
                fontsize=8,
            )
        # Plot th stress distribution in 3D if wanted
        #####___#####
    ####Patches of cross section
    PatchCollection(patches, cmap="jet", alpha=0.4)
    # colors = np.zeros(len(patches))
    # patch.set_array(np.array(colors))
    # plt.add_collection(patch)
    # plt.xlim((x_min - 25, x_max + 25))
    # plt.ylim((y_min - 25, y_max + 25))
    # Plot the node labels if wanted
    if nodeflag == 1:
        for i in range(len(nodes)):
            plt.text(nodes[i, 1], nodes[i, 2], str(nodes[i, 0] + 1))
    # Plot the stress at the nodes if wanted
    if stressflag == 1 and stresspicflag == 0:
        for i in range(len(nodes)):
            plt.text(nodes[i, 1], nodes[i, 2], str(round(nodes[i, 7], 2)))
    # Plot the origin point
    if originflag == 1:
        plt.plot(
            0,
            0,
            "ko",
        )
        xmax = np.max(np.max(nodes[:, 1]))
        zmax = np.max(np.max([nodes[:, 2]]))
        ax_len = min(xmax, zmax)
        plt.plot([0, 0.2 * ax_len], [0, 0], "k")
        plt.text(0.22 * ax_len, 0, "x_o")
        plt.plot([0, 0], [0, 0.2 * ax_len], "k")
        plt.text(0, 0.22 * ax_len, "z_o")
    if constraintsflag == 1:
        for i in range(len(nodes)):
            dofx = nodes[i, 3]
            dofz = nodes[i, 4]
            dofy = nodes[i, 5]
            dofq = nodes[i, 6]
            if min([dofx, dofz, dofy, dofq]) == 0:
                plt.plot(nodes[i, 1], nodes[i, 2], "sq")
        if len(constraints) == 0:
            print("No constraints")
        else:
            for i in range(len(constraints)):
                nodee = constraints[i, 0]
                nodek = constraints[i, 3]
                plt.plot(nodes[nodee, 1], nodes[nodee, 2], "xg")
                plt.plot(nodes[nodek, 1], nodes[nodek, 2], "hg")
    # Plot the springs if wanted
    ####SPRINGS AND CONSTRAINTS REMAINING
    # springsscale = 0.05*np.max(np.max(np.abs(nodes[:, 1:3])))
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    # plt.savefig('Validation/'+address+'/CS.png')
    plt.show()


def dispshap(
    undef: int,
    nodes: np.ndarray,
    elements: np.ndarray,
    mode: np.ndarray,
    scalem: float,
    springs: np.ndarray,  # pylint: disable=unused-argument
    m_a: np.ndarray,
    B_C: str,
    surf_pos: float,
) -> None:
    """Cross section displacement function

    Args:
        undef (int): _description_
        nodes (np.ndarray): _description_
        elements (np.ndarray): _description_
        mode (np.ndarray): _description_
        scalem (float): _description_
        springs (np.ndarray): _description_
        B_C (str): _description_
        surf_pos (float): _description_
    """
    # Determining Scaling Factor for the displaced shape
    ##dispmax=np.max(np.abs(mode))
    dispmax = np.max(np.abs(mode))
    membersize = np.max(np.max(nodes[:, 1:2])) - np.min(np.min(nodes[:, 1:2]))
    scale = scalem * membersize / dispmax / 10
    # Generate and Plot
    _, axes = plt.subplots(constrained_layout=True, figsize=(6, 6))
    x_max = -np.inf
    y_max = -np.inf
    x_min = np.inf
    y_min = np.inf
    if undef == 1:
        for i in range(len(elements)):
            nodei = int(elements[i, 1])
            nodej = int(elements[i, 2])
            x_i = nodes[nodei, 1]
            x_j = nodes[nodej, 1]
            z_i = nodes[nodei, 2]
            z_j = nodes[nodej, 2]
            # PLOT undeformed geometry
            theta = np.arctan2((z_j - z_i), (x_j - x_i))
            thick = elements[i, 3]
            points = np.array(
                [
                    [x_i - np.sin(theta) * thick / 2, z_i + np.cos(theta) * thick / 2],
                    [x_j - np.sin(theta) * thick / 2, z_j + np.cos(theta) * thick / 2],
                    [x_j + np.sin(theta) * thick / 2, z_j - np.cos(theta) * thick / 2],
                    [x_i + np.sin(theta) * thick / 2, z_i - np.cos(theta) * thick / 2],
                ]
            )
            # Plot axis limits
            x_max = max(x_max, np.max(points[:, 0]))
            y_max = max(y_max, np.max(points[:, 1]))
            x_min = min(x_min, np.min(points[:, 0]))
            y_min = min(y_min, np.min(points[:, 1]))
            # points = np.random.rand(5 ,2)
            polygon = Polygon(xy=points, closed=True, ec="b", fc="y", lw=0.5)
            axes.add_artist(polygon)
            plt.plot([x_i, x_j], [z_i, z_j], "bo", markersize=2)
    # patch = PatchCollection(patches, cmap =jet, alpha=0.4)
    # colors = np.zeros(len(patches))
    # patch.set_array(np.array(colors))
    # axes.add_collection(patch)
    # plt.xlim((x_min - 25, x_max + 25))
    # plt.ylim((y_min - 25, y_max + 25))
    nnodes = len(nodes)
    for i in range(len(elements)):
        # Get Element Geometry
        nodei = int(elements[i, 1])
        nodej = int(elements[i, 2])
        x_i = nodes[nodei, 1]
        x_j = nodes[nodej, 1]
        z_i = nodes[nodei, 2]
        z_j = nodes[nodej, 2]
        # Determine the global element displacements
        # dbar is the nodal displacements for the element in global
        # coordinates dbar=[u1 v1 u2 v2 w1 o1 w2 o2]
        dbar = np.zeros((8, 1))
        dbarm = np.zeros((8, 1))
        dlbarm = np.zeros((3, 9))
        defpatches: list = []
        for j, m_a_j in enumerate(m_a):
            dbar[0:2, 0] = mode[4 * nnodes * j + 2 * (nodei + 1) - 2 : 4 * nnodes * j + 2 * (nodei + 1)]
            dbar[2:4, 0] = mode[4 * nnodes * j + 2 * (nodej + 1) - 2 : 4 * nnodes * j + 2 * (nodej + 1)]
            dbar[4:6, 0] = mode[
                4 * nnodes * j + 2 * nnodes + 2 * (nodei + 1) - 2 : 4 * nnodes * j + 2 * nnodes + 2 * (nodei + 1)
            ]
            dbar[6:8, 0] = mode[
                4 * nnodes * j + 2 * nnodes + 2 * (nodej + 1) - 2 : 4 * nnodes * j + 2 * nnodes + 2 * (nodej + 1)
            ]
            # Transform dbar into local coordinates
            phi = np.arctan2(-(z_j - z_i), (x_j - x_i))
            gamma = analysis.trans(phi, 1)
            disp = np.dot(gamma, dbar)
            # Determine additional displacements in each element
            links = 10
            length = np.sqrt((x_j - x_i) ** 2 + (z_j - z_i) ** 2)
            disp_local = helpers.shapef(links, disp, length)
            # Transform additional displacements into global coordinates
            dlbar = helpers.gammait2(phi, disp_local)
            cutloc = 1 / surf_pos
            if B_C.startswith("S-S"):
                dbarm = dbar * np.sin(m_a_j * np.pi / cutloc) + dbarm
                dlbarm = dlbar * np.sin(m_a_j * np.pi / cutloc) + dlbarm
            elif B_C.startswith("C-C"):
                dbarm = dbar * np.sin(m_a_j * np.pi / cutloc) * np.sin(np.pi / cutloc) + dbarm
                dlbarm = dlbar * np.sin(m_a_j * np.pi / cutloc) * np.sin(np.pi / cutloc) + dlbarm
            elif B_C.startswith("S-C") or B_C.startswith("C-S"):
                dbarm = (
                    dbar * (np.sin((m_a_j + 1) * np.pi / cutloc) + (m_a_j + 1) * np.sin(np.pi / cutloc) / m_a_j) + dbarm
                )
                dlbarm = (
                    dlbar * (np.sin((m_a_j + 1) * np.pi / cutloc) + (m_a_j + 1) * np.sin(np.pi / cutloc) / m_a_j)
                    + dlbarm
                )
            elif B_C.startswith("F-C") or B_C.startswith("C-F"):
                dbarm = dbar * (1 - np.cos((m_a_j - 1 / 2) * np.pi / cutloc)) + dbarm
                dlbarm = dlbar * (1 - np.cos((m_a_j - 1 / 2) * np.pi / cutloc)) + dlbarm
            elif B_C.startswith("G-C") or B_C.startswith("C-G"):
                dbarm = dbar * (np.sin((m_a_j - 1 / 2) * np.pi / cutloc) * np.sin(np.pi / cutloc / 2)) + dbarm
                dlbarm = dlbar * (np.sin((m_a_j - 1 / 2) * np.pi / cutloc) * np.sin(np.pi / cutloc / 2)) + dlbarm
        # Create a vertor of undisplaced coordinates "undisp"
        undisp = np.zeros((2, links + 1))
        # undisp[:, 0] = np.transpose([x_i, z_i])
        # undisp[:, links] = np.transpose([x_j, z_j])
        for j in range(0, links + 1):
            undisp[:, j] = np.transpose([x_i + (x_j - x_i) * (j) / links, z_i + (z_j - z_i) * (j) / links])
        # create a vector of displaced coordinated "disp"
        disp = np.zeros((2, links + 1))
        disp[:, 0] = np.transpose([x_i + scale * dbarm[0], z_i + scale * dbarm[4]])
        disp[:, links] = np.transpose([x_j + scale * dbarm[2], z_j + scale * dbarm[6]])
        disp[0, 1:links] = undisp[0, 1:links] + scale * dlbarm[0, :]
        disp[1, 1:links] = undisp[1, 1:links] + scale * dlbarm[2, :]
        # The angle of each link
        thetalinks = np.arctan2(
            disp[1, 1 : links + 1] - disp[1, 0:links],
            disp[0, 1 : links + 1] - disp[0, 0:links],
        )
        thetalinks = np.append(thetalinks, thetalinks[links - 1])
        # Plot the deformed geometry
        theta = np.arctan2((z_j - z_i), (x_j - x_i))
        thick = elements[i, 3]
        # Deformed geomtery with appropriate thickness
        dispout = np.array(
            [
                [disp[0, :] + np.sin(thetalinks) * thick / 2],
                [disp[1, :] - np.cos(thetalinks) * thick / 2],
            ]
        ).T
        dispin = np.array(
            [
                [disp[0, :] - np.sin(thetalinks) * thick / 2],
                [disp[1, :] + np.cos(thetalinks) * thick / 2],
            ]
        ).T
        dispout = dispout.reshape((11, 2))
        dispin = dispin.reshape((11, 2))
        for j in range(links):
            defpoints = np.array(
                [
                    [dispout[j, 0], dispout[j, 1]],
                    [dispin[j, 0], dispin[j, 1]],
                    [dispin[j + 1, 0], dispin[j + 1, 1]],
                    [dispout[j + 1, 0], dispout[j + 1, 1]],
                ]
            )
            polygon = Polygon(xy=defpoints, closed=True, ec="r", fc="r", lw=0.5)
            # defpatches = defpatches.append(polygon)
            axes.add_artist(polygon)
        plt.plot(
            [disp[0, 0], disp[0, links]],
            [disp[1, 0], disp[1, links]],
            "bo",
            markersize=2,
        )
    PatchCollection(defpatches, cmap="jet", alpha=0.4)
    # dcolors = 100*np.random.rand(len(patches))
    # def_patch.set_array(np.array(dcolors))
    # axes.add_collection(def_patch)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")

    # if(figure == 1):
    #     plt.savefig('Validation/'+address+'/local.png')
    # if(figure == 2):
    #     plt.savefig('Validation/'+address+'/distortional.png')
    # if(figure == 3):
    #     plt.savefig('Validation/'+address+'/global.png')
    # if(figure == 4):
    #     plt.savefig('Validation/'+address+'/global1.png')
    # plt.show()


def thecurve3(
    curvecell: np.ndarray,
    clas: int,  # pylint: disable=unused-argument
    filedisplay: list,
    minopt: int,
    logopt: int,
    clasopt: int,  # pylint: disable=unused-argument
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    modedisplay: list,
    fileindex: int,  # pylint: disable=unused-argument
    modeindex: int,  # pylint: disable=unused-argument
    picpoint: list,  # pylint: disable=unused-argument
) -> None:
    """_summary_

    Args:
        curvecell (np.ndarray): _description_
        clas (int): _description_
        minopt (int): _description_
        logopt (int): _description_
        clasopt (int): _description_
        xmax (float): _description_
        ymin (float): _description_
        ymax (float): _description_
        modedisplay (list): _description_
        fileindex (int): _description_
    """
    curve = curvecell
    marker = ".x+*sdv^<"
    color1 = "bgky"
    _, ax2 = plt.subplots(constrained_layout=True, figsize=(6, 6))
    for file_i in filedisplay:
        mark = ["b", marker[(file_i) % 10]]
        if logopt == 1:
            for j, mode in enumerate(modedisplay):
                ax2.semilogx(
                    curve[:, mode - 1, 0],
                    curve[:, mode - 1, 1],
                    color=color1[(j % 4)],
                    marker=mark[1],
                )
                # ax2.semilogx(curve_sign[:,0], curve_sign[:,1], 'k')
        else:
            for mode in modedisplay:
                ax2.plot(
                    curve[:, mode - 1, 0],
                    curve[:, mode - 1, 1],
                    color=mark[0],
                    marker=mark[1],
                )
                # ax2.plot(curve_sign[:,0], curve_sign[:,1], 'k')

        c_r = 0
        if minopt == 1:
            for mode in modedisplay:
                for j in range(len(curve[:, 1, 1]) - 2):
                    load1 = curve[j, mode - 1, 1]
                    load2 = curve[j + 1, mode - 1, 1]
                    load3 = curve[j + 2, mode - 1, 1]
                    if load2 < load1 and load2 <= load3:
                        c_r = c_r + 1
                        ax2.text(
                            curve[j + 1, mode - 1, 0],
                            curve[j + 1, mode - 1, 1] - (ymax - ymin) / 20,
                            f"{curve[j + 1, mode - 1, 0]:.2f}, {curve[j + 1, mode - 1, 1]:.2f}",
                        )
    # ax2.text(picpoint[0], picpoint[1],
    #  "{0:.2f}, {0:.2f}".format(curve[j + 1, j, 0], curve[j + 1, j, 1], color = 'r' )
    # )
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.xlabel("length")
    plt.ylabel("load factor")
    plt.title("Buckling curve")
    plt.show()
    # set the callback of curve
