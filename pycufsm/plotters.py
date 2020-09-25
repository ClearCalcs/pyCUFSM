from copy import deepcopy
from scipy import linalg as spla
import numpy as np
import pycufsm.analysis
import pycufsm.cfsm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.cm import jet
import pycufsm.helpers as helpers


##Cross section
def crossect(node, elem, springs, constraint, flag):
    nodeflag = flag[0]
    elemflag = flag[1]
    matflag = flag[2]
    stressflag = flag[3]
    stresspicflag = flag[4]
    coordflag = flag[5]
    constraintsflag = flag[6]
    springsflag = flag[7]
    originflag = flag[8]
    patches = []
    if len(flag) > 10:
        propaxisflag = flag[9]
    else:
        propaxisflag = 0
    if stresspicflag == 1:
        scale = 1
        maxstress = max(np.abs(node[:, 7]))
        stress = np.append(
            node[:, 0].reshape((len(node), 1)), (node[:, 7]/maxstress).reshape((len(node), 1)),
            axis=1
        )
        maxi = np.max(np.abs(node[:, 1:3]))
        maxoffset = scale*np.max(maxi)/10
        stresscord = np.zeros((len(node), 3))
        for i in range(len(stress)):
            stresscord[i, 0:3] = [
                node[i, 0], node[i, 1] + maxoffset*stress[i, 1], node[i, 2] - maxoffset*stress[i, 1]
            ]
    #Plot the nodes
    fig, ax1 = plt.subplots(constrained_layout=True, figsize=(6, 6))
    plt.plot(node[:, 1], node[:, 2], 'bo', markersize=2)
    #Plot the elements
    for i in range((len(elem))):
        nodei = int(elem[i, 1])
        nodej = int(elem[i, 2])
        xi = node[nodei, 1]
        zi = node[nodei, 2]
        xj = node[nodej, 1]
        zj = node[nodej, 2]
        theta = np.arctan2((zj - zi), (xj - xi))
        t = elem[i, 3]*1
        points = np.array([[xi - np.sin(theta)*t/2, zi + np.cos(theta)*t/2],
                           [xj - np.sin(theta)*t/2, zj + np.cos(theta)*t/2],
                           [xj + np.sin(theta)*t/2, zj - np.cos(theta)*t/2],
                           [xi + np.sin(theta)*t/2, zi - np.cos(theta)*t/2]])
        plt.plot([xi, xj], [zi, zj], 'bo', markersize=0.5)
        polygon = Polygon(points, True, ec='b', fc=(1, 1, 0, 1), lw=0.5)
        ax1.add_artist(polygon)
        #patches.append(polygon)
        if stresspicflag == 1:
            #get the stresses
            sxi = stresscord[nodei, 1]
            szi = stresscord[nodei, 2]
            sxj = stresscord[nodej, 1]
            szj = stresscord[nodej, 2]
            #plot the stress in pseudo 3D
            if node[nodei, 7] >= 0:
                plt.plot([xi, sxi], [zi, szi], 'r')
            else:
                plt.plot([xi, sxi], [zi, szi], 'b')
            if node[nodej, 7] >= 0:
                plt.plot([xj, sxj], [zj, szj], 'r')
            else:
                plt.plot([xj, sxj], [zj, szj], 'b')
            plt.plot([sxi, sxj], [szi, szj], 'k')
            if stressflag == 1:
                plt.text(sxi, szi, str(node[nodei, 7]))
                plt.text(sxj, szj, str(node[nodej, 7]))
        #plot the element labels if wanted
        if elemflag == 1:
            plt.text((xi + xj)/2, (zi + zj)/2, str(elem[i, 0] + 1), fontsize=8)
        #plot the materials labels if wanted
        if matflag == 1:
            plt.text((xi + xj)/2 + 10, (zi + zj)/2 + 10, str(elem[i, 4]), fontsize=8)
        #Plot th stress distribution in 3D if wanted
        #####___#####
    ####Patches of cross section
    p = PatchCollection(patches, cmap=jet, alpha=0.4)
    #colors = np.zeros(len(patches))
    #p.set_array(np.array(colors))
    #plt.add_collection(p)
    #plt.xlim((x_min - 25, x_max + 25))
    #plt.ylim((y_min - 25, y_max + 25))
    #Plot the node labels if wanted
    if nodeflag == 1:
        for z in range(len(node)):
            plt.text(node[z, 1], node[z, 2], str(node[z, 0] + 1))
    #Plot the stress at the node if wanted
    if stressflag == 1 and stresspicflag == 0:
        for z in range(len(node)):
            plt.text(node[z, 1], node[z, 2], str(node[z, 7]))
    #Plot the origin point
    if originflag == 1:
        plt.plot(
            0,
            0,
            'ko',
        )
        xmax = np.max(np.max(node[:, 1]))
        zmax = np.max(np.max([node[:, 2]]))
        ax_len = min(xmax, zmax)
        plt.plot([0, 0.2*ax_len], [0, 0], 'k')
        plt.text(0.22*ax_len, 0, 'x_o')
        plt.plot([0, 0], [0, 0.2*ax_len], 'k')
        plt.text(0, 0.22*ax_len, 'z_o')
    if constraintsflag == 1:
        for i in range(len(node)):
            dofx = node[i, 3]
            dofz = node[i, 4]
            dofy = node[i, 5]
            dofq = node[i, 6]
            if min([dofx, dofz, dofy, dofq]) == 0:
                plt.plot(node[i, 1], node[i, 2], 'sq')
        if len(constraint) == 0:
            print('No constraints')
        else:
            for i in range(len(constraint)):
                nodee = constraint[i, 0]
                nodek = constraint[i, 3]
                plt.plot(node[nodee, 1], node[nodee, 2], 'xg')
                plt.plot(node[nodek, 1], node[nodek, 2], 'hg')
    #Plot the springs if wanted
    ####SPRINGS AND CONSTRAINTS REMAINING
    springsscale = 0.05*np.max(np.max(np.abs(node[:, 1:3])))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    # plt.savefig('Validation/'+address+'/CS.png')
    plt.show()


#Cross section displacement function
def dispshap(undef, node, elem, mode, scalem, springs, m_a, BC, SurfPos):
    #Determining Scaling Factor for the displaced shape
    ##dispmax=np.max(np.abs(mode))
    dispmax = np.max(np.abs(mode))
    membersize = np.max(np.max(node[:, 1:2])) - np.min(np.min(node[:, 1:2]))
    scale = scalem*membersize/dispmax/10
    #Generate and Plot
    fig, ax = plt.subplots()
    nnnodes = len(node)
    patches = []
    defpatches = []
    x_max = -np.inf
    y_max = -np.inf
    x_min = np.inf
    y_min = np.inf
    defpoints = []
    if undef == 1:
        for i in range(len(elem)):
            nodei = int(elem[i, 1])
            nodej = int(elem[i, 2])
            xi = node[nodei, 1]
            xj = node[nodej, 1]
            zi = node[nodei, 2]
            zj = node[nodej, 2]
            #PLOT undeformed geometry
            theta = np.arctan2((zj - zi), (xj - xi))
            t = elem[i, 3]
            points = np.array([[xi - np.sin(theta)*t/2, zi + np.cos(theta)*t/2],
                               [xj - np.sin(theta)*t/2, zj + np.cos(theta)*t/2],
                               [xj + np.sin(theta)*t/2, zj - np.cos(theta)*t/2],
                               [xi + np.sin(theta)*t/2, zi - np.cos(theta)*t/2]])
            #Plot axis limits
            x_max = max(x_max, np.max(points[:, 0]))
            y_max = max(y_max, np.max(points[:, 1]))
            x_min = min(x_min, np.min(points[:, 0]))
            y_min = min(y_min, np.min(points[:, 1]))
            #points = np.random.rand(5 ,2)
            polygon = Polygon(points, True, ec='b', fc='y', lw=0.5)
            ax.add_artist(polygon)
            plt.plot([xi, xj], [zi, zj], 'bo', markersize=2)
    #p = PatchCollection(patches, cmap =jet, alpha=0.4)
    # colors = np.zeros(len(patches))
    # p.set_array(np.array(colors))
    #ax.add_collection(p)
    #plt.xlim((x_min - 25, x_max + 25))
    #plt.ylim((y_min - 25, y_max + 25))
    nnodes = len(node)
    for i in range(len(elem)):
        #Get Element Geometry
        nodei = int(elem[i, 1])
        nodej = int(elem[i, 2])
        xi = node[nodei, 1]
        xj = node[nodej, 1]
        zi = node[nodei, 2]
        zj = node[nodej, 2]
        #Determine the global element displacements
        #dbar is the nodal displacements for the element in global
        #coordinates dbar=[u1 v1 u2 v2 w1 o1 w2 o2]
        dbar = np.zeros((8, 1))
        dbarm = np.zeros((8, 1))
        dlbarm = np.zeros((3, 9))
        totalm = len(m_a)
        for z in range(len(m_a)):
            dbar[0:2, 0] = mode[4*nnodes*z + 2*(nodei + 1) - 2:4*nnodes*z + 2*(nodei + 1)]
            dbar[2:4, 0] = mode[4*nnodes*z + 2*(nodej + 1) - 2:4*nnodes*z + 2*(nodej + 1)]
            dbar[4:6, 0] = mode[4*nnodes*z + 2*nnodes + 2*(nodei + 1) - 2:4*nnodes*z + 2*nnodes
                                + 2*(nodei + 1)]
            dbar[6:8, 0] = mode[4*nnodes*z + 2*nnodes + 2*(nodej + 1) - 2:4*nnodes*z + 2*nnodes
                                + 2*(nodej + 1)]
            #Transform dbar into local coordinates
            phi = np.arctan2(-(zj - zi), (xj - xi))
            d = helpers.gammait(phi, dbar)
            #Determine additional displacements in each element
            links = 10
            b = np.sqrt((xj - xi)**2 + (zj - zi)**2)
            dl = helpers.shapef(links, d, b)
            #Transform additional displacements into global coordinates
            dlbar = helpers.gammait2(phi, dl)
            cutloc = 1/SurfPos
            if BC.startswith('S-S'):
                dbarm = dbar*np.sin(m_a[z]*np.pi/cutloc) + dbarm
                dlbarm = dlbar*np.sin(m_a[z]*np.pi/cutloc) + dlbarm
            elif BC.startswith('C-C'):
                dbarm = dbar*np.sin(m_a[z]*np.pi/cutloc)*sin(np.pi/cutloc) + dbarm
                dlbarm = dlbar*np.sin(m_a[z]*np.pi/cutloc)*sin(np.pi/cutloc) + dlbarm
            elif BC.startswith('S-C') or BC.startswith('C-S'):
                dbarm = dbar*(
                    np.sin((m_a[z] + 1)*np.pi/cutloc) + (m_a[z] + 1)*np.sin(np.pi/cutloc)/m_a[z]
                ) + dbarm
                dlbarm = dlbar*(
                    np.sin((m_a[z] + 1)*np.pi/cutloc) + (m_a[z] + 1)*np.sin(np.pi/cutloc)/m_a[z]
                ) + dlbarm
            elif BC.startswith('F-C') or BC.startswith('C-F'):
                dbarm = dbar*(1 - np.cos((m_a[z] - 1/2)*np.pi/cutloc)) + dbarm
                dlbarm = dlbar*(1 - np.cos((m_a[z] - 1/2)*np.pi/cutloc)) + dlbarm
            elif BC.startswith('G-C') or BC.startswith('C-G'):
                dbarm = dbar*(sin((m_a[z] - 1/2)*pi/cutloc)*sin(pi/cutloc/2)) + dbarm
                dlbarm = dlbar*(sin((m_a[z] - 1/2)*pi/cutloc)*sin(pi/cutloc/2)) + dlbarm
        #Create a vertor of undisplaced coordinates "undisp"
        undisp = np.zeros((2, links + 1))
        # undisp[:, 0] = np.transpose([xi, zi])
        # undisp[:, links] = np.transpose([xj, zj])
        for j in range(0, links + 1):
            undisp[:, j] = np.transpose([xi + (xj - xi)*(j)/links, zi + (zj - zi)*(j)/links])
        #create a vector of displaced coordinated "disp"
        disp = np.zeros((2, links + 1))
        disp[:, 0] = np.transpose([xi + scale*dbarm[0], zi + scale*dbarm[4]])
        disp[:, links] = np.transpose([xj + scale*dbarm[2], zj + scale*dbarm[6]])
        disp[0, 1:links] = undisp[0, 1:links] + scale*dlbarm[0, :]
        disp[1, 1:links] = undisp[1, 1:links] + scale*dlbarm[2, :]
        #The angle of each link
        thetalinks = np.arctan2(
            disp[1, 1:links + 1] - disp[1, 0:links], disp[0, 1:links + 1] - disp[0, 0:links]
        )
        thetalinks = np.append(thetalinks, thetalinks[links - 1])
        #Plot the deformed geometry
        theta = np.arctan2((zj - zi), (xj - xi))
        t = elem[i, 3]
        #Deformed geomtery with appropriate thickness
        dispout = np.array([[disp[0, :] + np.sin(thetalinks)*t/2],
                            [disp[1, :] - np.cos(thetalinks)*t/2]]).T
        dispin = np.array([[disp[0, :] - np.sin(thetalinks)*t/2],
                           [disp[1, :] + np.cos(thetalinks)*t/2]]).T
        dispout = dispout.reshape((11, 2))
        dispin = dispin.reshape((11, 2))
        for j in range(links):
            defpoints = np.array([[dispout[j, 0], dispout[j, 1]], [dispin[j, 0], dispin[j, 1]],
                                  [dispin[j + 1, 0], dispin[j + 1, 1]],
                                  [dispout[j + 1, 0], dispout[j + 1, 1]]])
            polygon = Polygon(defpoints, True, ec='r', fc='r', lw=0.5)
            #defpatches = defpatches.append(polygon)
            ax.add_artist(polygon)
        plt.plot([disp[0, 0], disp[0, links]], [disp[1, 0], disp[1, links]], 'bo', markersize=2)
    dp = PatchCollection(defpatches, cmap=jet, alpha=0.4)
    # dcolors = 100*np.random.rand(len(patches))
    # dp.set_array(np.array(dcolors))
    #ax.add_collection(dp)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')


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
    curvecell, clas, filedisplay, minopt, logopt, clasopt, xmin, xmax, ymin, ymax, modedisplay,
    fileindex, modeindex, picpoint
):
    curve = curvecell
    marker = '.x+*sdv^<'
    color1 = 'bgky'
    fig, ax2 = plt.subplots()
    for i in range(len(filedisplay)):
        mark = ['b', marker[(filedisplay[i]) % 10]]
        mark2 = [marker[(filedisplay[i] % 10)], ':']
        if logopt == 1:
            for j in range(len(modedisplay)):
                ax2.semilogx(
                    curve[:, modedisplay[j] - 1, 0],
                    curve[:, modedisplay[j] - 1, 1],
                    color=color1[(j%4)],
                    marker=mark[1]
                )
                # ax2.semilogx(curve_sign[:,0], curve_sign[:,1], 'k')
        else:
            for j in range(len(modedisplay)):
                ax2.plot(
                    curve[:, modedisplay[j] - 1, 0],
                    curve[:, modedisplay[j] - 1, 1],
                    color=mark[0],
                    marker=mark[1]
                )
                # ax2.plot(curve_sign[:,0], curve_sign[:,1], 'k')

        cr = 0
        handl = []
        if minopt == 1:
            for j in range(len(modedisplay)):
                for m in range(len(curve[:, 1, 1]) - 2):
                    load1 = curve[m, j, 1]
                    load2 = curve[m + 1, j, 1]
                    load3 = curve[m + 2, j, 1]
                    if load2 < load1 and load2 <= load3:
                        cr = cr + 1
                        mstring = [
                            "{0:.2f}, {0:.2f}".format(curve[m + 1, j, 0], curve[m + 1, j, 1])
                        ]
                        ax2.text(
                            curve[m + 1, j, 0], curve[m + 1, j, 1] - (ymax - ymin)/20,
                            "{0:.2f}, {0:.2f}".format(curve[m + 1, j, 0], curve[m + 1, j, 1], color = 'r')
                        )
    ax2.text(picpoint[0], picpoint[1],
     "{0:.2f}, {0:.2f}".format(curve[m + 1, j, 0], curve[m + 1, j, 1], color = 'r' )
    )
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.xlabel('length')
    plt.ylabel('load factor')
    plt.title('Buckling curve')
    plt.show()
    #set the callback of curve
