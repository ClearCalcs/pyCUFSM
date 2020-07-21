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
##Cross section
def crossect(node,elem,springs,constraint,flag):
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
    if len(flag)>10:
        propaxisflag = flag[9]
    else:
        propaxisflag = 0
    if stresspicflag==1:
        scale = 1
        maxstress = max(np.abs(node[:, 7]))
        stress = np.append(node[:, 0].reshape((len(node), 1)), (node[:, 7]/maxstress).reshape((len(node), 1)), axis = 1)
        maxi = np.max(np.abs(node[:, 1:3]))
        maxoffset = scale*np.max(maxi)/10
        stresscord=np.zeros((len(node),3))
        for i in range(len(stress)):
            stresscord[i,0:3] = [node[i,0], node[i,1] + maxoffset*stress[i,1],
            node[i,2] - maxoffset*stress[i,1]]
    #Plot the nodes
    fig, ax1 = plt.subplots(constrained_layout=True, figsize=(6, 6))
    plt.plot(node[:,1], node[:,2],'bo', markersize = 2)
    #Plot the elements
    for i in range((len(elem))):
        nodei = int(elem[i, 1])
        nodej = int(elem[i, 2])
        xi = node[nodei, 1]
        zi = node[nodei, 2]
        xj = node[nodej, 1]
        zj = node[nodej, 2]
        theta = np.arctan2((zj-zi), (xj-xi))
        t = elem[i, 3]*1
        points = np.array([[xi-np.sin(theta)*t/2, zi+np.cos(theta)*t/2],
            [xj-np.sin(theta)*t/2, zj+np.cos(theta)*t/2],
            [xj+np.sin(theta)*t/2, zj-np.cos(theta)*t/2],
            [xi+np.sin(theta)*t/2, zi-np.cos(theta)*t/2]])
        plt.plot([xi, xj], [zi, zj], 'bo', markersize = 0.5 )
        polygon = Polygon(points, True, ec = 'b', fc = (1,1,0,1), lw=0.5)
        ax1.add_artist(polygon)
        #patches.append(polygon)
        if stresspicflag==1:
            #get the stresses
            sxi = stresscord[nodei, 1]
            szi = stresscord[nodei, 2]
            sxj = stresscord[nodej, 1]
            szj = stresscord[nodej, 2]
            #plot the stress in pseudo 3D
            if node[nodei, 7]>= 0:
                plt.plot([xi, sxi], [zi, szi], 'r')
            else:
                plt.plot([xi, sxi], [zi, szi], 'b')
            if node[nodej, 7]>= 0:
                plt.plot([xj, sxj], [zj, szj], 'r')
            else:
                plt.plot([xj, sxj], [zj, szj], 'b') 
            plt.plot([sxi, sxj], [szi, szj], 'k')
            if stressflag==1:
                plt.text(sxi, szi, str(node[nodei, 7]))
                plt.text(sxj, szj, str(node[nodej, 7]))
        #plot the element labels if wanted
        if elemflag == 1:
            plt.text((xi+xj)/2, (zi+zj)/2, str(elem[i,0]+1), fontsize = 8)
        #plot the materials labels if wanted
        if matflag == 1:
            plt.text((xi+xj)/2+10, (zi+zj)/2+10, str(elem[i,4]), fontsize = 8)
        #Plot th stress distribution in 3D if wanted
        #####___#####
    ####Patches of cross section
    p = PatchCollection(patches, cmap =jet, alpha=0.4)
    #colors = np.zeros(len(patches))
    #p.set_array(np.array(colors))
    #plt.add_collection(p)
    #plt.xlim((x_min - 25, x_max + 25))
    #plt.ylim((y_min - 25, y_max + 25))
    #Plot the node labels if wanted
    if nodeflag==1:
        for z in range(len(node)):
            plt.text(node[z,1],node[z,2], str(node[z,0]+1))
    #Plot the stress at the node if wanted
    if stressflag==1 and stresspicflag==0:
        for z in range(len(node)):
            plt.text(node[z,1],node[z,2], str(node[z,7]))
    #Plot the origin point
    if originflag==1:
        plt.plot(0, 0, 'ko',)
        xmax = np.max(np.max(node[:,1]))
        zmax = np.max(np.max([node[:,2]]))
        ax_len = min(xmax, zmax)
        plt.plot([0, 0.2*ax_len],[0, 0], 'k')
        plt.text(0.22*ax_len, 0, 'x_o')
        plt.plot([0, 0],[0, 0.2*ax_len], 'k')
        plt.text(0, 0.22*ax_len, 'z_o')
    if constraintsflag == 1:
        for i in range(len(node)):
            dofx = node[i, 3]
            dofz = node[i, 4]
            dofy = node[i, 5]
            dofq = node[i, 6]
            if min([dofx, dofz, dofy, dofq]) == 0:
                plt.plot(node[i,1], node[i,2], 'sq')
        if constraint == 0:
            print('No constraints')
        else:
            for i in range(len(constraint)):
                nodee = constraint[i, 0]
                nodek = constraint[i, 3]
                plt.plot(node[nodee, 1], node[nodee, 2], 'xg')
                plt.plot(node[nodek, 1], node[nodek, 2], 'hg')
    #Plot the springs if wanted
    ####SPRINGS AND CONSTRAINTS REMAINING
    springsscale = 0.05*np.max(np.max(np.abs(node[:,1:3])))
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.axis('off')
    # plt.savefig('Validation/'+address+'/CS.png')
    plt.show()