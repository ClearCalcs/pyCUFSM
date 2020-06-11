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
#Helper Function
def gammait(phi, dbar):
    p = phi
    gamma = np.array([[np.cos(p), 0, 0, 0, -np.sin(p), 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, np.cos(p), 0, 0, 0, -np.sin(p), 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
            [np.sin(p),	0,	0,	0,	np.cos(p),	0,	0,	0],
            [0,	0,	0,	0,	0,	1,	0,	0],
            [0,	0,	np.sin(p),	0,	0,	0, np.cos(p),	0],
            [0,	0,	0,	0,	0,	0,	0,	1]]) 
    d = np.dot(gamma,dbar)
    return d
#Helper Fucntion
def gammait2(phi, dl):
    p = phi
    gamma = np.array([[np.cos(p),	0,	-np.sin(p)],
    [0,	1,	0],
    [np.sin(p),	0,	np.cos(p)]])
    dlbar = np.dot(np.linalg.inv(gamma),dl)
    return dlbar
#Helper function
def shapef(links,d,b):
    inc = 1/(links)
    xb = np.linspace(inc, 1-inc, links-1)
    dl = np.zeros((3,len(xb)))
    for i in range(len(xb)):
        N1 = 1-3*xb[i]*xb[i]+2*xb[i]*xb[i]*xb[i]
        N2 = xb[i]*b*(1-2*xb[i]+xb[i]**2)
        N3 = 3*xb[i]**2-2*xb[i]**3
        N4 = xb[i]*b*(xb[i]**2-xb[i])
        N = np.array([[(1-xb[i]), 0, xb[i], 0, 0, 0, 0, 0],
        [0,	(1-xb[i]),	0,	xb[i],	0,	0,	0,	0],
        [0,	0,	0,	0,	N1,	N2,	N3,	N4]])
        dl[:, i] = np.dot(N, d).reshape(3)
    return dl
#Cross section displacement function
def dispshap(
    undef, node, elem, mode, scalem, springs, m_a, BC, SurfPos
):
    #Determining Scaling Factor for the displaced shape
    ##dispmax=np.max(np.abs(mode))
    dispmax = np.max(np.abs(mode))
    membersize = np.max(np.max(node[:, 1:2]))-np.min(np.min(node[:, 1:2]))
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
    defpoints =[]
    if undef == 1:
        for i in range(len(elem)):
            nodei = elem[i, 1]
            nodej = elem[i, 2]
            xi = node[nodei, 1]
            xj = node[nodej, 1]
            zi = node[nodei, 2]
            zj = node[nodej, 2]
            #PLOT undeformed geometry
            theta = np.arctan2((zj-zi) , (xj-xi))
            t = elem[i,3]
            points = np.array([[xi-np.sin(theta)*t/2, zi+np.cos(theta)*t/2],
            [xj-np.sin(theta)*t/2, zj+np.cos(theta)*t/2],
            [xj+np.sin(theta)*t/2, zj-np.cos(theta)*t/2],
            [xi+np.sin(theta)*t/2, zi-np.cos(theta)*t/2]])
            x_max = max(x_max, np.max(points[:,0]))
            y_max = max(y_max, np.max(points[:,1]))
            x_min = min(x_min, np.min(points[:,0]))
            y_min = min(y_min, np.min(points[:,1]))
            #points = np.random.rand(5 ,2)
            polygon = Polygon(points, True)
            patches.append(polygon)
    p = PatchCollection(patches, cmap =jet, alpha=0.4)
    colors = np.zeros(len(patches))
    p.set_array(np.array(colors))
    ax.add_collection(p)
    plt.xlim((x_min - 25, x_max + 25))
    plt.ylim((y_min - 25, y_max + 25))
    nnodes = len(node)
    for i in range(len(elem)):
        #Get Element Geometry
        nodei = elem[i, 1]
        nodej = elem[i, 2]
        xi = node[nodei, 1]
        xj = node[nodej, 1]
        zi = node[nodei, 2]
        zj = node[nodej, 2]
        #Determine the global element displacements
        #dbar is the nodal displacements for the element in global 
        #coordinates dbar=[u1 v1 u2 v2 w1 o1 w2 o2]
        dbar = np.zeros((8,1))
        dbarm = np.zeros((8,1))
        dlbarm = np.zeros((3,9))
        totalm = len(m_a)
        for z in range(len(m_a)):
            dbar[0:2,0] = mode[4*nnodes*z+2*(nodei+1)-2:4*nnodes*z+2*(nodei+1)]
            dbar[2:4,0] = mode[4*nnodes*z+2*(nodej+1)-2:4*nnodes*z+2*(nodej+1)]
            dbar[4:6,0] = mode[4*nnodes*z+2*nnodes+2*(nodei+1)-2:4*nnodes*z+2*nnodes+2*(nodei+1)]
            dbar[6:8,0] = mode[4*nnodes*z+2*nnodes+2*(nodej+1)-2:4*nnodes*z+2*nnodes+2*(nodej+1)]
            #Transform dbar into local coordinates
            phi = np.arctan2(-(zj-zi) , (xj-xi))
            d = gammait(phi, dbar)
            #Determine additional displacements in each element
            links = 10
            b = np.sqrt((xj-xi)**2+(zj-zi)**2)
            dl = shapef(links, d, b)
            #Transform additional displacements into global coordinates
            dlbar = gammait2(phi, dl)
            cutloc = 1/SurfPos
            if BC.startswith('S-S'):
                dbarm = dbar*np.sin(m_a[z]*np.pi/cutloc)+dbarm
                dlbarm = dlbar*np.sin(m_a[z]*np.pi/cutloc)+dlbarm
            elif BC.startswith('C-C'):
                dbarm = dbar*np.sin(m_a[z]*np.pi/cutloc)*sin(np.pi/cutloc)+dbarm
                dlbarm = dlbar*np.sin(m_a[z]*np.pi/cutloc)*sin(np.pi/cutloc)+dlbarm
            elif BC.startswith('S-C') or BC.startswith('C-S'):
                dbarm = dbar*(np.sin((m_a[z]+1)*np.pi/cutloc)+(m_a[z]+1)*np.sin(np.pi/cutloc)/m_a[z])+dbarm
                dlbarm = dlbar*(np.sin((m_a[z]+1)*np.pi/cutloc)+(m_a[z]+1)*np.sin(np.pi/cutloc)/m_a[z])+dlbarm
            elif BC.startswith('F-C') or BC.startswith('C-F'):
                dbarm = dbar*(1-np.cos((m_a[z]-1/2)*np.pi/cutloc))+dbarm
                dlbarm = dlbar*(1-np.cos((m_a[z]-1/2)*np.pi/cutloc))+dlbarm
            elif BC.startswith('G-C') or BC.startswith('C-G'):
                dbarm = dbar*(sin((m_a[z]-1/2)*pi/cutloc)*sin(pi/cutloc/2))+dbarm
                dlbarm = dlbar*(sin((m_a[z]-1/2)*pi/cutloc)*sin(pi/cutloc/2))+dlbarm
        #Create a vertor of undisplaced coordinates "undisp"
        undisp = np.zeros((2, links+1))
        undisp[:, 0] = np.transpose([xi, zi])
        undisp[:, links] = np.transpose([xj, zj])
        for j in range(1, links):
            undisp[:, j] = np.transpose([xi+(xj-xi)*(j)/links, zi+(zj-zi)*(j)/links])
        #create a vector of displaced coordinated "disp"
        disp = np.zeros((2, links+1))
        disp[:, 0] = np.transpose([xi+scale*dbarm[0], zi+scale*dbarm[4]])
        disp[:, links] = np.transpose([xj+scale*dbarm[2], zj+scale*dbarm[6]])
        disp[0, 1:links] = undisp[0, 1:links] + scale*dlbarm[0, :]
        disp[1, 1:links] = undisp[1, 1:links] + scale*dlbarm[2, :]
        #The angle of each link
        thetalinks = np.arctan2(disp[1, 1:links+1]-disp[1, 0:links] , disp[0, 1:links+1]-disp[0, 0:links])
        thetalinks = np.append(thetalinks, thetalinks[links-1])
        #Plot the deformed geometry
        theta = np.arctan2((zj-zi),(xj-xi))
        t = elem[i,3]
        #Deformed geomtery with appropriate thickness
        dispout = np.array([[disp[0, :] + np.sin(thetalinks)*t/2], [disp[1, :] - np.cos(thetalinks)*t/2]]).T
        dispin = np.array([[disp[0, :] - np.sin(thetalinks)*t/2], [disp[1, :] + np.cos(thetalinks)*t/2]]).T
        dispout = dispout.reshape((11,2))
        dispin = dispin.reshape((11,2))
        for j in range(links):
            defpoints = np.array([[dispout[j, 0], dispout[j, 1]],
            [dispin[j, 0], dispin[j, 1]],
            [dispin[j+1, 0], dispin[j+1, 1]], 
            [dispout[j+1, 0], dispout[j+1,1]]])
            defpolygon = Polygon(defpoints, True)
            defpatches.append(defpolygon)
    dp = PatchCollection(defpatches, cmap=jet, alpha=0.4)
    dcolors = 100*np.random.rand(len(patches))
    dp.set_array(np.array(dcolors))
    ax.add_collection(dp)
    plt.show()