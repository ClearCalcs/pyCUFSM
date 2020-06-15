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

def thecurve3(curvecell, clas, filedisplay, minopt, logopt, clasopt, xmin, xmax,
    ymin, ymax, modedisplay, fileindex, modeindex, picpoint):
    marker = '.x+*sdv^<'
    ###If clasopt == 1
    ##...
    ###
    fig, ax2 =plt.subplots()
    hndlmark = []
    hndl = []
    hnd2 = []
    hnd12 = []
    for i in range(len(filedisplay)):
        curve = []
        curve = curvecell
        templ = np.zeros((len(curve), len(modedisplay)))
        templf = np.zeros((len(curve), len(modedisplay)))
        mark = ['b', marker[(filedisplay[i])%10]]
        mark2 = [marker[(filedisplay[i]%10)],':']
        curve_sign = np.zeros((len(curve),2))
        for j in range(len(curve)):
            curve_sign[j,0] = curve[j, modedisplay[0], 0]
            curve_sign[j,1] = curve[j, modedisplay[0], 1]
            if len(modedisplay)>1:
                for mn in range(len(modedisplay)):
                    templ[j, modedisplay[mn]] = curve[j, modedisplay[mn], 0]
                    templf[j, modedisplay[mn]] = curve[j, modedisplay[mn], 1]
        if logopt == 1:
            hndlmark.append(plt.semilogx(curve_sign[:, 0], curve_sign[:,1], mark, markersize = 5))
            hndl.append(plt.semilogx(curve_sign[:,0],curve_sign[:,1], 'k'))
            #####RECHECK
            if len(modedisplay)>1:
                hndlmark.append(plt.semilogx(curve_sign[:, 0], curve_sign[:,1], marker = mark))
                hnd2.append(plt.semilogx(curve_sign[:,0], curve_sign[:,1], 'k'))
        else:
            hndlmark.append(plt.plot(curve_sign[:, 0], curve_sign[:,1], mark, markersize = 5))
            hndl.append(plt.plot(curve_sign[:,0],curve_sign[:,1], 'k'))
            #####RECHECK
            if len(modedisplay)>1:
                hndlmark.append(plt.plot(curve_sign[:, 0], curve_sign[:,1], marker = mark))
                hnd2.append(plt.plot(curve_sign[:,0], curve_sign[:,1], 'k'))
        
        cr = 0
        handl = []
        if minopt == 1:
            for m in range(len(curve_sign[:, 1])-2):
                load1 = curve_sign[m, 1]
                load2 = curve_sign[m+1, 1]
                load3 = curve_sign[m+2, 1]
                if load2<load1 and load2<=load3:
                    cr = cr+1
                    hnd12.append(plt.plot(curve_sign[m+1, 0], curve_sign[m+1, 1], 'o'))
                    mstring = ['{0:.2f},{0:.2f}'.format(curve_sign[m+1, 0], curve_sign[m+1, 1])]
                    plt.text(curve_sign[m+1, 0], curve_sign[m+1, 1]-(ymax-ymin)/20, mstring)
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))    
    plt.show()
        #set the callback of curve


