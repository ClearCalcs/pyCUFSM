import numpy as np
def cutwp_prop2(coord, ends):
    ###Calculates Section Properties
    nele = len(ends)
    node = ends[:, 0:2]
    # node = node(:)
    # nnode = 0
    # j = 0
    # while len(node)>0:
    #     i = 
    nodes = np.append(node[:,0], node[:, 1])
    nodes = set(nodes)
    # j = len(nodes)-1
    # if j == nele:
    #     section = 'close'
    # elif j == nele - 1:
    #     section = 'open'
    # else:
    #     section = 'arbitrary'
    
    # #if the section is closed re-order the elements
    # if (section == 'close'):
    #     xnele = nele - 1
    #     for i in range(xnele):
    #         en = ends
    #         en[i, 1] = 0

    ###Find the element properties
    t = np.zeros(len(ends))
    xm = np.zeros(len(ends))
    ym = np.zeros(len(ends))
    xd = np.zeros(len(ends))
    yd = np.zeros(len(ends))
    L = np.zeros(len(ends))
    for i in range(nele):
        sn = int(ends[i, 0])
        fn = int(ends[i, 1])
        t[i] = ends[i, 2]
        #Compute coordinate of midpoint of the element
        xm[i] = np.mean([coord[sn, 0], coord[fn, 0]])
        ym[i] = np.mean([coord[sn, 1], coord[fn, 1]])
        #Compute the dimension of the element
        xd[i] = np.diff([coord[sn, 0], coord[fn, 0]])
        yd[i] = np.diff([coord[sn, 1], coord[fn, 1]])
        #Compute length
        L[i] = np.sqrt(xd[i]**2 + yd[i]**2)
    #Compute Area
    A = np.sum(L*t)
    #Compute centroid
    xc = np.sum(L*t*xm)/A
    yc = np.sum(L*t*ym)/A

    if np.abs(xc/np.sqrt(A))<1e-12:
        xc = 0
    if np.abs(yc/np.sqrt(A))<1e-12:
        yc = 0

    #Compute moment of inertia
    Ix = np.sum((yd**2/12 + (ym-yc)**2)*L*t)
    Iy = np.sum((xd**2/12 + (xm-xc)**2)*L*t)
    Ixy = np.sum((xd*yd/12 + (xm-xc)*(ym-yc)*L*t))

    if np.abs(Ixy/A**2)<1e-12:
        Ixy = 0
    #Compute rotation angle for the principal axes
    theta = (np.angle([(Ix-Iy)-2*Ixy*1j])/2)[0]

    #Transfer section coordinates to the centroid principal coordinates
    coord12 = np.zeros((len(coord), 2))
    coord12[:, 0] = coord[:, 0] - xc
    coord12[:, 1] = coord[:, 1] - yc
    coord12 = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]@(np.array(coord12).T)
    coord12 = np.array(coord12).T

    #Find the element properties
    for i in range(nele):
        sn = int(ends[i, 0])
        fn = int(ends[i, 1])
        #Compute coordinate of midpoint of the element
        xm[i] = np.mean([coord12[sn, 0], coord12[fn, 0]])
        ym[i] = np.mean([coord12[sn, 1], coord12[fn, 1]])
        #Compute the dimension of the element
        xd[i] = np.diff([coord12[sn, 0], coord12[fn, 0]])
        yd[i] = np.diff([coord12[sn, 1], coord12[fn, 1]])
    #Compute the principal moment of inertia
    I1 = np.sum((yd**2/12 + (ym)**2)*L*t)
    I2 = np.sum((xd**2/12 + (xm)**2)*L*t)
    section = 'open'
    if section == 'open':
        #Compute torsional constant
        Ja = np.sum(L*t**3)/3
        #Compute shear center and initialize variables
        nnode = len(coord)
        w = np.zeros((nnode, 2))
        w[int(ends[0, 0]), 0] = int(ends[0, 0])+1
        wo = np.zeros((nnode, 2))
        wo[int(ends[0, 0]), 0] = int(ends[0, 0])+1
        Iwx = 0
        Iwy = 0
        wno = 0
        Cw = 0
        ends[:,0:2] = (ends[:, 0:2]) + 1
        for m in range(nele):
            i = 0
            while(i < len(ends) - 1 and ((np.any(w[:, 0]==ends[i, 0]) and np.any(w[:, 0]==ends[i, 1])) or (not (np.any(w[:, 0]==ends[i, 0])) and (not np.any(w[:, 0]==ends[i, 1]))))):
                i = i+1
            sn = int(ends[i, 0]) - 1
            fn = int(ends[i, 1]) - 1
            p = ((coord[sn, 0]-xc)*(coord[fn, 1]-yc)-(coord[fn, 0]-xc)*(coord[sn, 1]-yc))/L[i]  
            if w[sn, 0] == 0:
                w[sn, 0] = sn+1
                w[sn, 1] = w[fn, 1]-p*L[i]
            elif w[fn, 0] == 0:
                w[fn, 0] = fn+1
                w[fn, 1] = w[sn, 1]+p*L[i]
            Iwx = Iwx+(1/3*(w[sn,1]*(coord[sn, 0]-xc)+w[fn, 1]*(coord[fn, 0]-xc))+1/6*(w[sn,1]*(coord[fn, 0]-xc)+w[fn, 1]*(coord[sn, 0]-xc)))*t[i]* L[i]
            Iwy = Iwy+(1/3*(w[sn,1]*(coord[sn, 1]-yc)+w[fn, 1]*(coord[fn, 1]-yc))+1/6*(w[sn,1]*(coord[fn, 1]-yc)+w[fn, 1]*(coord[sn, 1]-yc)))*t[i]* L[i]
        if (Ix*Iy-Ixy**2)!=0:
            xs = (Iy*Iwy-Ixy*Iwx)/(Ix*Iy-Ixy**2)+xc
            ys = -(Ix*Iwx-Ixy*Iwy)/(Ix*Iy-Ixy**2)+yc
        else:
            xs = xc
            ys = yc
        if np.abs(xs/np.sqrt(A))<1e-12:
            xs = 0
        if np.abs(ys/np.sqrt(A))<1e-12:
            ys = 0
        #Compute unit warping
        for m in range(nele):
            i = 0
            while(i < len(ends) - 1 and ((np.any(w[:, 0]==ends[i, 0]) and np.any(w[:, 0]==ends[i, 1])) or (not (np.any(w[:, 0]==ends[i, 0])) and (not np.any(w[:, 0]==ends[i, 1]))))):
                i = i+1
            sn = int(ends[i, 0]) - 1
            fn = int(ends[i, 0]) - 1
            po = ((coord[sn, 0]-xs)*(coord[fn, 1]-ys)-(coord[fn, 0]-xs)*(coord[sn, 1]-ys))/L[i]  
            if w[sn, 0] == 0:
                w[sn, 0] = sn+1
                w[sn, 1] = w[fn, 1]-po*L[i]
            elif w[fn, 0] == 0:
                w[fn, 0] = fn+1
                w[fn, 1] = w[sn, 1]+po*L[i]
            wno = wno + 1/(2*A)*(wo[sn, 1]+wo[fn, 1])*t[i]*L[i]
        wn = np.zeros((len(wo), 2))
        wn = wno - wo[:, 1]
        #Compute the warping constant
        for i in range(nele):
            sn = int(ends[i, 0]) - 1
            fn = int(ends[i, 1]) - 1
            Cw = Cw + 1/3*(wn[sn]**2+wn[sn]*wn[fn]+wn[fn]**2)*t[i]*L[i]
        #transfer the shear center coordinates to the centroid principal coordinates
        s12 = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]@(np.array([xs-xc, ys-yc]).T)
        #compute the polar radius of gyration of cross section about shear center
        ro = np.sqrt((I1+I2)/A+s12[0]**2+s12[1]**2)

        #Compute B1 and B2
        B1 = 0
        B2 = B1
        for i in range(nele):
            sn = int(ends[i, 0])-1
            fn = int(ends[i, 1])-1
            x1 = coord12[sn, 0]
            y1 = coord12[sn, 1]
            x2 = coord12[fn, 0]
            y2 = coord12[fn, 1]
            B1 = B1 + ((y1+y2)*(y1**2+y2**2)/4+(y1*(2*x1**2+(x1+x2)**2)+y2*(2*x2**2+(x1+x2)**2))/12)*L[i]*t[i]
            B2 = B2 + ((x1+x2)*(x1**2+x2**2)/4+(x1*(2*y1**2+(y1+y2)**2)+x2*(2*y2**2+(y1+y2)**2))/12)*L[i]*t[i]
        B1 = B1/I1 - 2*s12[1]
        B2 = B2/I2 - 2*s12[0]

        if np.abs(B1/np.sqrt(A)<1e-12):
            B1 = 0
        if np.abs(B2/np.sqrt(A)<1e-12):
            B2 = 0  
    ends[:,0:2] = (ends[:, 0:2]) - 1
    sect_props = {
        'A' : A,
        'cx' : xc,
        'cy' : yc,
        'Ixx' : Ix,
        'Iyy' : Iy,
        'Ixy' : Ixy,
        'phi' : theta,
        'I1' : I1,
        'I2' : I2,
        'J' : Ja,
        'xs' : xs,
        'ys' : ys,
        'Cw' : Cw,
        'B1' : B1,
        'B2' : B2,
        'wn' : wn
    }
    return sect_props