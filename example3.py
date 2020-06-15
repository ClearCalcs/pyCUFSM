# This example presents a very simple Zed section,
# solved for pure bending about the X-axis,
# in the Metric unit system
import numpy as np
import matplotlib.pyplot as plt
from pycufsm.fsm import strip
from pycufsm.preprocess import stress_gen
import dispshap
import crossect
import thecurve3
#import example4


import window
window.window()
# Define an isotropic material with E = 203,000 MPa and nu = 0.3
props = [[0, 203000, 203000, 0.30, 0.30, 78077]]
# Define a lightly-meshed C shape
# Nodal location units are inches
nodes = np.array([[0, 100, 25, 1, 1, 1, 1, 1], 
                  [1, 100, 0, 1, 1, 1, 1, 1],
                  [2, 50, 0, 1, 1, 1, 1, 1],
                  [3, 0, 0, 1, 1, 1, 1, 1],
                  [4, 0, 100, 1, 1, 1, 1, 1],
                  [5, 0, 200, 1, 1, 1, 1, 1],
                  [6, 0, 300, 1, 1, 1, 1, 1],
                  [7, -50, 300, 1, 1, 1, 1, 1],
                  [8, -100, 300, 1, 1, 1, 1, 1],
                  [9, -100, 275, 1, 1, 1, 1, 1]])
elements = np.array([[0, 0, 1, 2, 0],
                     [1, 1, 2, 2, 0],
                     [2, 2, 3, 2, 0],
                     [3, 3, 4, 2, 0],
                     [4, 4, 5, 2, 0],
                     [5, 5, 6, 2, 0],
                     [6, 6, 7, 2, 0],
                     [7, 7, 8, 2, 0],
                     [8, 8, 9, 2, 0]])
# These lengths will generally provide sufficient accuracy for
# local, distortional, and global buckling modes
# Length units are millimetres
# lengths = [
#     10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
#     170, 180, 190, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900,
#     1000, 1100, 1200, 1300, 1400, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
#     5000, 6000, 7000, 8000, 9000, 10000
# ]
###Start at a, End at b, n equal spaced intervals
#length=np.linspace(a,b,n)
###Start
lengths = np.logspace(0,4,100)

# No special springs or constraints
springs = []
constraints = []

# Values here correspond to signature curve basis and orthogonal based upon geometry
gbt_con = {
    'glob': [0],
    'dist': [0],
    'local': [0],
    'other': [0],
    'o_space': 1,
    'couple': 1,
    'orth': 2,
    'norm': 0,
}
# Simply-supported boundary conditions
b_c = 'S-S'

# For signature curve analysis, only a single array of ones makes sense here
m_all = np.ones((len(lengths), 1))

# Solve for 10 eigenvalues
n_eigs = 10

# Set the section properties for this simple section
# Normally, these might be calculated by an external package
sect_props = {
    'cx': 0,
    'cy': 150,
    'x0': 0,
    'y0': 150,
    'phi': 16.54,
    'A': 1084,
    'Ixx': 14921145,
    'Ixy': -4151084,
    'Iyy': 2177529,
    'I11': 16154036,
    'I22': 944639
}
# Generate the stress points assuming 500 MPa yield and X-axis bending
nodes_p = stress_gen(nodes=nodes,
                     forces={
                         'P': sect_props['A'] * 50,
                         'Mxx': 0,
                             'Myy': 0,
                             'M11': 0,
                             'M22': 0
                     },
                     sect_props=sect_props)
print(nodes_p)
#####CALL CROSS SECTION
#Flag = {node, elem, mat, stress, stresspic, coord, constraints, springs, origin, propaxis}
flag = np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
crossect.crossect(nodes_p, elements, springs, constraints,flag)
plt.show()
# Perform the Finite Strip Method analysis
signature, curve, shapes = strip(props=props,
                                 nodes=nodes_p,
                                 elements=elements,
                                 lengths=lengths,
                                 springs=springs,
                                 constraints=constraints,
                                 gbt_con=gbt_con,
                                 b_c=b_c,
                                 m_all=m_all,
                                 n_eigs=n_eigs,
                                 sect_props=sect_props)
# Return the important example results
# The signature curve is simply a matter of plotting the
# 'signature' values against the lengths
# (usually on a logarithmic axis)
# return {
#     'X_values': lengths,
#     'Y_values': signature,
#     'Y_values_allmodes': curve,
#     'Orig_coords': nodes,
#     'Deformations': shapes,
#     'Elements': elements 
# }
lengths = np.array(lengths)
signature = np.array(signature)
curves = np.array(curve)
node=np.array(nodes)
shapes = np.array(shapes)
element = np.array(elements)
######POST PROCESSING#########
#Choose mode number
modeindex = 1
length_index = 20
fileddisplay = [1]
######DISPLACED SHAPE####
undef = 1
scalem = 1
m_a = [1]
SurfPos = 1/2
mode = shapes[length_index-1, :, modeindex-1]
print(mode)
####CALL DISPLACED SHAPE
x = dispshap.dispshap(undef, node, element, mode, scalem, springs, m_a, b_c ,SurfPos)
####SIGNATURE CURVE####
###Buckling halfwavelenth plot from a signature analysis
plt.semilogx(lengths, signature)
plt.show()
clas = 0
minopt = 1
logopt = 1
clasopt = 0
xmin = np.min(lengths)*10/11
xmax = np.max(lengths)*11/10
ymin = 0
curve_sign = np.zeros((len(lengths),2))
curve = np.zeros(((len(lengths),n_eigs,2)))
for j in range(len(lengths)):
    for i in range(n_eigs):
        curve[j, i, 0] = lengths[j]
        curve[j, i, 1] = curves[j, i]
for j in range(len(lengths)):
    curve_sign[j, 0] = lengths[j]
    curve_sign[j, 1] = curves[j, modeindex-1]
ymax = np.min([np.max(curve_sign[:, 1]), 3*np.median(curve_sign[:, 1])])
modedisplay = [1]
fileindex = 1
picpoint = [lengths[length_index-1], curves[length_index-1, modeindex-1]]
#CALL THE CURVE
thecurve3.thecurve3(curve, clas, fileddisplay, minopt, logopt, clasopt, xmin, xmax,
ymin, ymax, modedisplay, fileindex, modeindex, picpoint)
plt.show()