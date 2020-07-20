import ipywidgets as widgets
import numpy as np
props = np.array([[0, 29500, 29500, 0.3, 0.3, 29500 / (2 * (1 + 0.3))]])
nodes =np.array([[0, 2.5, 0.773, 1, 1, 1, 1, 0], [1, 2.5, 0, 1, 1, 1, 1, 0],
             [2, 1.25, 0, 1, 1, 1, 1, 0], [3, 0, 0, 1, 1, 1, 1, 0],
             [4, 0, 3, 1, 1, 1, 1, 0], [5, 0, 6, 1, 1, 1, 1, 0],
             [6, 0, 9, 1, 1, 1, 1, 0], [7, 1.25, 9, 1, 1, 1, 1, 0],
             [8, 2.5, 9, 1, 1, 1, 1, 0], [9, 2.5, 8.227, 1, 1, 1, 1, 0]])
elements = np.array([[0, 0, 1, 0.059, 0], [1, 1, 2, 0.059, 0], [2, 2, 3, 0.059, 0],
                [3, 3, 4, 0.059, 0], [4, 4, 5, 0.059, 0], [5, 5, 6, 0.059, 0],
                [6, 6, 7, 0.059, 0], [7, 7, 8, 0.059, 0], [8, 8, 9, 0.059, 0]])
mattext = ['mat#', 'Ex', 'Ey', 'vx', 'vy', 'G']
nodetext = ['Node#','x','y','xdof', 'zdof', 'ydof','qdof', 'stress']
elemtext = ['Element#', 'Nodei', 'Nodej', 't', 'Mat#']
ADDNODE = widgets.Button(description="Add Node")
ADDELEM = widgets.Button(description="Add Element")
ADDMAT = widgets.Button(description="Add Material")
prop = [[] for i in range(1)]
node = [[] for i in range(10)]
elem = [[] for i in range(9)]
mitems = [[] for i in range(1)]
nitems = [[] for i in range(10)]
eitems = [[] for i in range(9)]
matlabel = widgets.GridBox([widgets.Label(value=mattext[j]) for j in range(6)], layout=widgets.Layout(grid_template_columns="repeat(6, 50px[col-start])"))
for i in range(1):
    mitems[i]= [widgets.FloatText(value=props[i,j]) for j in range(6)]
    prop[i] = widgets.GridBox(mitems[i], layout=widgets.Layout(grid_template_columns="repeat(6, 50px[col-start])"))
matr0 = widgets.VBox([prop[j] for j in range(1)])
matr = widgets.VBox([matlabel, matr0])
nlabel= widgets.GridBox([widgets.Label(value=nodetext[j]) for j in range(8)], layout=widgets.Layout(grid_template_columns="repeat(8, 50px[col-start])"))
for i in range(10):
    nitems[i]= [widgets.FloatText(value=nodes[i, j]) for j in range(8)]
    node[i] = widgets.GridBox(nitems[i], layout=widgets.Layout(grid_template_columns="repeat(8, 50px[col-start])"), width = '70%')
noder0 = widgets.VBox([node[j] for j in range(10)])
noder = widgets.VBox([nlabel, noder0])
elabel = widgets.GridBox([widgets.Label(value=elemtext[j]) for j in range(5)], layout=widgets.Layout(grid_template_columns="repeat(5, 50px[col-start])"))
for i in range(9):
    eitems[i]= [widgets.FloatText(value=elements[i, j]) for j in range(5)]
    elem[i] = widgets.GridBox(eitems[i], layout=widgets.Layout(grid_template_columns="repeat(5, 50px[col-start])"), width = '30%')
elemr0 = widgets.VBox([elem[j] for j in range(9)])
elemr = widgets.VBox([elabel, elemr0])
# node = widgets.GridBox(noder, layout=widgets.Layout(grid_template_columns="repeat(6, 50px[col-start])"))
row0 = widgets.VBox([matr, ADDMAT])
row1 = widgets.VBox([noder, ADDNODE])
row2 = widgets.VBox([elemr, ADDELEM])
row = widgets.HBox([row1, row2])
page = widgets.VBox([row0, row])
return page, mitems, nitems, eitems