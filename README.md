# pyCUFSM

## Description
This is a port of the analysis portions of CUFSM v5.01, written by Benjamin Schafer PhD et al at Johns Hopkins University, from its original MATLAB language to Python v3, using the Numpy and Scipy packages for matrix manipulation and other advanced mathematics functionality. The goal of this project is to create a derivative of CUFSM which can be used in cloud-based applications. This project is not affiliated with Benjamin Schafer PhD or Johns Hopkins University in any way. 

### Limitations
- **No GUI** - While the MATLAB version of CUFSM contains a full graphical user interface, I will be making no effort in this project to create anything more than a basic command-line interface (though other contributions will be welcome). I anticipate that users of this package will have their own user interface and that this package will function as something of a plug-in.
- **No CUTWP** - The MATLAB version of CUFSM makes use of CUTWP code for calculating section properties (A, I, J, etc). There are already several mature, open-source Python section properties calculators available, and CUTWP is inherently less accurate because it is based upon section centerline calculations only. As such, I have no plans to port the CUTWP code and anticipate that users of this package will make use of other section properties calculators instead.

## Current Status

#### Complete and Generally Tested
  - [x] Unconstrained FSM (signature curve generation)
  
#### Complete But Untested
  - [x] Equation constraints
  - [x] Spring constraints
  - [x] General boundary conditions
  - [x] Constrained FSM
  
#### Planned Further Work
  - [ ] Improve efficiency by making use of scipy.sparse for sparse matrices where possible
  - [ ] Handle holes in cross-sections in some meaningful way
  - [ ] Expand templatecalc to handle more shapes other than just Cees and Zeds

## Disclaimer
While the original MATLAB CUFSM has been extensively tested, and best efforts have been made to check accuracy of this package against the original MATLAB CUFSM program, no warrant is made as to the accuracy of this package. I accept no liability for any errors or inaccuracies in this package, including, but not limited to, any problems which may stem from such errors or inaccuracies in this package such as under-conservative engineering designs or structural failures. 

Always check your designs and never blindly trust any engineering program, including this one.
