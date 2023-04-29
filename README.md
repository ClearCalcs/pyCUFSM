# pyCUFSM

## Description

This is a port of the analysis portions of CUFSM v5.01, written by Benjamin Schafer PhD et al at Johns Hopkins University, from its original MATLAB language to Python v3, using the Numpy and Scipy packages for matrix manipulation and other advanced mathematics functionality. The goal of this project is to create a derivative of CUFSM which can be used in cloud-based applications. This project is not affiliated with Benjamin Schafer PhD or Johns Hopkins University in any way.

The original MATLAB CUFSM program may be accessed at the following address: https://www.ce.jhu.edu/bschafer/cufsm/

### Installation

This package is still under check and development in its alpha state, therefore it has not yet been published to the Python's PyPI repository. Users may install clone and install the package on their systems using relevant codes the same as printed bellow:

`python -m pip install git+https://github.com/ClearCalcs/pyCUFSM.git`

### Limitations

-   **No GUI** - While the MATLAB version of CUFSM contains a full graphical user interface, I will be making no effort in this project to create anything more than a basic command-line interface (though other contributions will be welcome). I anticipate that users of this package will have their own user interface and that this package will function as something of a plug-in.
-   **No CUTWP** - The MATLAB version of CUFSM makes use of CUTWP code for calculating section properties (A, I, J, etc). There are already several mature, open-source Python section properties calculators available, and CUTWP is inherently less accurate because it is based upon section centerline calculations only. As such, I have no plans to port the CUTWP code and anticipate that users of this package will make use of other section properties calculators instead.

## Current Status

#### Complete and Generally Tested

-   [x] Unconstrained FSM (signature curve generation)
-   [x] Constrained FSM
-   [x] Added template_path() function to define a complete arbitrary cross-section by simple paths
-   [x] Add automated validation testing of FSM calculations via pytest

#### Complete But Untested

-   [x] Equation constraints
-   [x] Spring constraints
-   [x] General boundary conditions

#### Planned Further Work

-   [ ] Handle holes in cross-sections in some meaningful way
-   [ ] Various efficiency and readability improvements:
    -   [ ] Make use of scipy.sparse for sparse matrices where possible
    -   [ ] Convert some numerical inputs and data to dictionaries with strings
    -   [ ] Eliminate matrix columns which are nothing more than the index number of the row
    -   [ ] Review code for places where matrices can be preallocated rather than concatenated together
    -   [ ] Review code for function calls that are unnecessarily repeated (a couple of these have already been addressed, e.g. `base_properties()` did not need to be re-run for every half wavelength)
-   [ ] Write API-style documentation (for now, generally refer to MATLAB CUFSM documentation and/or comments)

## Disclaimer

While the original MATLAB CUFSM has been extensively tested, and best efforts have been made to check accuracy of this package against the original MATLAB CUFSM program, no warrant is made as to the accuracy of this package. The developers accept no liability for any errors or inaccuracies in this package, including, but not limited to, any problems which may stem from such errors or inaccuracies in this package such as under-conservative engineering designs or structural failures.

Always check your designs and never blindly trust any engineering program, including this one.
