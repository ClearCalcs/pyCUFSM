# pyCUFSM

## Description

This package is primarily a port of CUFSM v5.01, written by Benjamin Schafer PhD et al at Johns Hopkins University, from its original MATLAB language to Python v3, using the Numpy and Scipy packages for matrix manipulation and other advanced mathematics functionality. The goal of this project is to create a derivative of CUFSM which can be used either in Jupyter Notebooks or in headless (library) applications. This project is not affiliated with Benjamin Schafer PhD or Johns Hopkins University in any way.

The original MATLAB CUFSM program may be accessed at the following address: https://www.ce.jhu.edu/cufsm/

### Installation

This package is still under heavy development, but it may be installed in several different possible forms, as described below:
1. Minimal (headless) installation: `pip install pycufsm`
2. Installation with plotting capabilities: `pip install pycufsm[plot]`
3. Installation with Jupyter Notebooks: `pip install pycufsm[jupyter]`
4. Installation with full development dependencies: `pip install pycufsm[dev]`

### Contributing

If you would like to contribute to the pyCUFSM project, then please do - all productive contributions are welcome! However, please make sure that you're working off of the most recent development version of the pyCUFSM code, by cloning the [GitHub repository](https://github.com/ClearCalcs/pyCUFSM), and please review our wiki article on [Contributing to the Code](https://github.com/ClearCalcs/pyCUFSM/wiki/Contributing-to-the-Code).

## Current Status

#### Complete and Generally Tested

-   [x] Unconstrained FSM (signature curve generation)
-   [x] Constrained FSM
-   [x] Added template_path() function to define a complete arbitrary cross-section by simple paths
-   [x] Add automated validation testing of FSM calculations via pytest
-   [x] Various efficiency and readability improvements:
    -   [x] Cythonise a few computation-heavy functions in analysis.py, including klocal(), kglocal(), and assemble()
    -   [x] Moved computation-heavy cFSM functions to analysis.py and cythonised them
    -   [x] Review code for places where matrices can be preallocated rather than concatenated together

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
    -   [ ] Review code for function calls that are unnecessarily repeated (a couple of these have already been addressed, e.g. `base_properties()` did not need to be re-run for every half wavelength)
-   [ ] Write API-style documentation (for now, generally refer to MATLAB CUFSM documentation and/or comments)

## Disclaimer

While the original MATLAB CUFSM has been extensively tested, and best efforts have been made to check accuracy of this package against the original MATLAB CUFSM program, including via automated validation testing, no warrant is made as to the accuracy of this package. The developers accept no liability for any errors or inaccuracies in this package, including, but not limited to, any problems which may stem from such errors or inaccuracies in this package such as under-conservative engineering designs or structural failures.

Always check your designs and never blindly trust any engineering program, including this one.
