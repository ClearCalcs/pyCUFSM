import pytest
import numpy as np
from pycufsm import preprocess, helpers, fsm

# import pycufsm.examples.example_1 as ex1_main


@pytest.fixture
def nodes():
    return np.array([[0, 0, 0, 1, 1, 1, 1, 0], [1, 5, 0, 1, 1, 1, 1, 0]])  # trivial 2-node cross-section


@pytest.fixture
def thickness():
    return 0.1


@pytest.fixture
def elements(thickness):
    return np.array([[0, 0, 1, thickness, 0]])  # trivial 1-element cross-section


@pytest.fixture
def props():
    # Standard steel stiffness properties in Imperial units
    e_stiff = 29500  # ksi
    nu_poisson = 0.3
    g_stiff = 29500 / (2 * (1 + 0.3))  # ksi
    return np.array([np.array([0, e_stiff, e_stiff, nu_poisson, nu_poisson, g_stiff])])


@pytest.fixture
def sect_props():
    return {
        "cx": 2.5,
        "cy": 0,
        "x0": 2.5,
        "y0": 0,
        "phi": 0,
        "A": 0.5,
        "Ixx": 0.0004167,
        "Ixy": 0,
        "Iyy": 1.04167,
        "I11": 0.0004167,
        "I22": 1.04167,
    }


@pytest.fixture
def lengths():
    # These lengths will generally provide sufficient accuracy for
    # local, distortional, and global buckling modes in units of inches
    return [
        0.5,
        0.75,
        1,
        1.25,
        1.5,
        1.75,
        2,
        2.25,
        2.5,
        2.75,
        3,
        3.25,
        3.5,
        3.75,
        4,
        4.25,
        4.5,
        4.75,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
        42,
        44,
        46,
        48,
        50,
        52,
        54,
        56,
        58,
        60,
        66,
        72,
        78,
        84,
        90,
        96,
        102,
        108,
        114,
        120,
        132,
        144,
        156,
        168,
        180,
        204,
        228,
        252,
        276,
        300,
    ]


@pytest.fixture
def springs():
    # No special springs
    return []


@pytest.fixture
def constraints():
    # No special constraints
    return []


@pytest.fixture
def gbt_con():
    # Signature curve basis, orthogonal based on geometry
    return {
        "glob": [0],
        "dist": [0],
        "local": [0],
        "other": [0],
        "o_space": 1,
        "couple": 1,
        "orth": 2,
        "norm": 0,
    }


@pytest.fixture
def b_c():
    # Simply supported boundary conditions
    return "S-S"


@pytest.fixture
def m_all(lengths):
    # For signature curve analysis, only a single array of ones makes sense here
    return np.ones((len(lengths), 1))


@pytest.fixture
def n_eigs():
    # Solve for 10 eigenvalues
    return 10


@pytest.fixture
def forces(sect_props):
    # Generate the stress points assuming 50 ksi yield and pure compression
    return {"P": sect_props["A"] * 50, "Mxx": 0, "Myy": 0, "M11": 0, "M22": 0}


@pytest.fixture
def offset_basis(thickness):
    return [-thickness / 2, -thickness / 2]


@pytest.fixture
def stress_gen(nodes, forces, sect_props, offset_basis):
    return preprocess.stress_gen(nodes=nodes, forces=forces, sect_props=sect_props, offset_basis=offset_basis)


@pytest.fixture
def stressed_nodes():
    return np.array(
        [[0, 0, 0, 1, 1, 1, 1, 50], [1, 5, 0, 1, 1, 1, 1, 50]]
    )  # trivial 2-node cross-section with 50ksi stress


@pytest.fixture
def strip(props, stressed_nodes, elements, lengths, springs, constraints, gbt_con, b_c, m_all, n_eigs, sect_props):
    return fsm.strip(
        props=props,
        nodes=stressed_nodes,
        elements=elements,
        lengths=lengths,
        springs=springs,
        constraints=constraints,
        gbt_con=gbt_con,
        b_c=b_c,
        m_all=m_all,
        n_eigs=n_eigs,
        sect_props=sect_props,
    )
