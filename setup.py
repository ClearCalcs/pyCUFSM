import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pycufsm",
    version="0.0.1",
    author="Brooks H. Smith, MEng, PE, CPEng",
    author_email="brooks.smith@clearcalcs.com",
    description="Python CUFSM (Constrained and Unconstrained Finite Strip Method)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ClearCalcs/pyCUFSM",
    packages=setuptools.find_packages(),
    license="AFL-3.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Academic Free License (AFL)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    keywords="cufsm cfs analysis engineering cold-formed steel aluminium aluminum thin-walled",
    install_requires=[
        "numpy>=1.17",
        "scipy>=1.4",
    ],
    python_requires=">=3.6",
)
