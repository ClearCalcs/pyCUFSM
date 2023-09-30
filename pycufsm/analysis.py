import numpy as np

try:
    try:
        # if the cython module is already built, use it
        import pycufsm.analysis_c as analysis  # pylint:disable=unused-import

    except ImportError:
        # if we can build the cython module, build and use it
        import pyximport  # type: ignore

        pyximport.install(
            build_in_temp=False,
            inplace=True,
            reload_support=True,
            setup_args={"include_dirs": np.get_include()},
        )
        import pycufsm.analysis_c as analysis  # type: ignore  # pylint:disable=ungrouped-imports,unused-import

except ImportError:
    # if cython just fails entirely, then use the pure python module
    import pycufsm.analysis_p as analysis  # pylint:disable=unused-import
