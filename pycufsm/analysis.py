try:
    try:
        # if the cython module is already built, use it
        import pycufsm.analysis_c as analysis

    except ImportError:
        # if we can build the cython module, build and use it
        import pyximport
        pyximport.install(build_in_temp=False, inplace=True, reload_support=True)
        import pycufsm.analysis_c as analysis

except ImportError:
    # if cython just fails entirely, then use the pure python module
    import pycufsm.analysis_p as analysis
