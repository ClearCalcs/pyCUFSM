import numpy as np
from pytest import approx, raises
from pycufsm import preprocess, helpers, fsm
import pycufsm.examples.example_1 as example_1
from .utils import pspec_context
from .fixtures.e2e_fixtures import *


def describe_end_to_end_tests():
    @pspec_context("End-to-End Tests (i.e. original unittest tests)")
    def describe():
        pass

    def context_example_1():
        @pspec_context("Example 1")
        def describe():
            pass

        results = example_1.__main__()

        def it_results_in_correct_solution():
            assert results["X_values"] == [
                0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75,
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36,
                38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120,
                132, 144, 156, 168, 180, 204, 228, 252, 276, 300
            ]
