# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

import pytest

from denmune_skl.utils.discovery import all_displays, all_estimators, all_functions


# We only have 1 estimator defined
def test_all_estimators():
    estimators = all_estimators()
    assert len(estimators) == 1

    # DenMune is a clusterer, not a classifier
    estimators = all_estimators(type_filter="classifier")
    assert len(estimators) == 0

    # Test for clusterer
    estimators = all_estimators(type_filter="cluster")
    assert len(estimators) == 1

    # Test for multiple types
    estimators = all_estimators(type_filter=["classifier", "cluster"])
    assert len(estimators) == 1

    err_msg = "Parameter type_filter must be"
    with pytest.raises(ValueError, match=err_msg):
        all_estimators(type_filter="xxxx")


def test_all_displays():
    displays = all_displays()
    assert len(displays) == 0


def test_all_functions():
    functions = all_functions()
    assert len(functions) == 3
