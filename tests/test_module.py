import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import model_utils
import model_main
import evaluation_utils


@pytest.fixture
def personnumber():
    return "30053238021"

# add different spaces in personnumber for tests?

# @pytest.fixture
# def personnumber_spaces():
#     return "30 05 32 3 8 0 2 1"

# Fixed parameters

@pytest.fixture
def dnumber():
    return "70053238021"

@pytest.fixture
def doc_id():
    return '2023_72893_200'

# Test functions

def test_check_controldigits(personnumber):
    assert model_utils.check_controldigits(personnumber) == True

def test_format_dnumber(dnumber):
    assert model_utils.format_dnumber(dnumber) == (True, "30053238021")

def test_main(doc_id):
    assert model_main.main(doc_id) == (True, "2023_72893_200")

