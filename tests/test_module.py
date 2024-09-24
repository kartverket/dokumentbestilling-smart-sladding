import pytest
import sys
import os

# Determine the absolute path to the app directory
app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app'))

# Add the app directory to the Python path
sys.path.insert(0, app_path)

import model_utils
import model_main

# Fixtures
@pytest.fixture
def personnumber():
    return "30053238021"

@pytest.fixture
def not_personnumber():
    return "30053238022"

@pytest.fixture
def not_dnumber():
    return "70053238021"

@pytest.fixture
def dnumber():
    return "55079012350"

@pytest.fixture
def doc_id():
    return '2023_72893_200'

@pytest.fixture
def json_response():
    return [
        {'page': 1, 'height': 279.72, 'width': 405.36, 'x': 6.48, 'y': 23.4},
        {'page': 1, 'height': 574.92, 'width': 208.44, 'x': 6.48, 'y': 23.4},
        {'page': 1, 'height': 702.72, 'width': 77.4, 'x': 6.48, 'y': 23.04},
        {'page': 1, 'height': 295.2, 'width': 405.36, 'x': 6.48, 'y': 23.04},
        {'page': 1, 'height': 590.76, 'width': 208.44, 'x': 6.48, 'y': 23.4},
        {'page': 1, 'height': 686.88, 'width': 77.04, 'x': 6.48, 'y': 23.4}
    ]

# Test functions
def test_find_matches(personnumber, not_personnumber, dnumber):
    assert model_utils.find_matches(f"This is a test with a valid personnumber {personnumber}") == [[f'{personnumber}', 'personnummer', 0]]
    assert model_utils.find_matches(f'This is a test with an invalid personnumber {not_personnumber}') == []
    assert model_utils.find_matches(f"This is a test with an invalid dnumber {not_dnumber}") == []
    assert model_utils.find_matches(f"This is a test with a valid dnumber {dnumber}") == [[f'{dnumber}', 'dnummer', 0]]
    assert model_utils.find_matches(f"This is a test with a valid personnumber {personnumber} and a valid dnumber {dnumber}") == [
        [f'{personnumber}', 'personnummer', 0],
        [f'{dnumber}', 'dnummer', 1]
    ]

def test_check_control_digits(personnumber, not_personnumber, dnumber):
    assert model_utils.check_control_digits(personnumber) is True
    assert model_utils.check_control_digits(not_personnumber) is False
    formatted_dnumber = model_utils.find_matches(dnumber)[0][0]
    assert model_utils.check_control_digits(formatted_dnumber) is True

def test_format_dnumber(dnumber):
    assert model_utils.find_matches(dnumber) == [[dnumber, 'dnummer', 0]]

def test_main(doc_id, json_response):
    assert model_main.main(doc_id) == json_response
