import pytest
import sys
import os

import url_utils
import model_utils
import model_main

# Determine the absolute path to the app directory
app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app'))

# Add the app directory to the Python path
sys.path.insert(0, app_path)

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
        {'page': 1, 'height': 10.08, 'width': 27.0, 'x': 403.55999999999995, 'y': 277.91999999999996},
        {'page': 1, 'height': 10.08, 'width': 27.0, 'x': 206.64, 'y': 573.12},
        {'page': 1, 'height': 10.08, 'width': 26.64, 'x': 75.6, 'y': 700.9200000000001},
        {'page': 1, 'height': 10.08, 'width': 26.64, 'x': 403.55999999999995, 'y': 293.4},
        {'page': 1, 'height': 10.08, 'width': 27.0, 'x': 206.64, 'y': 588.96},
        {'page': 1, 'height': 10.08, 'width': 27.0, 'x': 75.24, 'y': 685.08}
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
    assert model_main.main(doc_id, f'{url_utils.api_base_url()}intern/pantebok/gjenpart') == json_response
