import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
def dnumber():
    return "70053238021"

@pytest.fixture
def doc_id():
    return '2023_72893_200'

@pytest.fixture
def json_response():
    return [{'page': 1, 'height': 279.71999999999997, 'width': 405.35999999999996, 'x': 6.4799999999999995, 'y': 23.4}, 
            {'page': 1, 'height': 574.92, 'width': 208.44, 'x': 6.4799999999999995, 'y': 23.4}, 
            {'page': 1, 'height': 702.72, 'width': 77.39999999999999, 'x': 6.4799999999999995, 'y': 23.04}, 
            {'page': 1, 'height': 295.2, 'width': 405.35999999999996, 'x': 6.4799999999999995, 'y': 23.04}, 
            {'page': 1, 'height': 590.76, 'width': 208.44, 'x': 6.4799999999999995, 'y': 23.4}, 
            {'page': 1, 'height': 686.88, 'width': 77.03999999999999, 'x': 6.4799999999999995, 'y': 23.4}]

# Test functions
def test_find_matches(personnumber, not_personnumber, dnumber):
    assert model_utils.find_matches(f"This is a test with a valid personnumber {personnumber}") == [[f'{personnumber}', 'personnummer', 0]]
    assert model_utils.find_matches(f'This is a test with an invalid personnumber {not_personnumber}') == []
    assert model_utils.find_matches(f"This is a test with a valid dnumber {dnumber}") == [[f'{dnumber}', 'dnummer', 0]]
    assert model_utils.find_matches(f"This is a test with a valid personnumber {personnumber} and a valid dnumber {dnumber}") == [[f'{personnumber}', 'personnummer', 0], [f'{dnumber}', 'dnummer', 1]]

def test_check_controldigits(personnumber, not_personnumber, dnumber):
    assert model_utils.check_controldigits(personnumber) == True
    assert model_utils.check_controldigits(not_personnumber) == False
    assert model_utils.check_controldigits(model_utils.format_dnumber(dnumber)[1]) == True

def test_format_dnumber(dnumber):
    assert model_utils.format_dnumber(dnumber) == (True, "30053238021")

def test_main(doc_id, json_response):
    assert model_main.main(doc_id) == json_response
