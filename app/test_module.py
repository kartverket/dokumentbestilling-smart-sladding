import pytest
import sys
import os

import model_utils

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
def json_response():
    return [
        {'page': 1, 'height': 11.19903206291591, 'width': 27.486842105263158, 'x': 112.01467029643072, 'y': 605.5983061101028},
        {'page': 2, 'height': 11.19903206291591, 'width': 29.556563823351482, 'x': 217.57047791893524, 'y': 57.392014519056254}
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

#def test_main(json_response): TODO: Enable this test and make it work in workflow
#    assert model_main.main('testdokument-2.pdf') == json_response
