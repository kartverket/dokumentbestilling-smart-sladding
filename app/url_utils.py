import os


def api_base_url():
    if os.getenv('API_URL') == None:
        return 'http://localhost:8080/'
    else:
        return os.getenv('API_URL')
def database_base_url():
    if os.getenv('DATABASE_URL') == None:
        return 'http://localhost:8000/'
    else:
        return os.getenv('DATABASE_URL')
