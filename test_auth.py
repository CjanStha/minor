import requests
import json

# Test user registration
url = 'http://127.0.0.1:8000/api/auth/register/'
data = {
    'username': 'testuser456',
    'email': 'test456@example.com',
    'password': 'testpass123'
}
headers = {'Content-Type': 'application/json'}

try:
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print(f'Registration status: {response.status_code}')
    print(f'Response: {response.text}')
except Exception as e:
    print(f'Error: {e}')

# Test user login
url = 'http://127.0.0.1:8000/api/auth/login/'
data = {
    'username': 'testuser456',
    'password': 'testpass123'
}

try:
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print(f'Login status: {response.status_code}')
    print(f'Response: {response.text}')
except Exception as e:
    print(f'Error: {e}')