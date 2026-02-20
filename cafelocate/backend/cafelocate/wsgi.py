"""
WSGI config for cafelocate project.
Used by production servers (Gunicorn, uWSGI).
Development uses: python manage.py runserver (not this file).
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cafelocate.settings')
application = get_wsgi_application()