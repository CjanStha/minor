"""
CafeLocate ML — Django Settings
Full configuration for the project.
"""
import os
from pathlib import Path
import environ

# ─── BASE DIRECTORY ────────────────────────────────────────────────────────────
# BASE_DIR points to the backend/ folder (where manage.py lives)
# Path(__file__) = this settings.py file
# .resolve() = get the absolute path
# .parent.parent = go up 2 levels (cafelocate/ → backend/)
BASE_DIR = Path(__file__).resolve().parent.parent

# ─── LOAD .env FILE ────────────────────────────────────────────────────────────
# env() reads values from .env file so we never hardcode secrets in this file
env = environ.Env(
    # Define types and defaults for each variable
    DEBUG=(bool, False),
    ALLOWED_HOSTS=(list, [])
)
environ.Env.read_env(BASE_DIR / '.env')   # reads backend/.env


# ─── SECURITY ──────────────────────────────────────────────────────────────────
# SECRET_KEY: used to sign cookies and CSRF tokens. NEVER share this.
SECRET_KEY = env('SECRET_KEY')

# DEBUG: True = show detailed error pages (development only!)
# In production, set DEBUG=False in .env
DEBUG = env('DEBUG')

# ALLOWED_HOSTS: which domain names can access this server
# During development: localhost and 127.0.0.1
ALLOWED_HOSTS = env.list('ALLOWED_HOSTS', default=['localhost', '127.0.0.1'])


# ─── INSTALLED APPS ────────────────────────────────────────────────────────────
# Django only uses apps listed here. Order matters for some cases.
# Add YOUR apps at the bottom.
INSTALLED_APPS = [
    # ── Django built-ins (leave these alone) ──
    'django.contrib.admin',        # admin panel at /admin/
    'django.contrib.auth',         # user authentication system
    'django.contrib.contenttypes',  # required by auth
    'django.contrib.sessions',     # session storage
    'django.contrib.messages',     # flash messages
    'django.contrib.staticfiles',  # serving CSS/JS files
    'django.contrib.gis',          # ← GIS support (PostGIS queries)

    # ── Third-party packages ──
    'rest_framework',              # Django REST Framework — turns Django into an API
    'corsheaders',                 # CORS — allows frontend to call this API

    # ── Your apps ──
    'api',                         # REST API endpoints (nearby, top5, suitability)
    'ml_engine',                   # ML prediction endpoint
]


# ─── MIDDLEWARE ────────────────────────────────────────────────────────────────
# Middleware runs on EVERY request before it reaches your view.
# CorsMiddleware MUST be first so it handles CORS headers before anything else.
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',           # ← MUST be first
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


# ─── URL CONFIGURATION ─────────────────────────────────────────────────────────
# Tells Django where the main urls.py file is
ROOT_URLCONF = 'cafelocate.urls'   # = cafelocate/urls.py


# ─── TEMPLATES ─────────────────────────────────────────────────────────────────
# Template engine config (for HTML responses — used by admin panel)
TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': [
        'django.template.context_processors.debug',
        'django.template.context_processors.request',
        'django.contrib.auth.context_processors.auth',
        'django.contrib.messages.context_processors.messages',
    ]},
}]

WSGI_APPLICATION = 'cafelocate.wsgi.application'


# ─── DATABASE ──────────────────────────────────────────────────────────────────
# Using PostGIS backend so Django understands spatial/geometry fields.
# All values come from .env file — never hardcode passwords here.
DATABASES = {
    'default': {
        'ENGINE':   'django.contrib.gis.db.backends.postgis',
        #            ↑ GIS-aware PostgreSQL backend (not regular 'postgresql')
        'NAME':     env('DB_NAME'),       # cafelocate_db
        'USER':     env('DB_USER'),       # cafelocate_user
        'PASSWORD': env('DB_PASSWORD'),   # from .env
        'HOST':     env('DB_HOST', default='localhost'),
        'PORT':     env('DB_PORT', default='5432'),
    }
}


# ─── DJANGO REST FRAMEWORK ─────────────────────────────────────────────────────
# Controls how the API behaves globally
REST_FRAMEWORK = {
    # Default: return JSON (not HTML)
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    # How incoming data is parsed
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
    ],
}


# ─── CORS SETTINGS ─────────────────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# Without this, the browser blocks JavaScript from calling a different port.
# Your frontend runs on port 5500, backend on 8000 → different "origins" → CORS needed.
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5500",     # VS Code Live Server
    "http://127.0.0.1:5500",
    "http://localhost:3000",     # if you ever switch to React
]

# Allow these HTTP methods from the frontend
CORS_ALLOW_METHODS = ['GET', 'POST', 'OPTIONS']

# Allow the Authorization header (for JWT tokens)
CORS_ALLOW_HEADERS = [
    'accept', 'authorization', 'content-type', 'x-csrftoken',
]


# ─── GOOGLE API KEYS ───────────────────────────────────────────────────────────
# Stored here so views.py can access them with: settings.GOOGLE_PLACES_API_KEY
GOOGLE_PLACES_API_KEY = env('GOOGLE_PLACES_API_KEY')
GOOGLE_CLIENT_ID      = env('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET  = env('GOOGLE_CLIENT_SECRET')


# ─── PASSWORD VALIDATORS ───────────────────────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


# ─── LOCALIZATION ──────────────────────────────────────────────────────────────
LANGUAGE_CODE = 'en-us'
TIME_ZONE     = 'Asia/Kathmandu'   # Nepal Time (UTC+5:45)
USE_I18N      = True
USE_TZ        = True


# ─── STATIC FILES ──────────────────────────────────────────────────────────────
# Where Django looks for CSS/JS files for the admin panel
STATIC_URL  = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'


# ─── DEFAULT PRIMARY KEY ───────────────────────────────────────────────────────
# Auto-increment integer IDs for all models (modern Django default)
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'