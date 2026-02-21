from django.urls import path
from . import views

urlpatterns = [
    # POST /api/auth/register/
    # Create new user account with username, email, password
    path('auth/register/', views.UserRegistrationView.as_view(), name='user-register'),

    # POST /api/auth/login/
    # Authenticate user with username/email and password
    path('auth/login/',    views.UserLoginView.as_view(),    name='user-login'),

    # GET /api/cafes/nearby/?lat=27.71&lng=85.32&radius=500
    # Returns list of all cafes within radius meters of point
    path('cafes/nearby/', views.NearbyCafesView.as_view(), name='cafes-nearby'),

    # POST /api/analyze/
    # Main analysis: nearby + top5 + suitability score + ML prediction
    path('analyze/',      views.SuitabilityAnalysisView.as_view(), name='analyze'),
]

# Final URL structure:
#   /api/auth/google/    ← from cafelocate/urls.py prefix "api/" + "auth/google/"
#   /api/cafes/nearby/   ← from cafelocate/urls.py prefix "api/" + "cafes/nearby/"
#   /api/analyze/        ← from cafelocate/urls.py prefix "api/" + "analyze/"
#   /api/predict/        ← handled by ml_engine/urls.py