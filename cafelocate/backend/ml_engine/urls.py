from django.urls import path
from . import views

urlpatterns = [
    # POST /api/predict/
    # Direct access to ML prediction without full analysis
    path('predict/', views.PredictView.as_view(), name='predict'),
]