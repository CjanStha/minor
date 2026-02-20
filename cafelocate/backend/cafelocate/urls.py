from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # Django admin panel — visit http://localhost:8000/admin/
    path('admin/', admin.site.urls),

    # All API routes — delegates to api/urls.py
    # Any URL starting with "api/" is handled by the api app
    # Example: /api/cafes/nearby/ → goes to api/urls.py for further routing
    path('api/', include('api.urls')),

    # ML prediction route — delegates to ml_engine/urls.py
    # /api/predict/ → handled by ml_engine
    path('api/', include('ml_engine.urls')),
]