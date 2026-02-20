from django.conf import settings
from django.contrib.gis.geos import Point
from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.measure import D  # Distance unit helper

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import jwt, math, logging

from .models import Cafe, Ward, Road, UserProfile
from .serializers import CafeSerializer, SuitabilityRequestSerializer, UserProfileSerializer
from ml_engine.predictor import get_prediction

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# VIEW 1: Google OAuth Login
# POST /api/auth/google/
# Frontend sends Google's ID token → we validate it → return our JWT
# ═══════════════════════════════════════════════════════════════════
class GoogleLoginView(APIView):
    authentication_classes = []  # no auth required to log in
    permission_classes = []     # open to anyone

    def post(self, request):
        token = request.data.get('token')
        if not token:
            return Response({'error': 'Token required'}, status=400)

        try:
            # Verify with Google — confirms the token is legitimate
            idinfo = id_token.verify_oauth2_token(
                token,
                google_requests.Request(),
                settings.GOOGLE_CLIENT_ID
            )

            # Get or create user in our database
            user, created = UserProfile.objects.get_or_create(
                google_id=idinfo['sub'],
                defaults={
                    'email':       idinfo['email'],
                    'name':        idinfo.get('name', ''),
                    'picture_url': idinfo.get('picture', ''),
                }
            )

            # Create our own JWT so frontend doesn't need Google's token again
            our_token = jwt.encode(
                {'user_id': user.id, 'email': user.email},
                settings.SECRET_KEY,
                algorithm='HS256'
            )
            return Response({
                'token': our_token,
                'user':  UserProfileSerializer(user).data
            })

        except ValueError as e:
            return Response({'error': 'Invalid Google token'}, status=401)


# ═══════════════════════════════════════════════════════════════════
# VIEW 2: Nearby Cafés
# GET /api/cafes/nearby/?lat=27.71&lng=85.32&radius=500
# Returns all cafés within 'radius' meters of the given point
# ═══════════════════════════════════════════════════════════════════
class NearbyCafesView(APIView):

    def get(self, request):
        try:
            lat    = float(request.GET.get('lat'))
            lng    = float(request.GET.get('lng'))
            radius = float(request.GET.get('radius', 500))  # default 500m
        except (TypeError, ValueError):
            return Response({'error': 'lat and lng are required numbers'}, status=400)

        # Create a PostGIS Point object from the user's click
        # (lng, lat) order — PostGIS uses (x=longitude, y=latitude)
        user_point = Point(lng, lat, srid=4326)

        # THE CORE SPATIAL QUERY
        # D(m=radius) creates a Distance object in meters
        # filter(location__distance_lte=...) → PostGIS ST_DWithin under the hood
        # annotate(distance=...) adds a 'distance' field to each result
        cafes = (
            Cafe.objects
            .filter(
                location__distance_lte=(user_point, D(m=radius)),
                is_open=True
            )
            .annotate(distance=Distance('location', user_point))
            .order_by('distance')   # nearest first
        )

        serializer = CafeSerializer(cafes, many=True)
        return Response({
            'count':  cafes.count(),
            'cafes':  serializer.data,
            'center': {'lat': lat, 'lng': lng}
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 3: Full Suitability Analysis
# POST /api/analyze/
# The main endpoint — takes a pinned location + cafe type
# and returns: nearby cafes, top 5, suitability score, prediction
# ═══════════════════════════════════════════════════════════════════
class SuitabilityAnalysisView(APIView):

    def post(self, request):
        # Step 1: Validate incoming data
        serializer = SuitabilityRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        lat       = serializer.validated_data['lat']
        lng       = serializer.validated_data['lng']
        cafe_type = serializer.validated_data['cafe_type']

        # Step 2: Build the search point
        point = Point(lng, lat, srid=4326)

        # Step 3: Get all cafes within 500m
        nearby = (
            Cafe.objects
            .filter(location__distance_lte=(point, D(m=500)), is_open=True)
            .annotate(distance=Distance('location', point))
        )
        total_competitors = nearby.count()

        # Step 4: Get Top 5 cafés ranked by our score formula
        top5_qs = sorted(
            nearby,
            key=lambda c: (c.rating or 0) * math.log(c.review_count + 1),
            reverse=True
        )[:5]

        # Step 5: Get population density from the ward containing this point
        ward = Ward.objects.filter(boundary__contains=point).first()
        pop_density = ward.population_density if ward else 5000  # fallback

        # Step 6: Compute road length within 500m
        # Sum the length of all road segments that intersect our buffer
        from django.contrib.gis.geos import GEOSGeometry
        from django.db.models import Sum
        from django.contrib.gis.db.models.functions import Length
        road_length = (
            Road.objects
            .filter(geometry__distance_lte=(point, D(m=500)))
            .aggregate(total=Sum(Length('geometry')))
            .get('total') or 0
        )
        road_m = float(road_length) * 111139  # degrees → meters approx

        # Step 7: Calculate suitability score (0–100)
        # w1=competition penalty, w2=road access bonus, w3=population bonus
        competitor_score = max(0, 1 - (total_competitors / 20)) * 40
        road_score        = min(1, road_m / 3000) * 30
        pop_score         = min(1, pop_density / 15000) * 30
        suitability_score = round(competitor_score + road_score + pop_score)

        # Step 8: Get ML prediction
        # Build the feature vector the model expects
        features = [
            total_competitors,
            nearby.aggregate(avg=__import__('django.db.models', fromlist=['Avg']).Avg('rating'))['avg'] or 0,
            road_m,
            pop_density,
        ]
        prediction = get_prediction(features)  # from ml_engine/predictor.py

        # Step 9: Build and return the full response
        return Response({
            'location':  {'lat': lat, 'lng': lng},
            'nearby_count': total_competitors,
            'top5':       CafeSerializer(top5_qs, many=True).data,
            'suitability': {
                'score':           suitability_score,
                'competitor_count': total_competitors,
                'road_length_m':   round(road_m),
                'population_density': pop_density,
            },
            'prediction': prediction,  # { type: "Bakery Café", confidence: 0.87 }
        })