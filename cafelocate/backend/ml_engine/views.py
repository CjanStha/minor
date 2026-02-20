from rest_framework.views import APIView
from rest_framework.response import Response
from .predictor import get_prediction


# POST /api/predict/
# Direct prediction endpoint — send features, get prediction back
# Useful for testing the ML model independently of the full analysis
class PredictView(APIView):

    def post(self, request):
        # Expect: { "features": [8, 4.2, 1200, 5000] }
        features = request.data.get('features')

        if not features or len(features) != 4:
            return Response(
                {'error': 'Send features: [competitor_count, avg_rating, road_m, pop_density]'},
                status=400
            )

        result = get_prediction(features)
        return Response(result)