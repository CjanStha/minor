from django.test import TestCase
from django.urls import reverse

from .models import Cafe


class CafeApiTests(TestCase):
    def test_cafe_stats_empty(self):
        response = self.client.get('/api/cafes/stats/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['total_cafes'], 0)
        self.assertEqual(data['open_cafes'], 0)
        self.assertIsNone(data['avg_rating'])
        self.assertEqual(data['avg_review_count'], 0)
        self.assertEqual(data['type_counts'], {})
        self.assertEqual(data['top_type_ranking'], [])

    def test_cafe_stats_aggregation(self):
        Cafe.objects.create(
            place_id='test-1',
            name='Cafe One',
            cafe_type='coffee_shop',
            latitude=27.70,
            longitude=85.30,
            location={'type': 'Point', 'coordinates': [85.30, 27.70]},
            rating=4.5,
            review_count=120,
            is_open=True
        )

        Cafe.objects.create(
            place_id='test-2',
            name='Cafe Two',
            cafe_type='bakery',
            latitude=27.71,
            longitude=85.31,
            location={'type': 'Point', 'coordinates': [85.31, 27.71]},
            rating=4.0,
            review_count=80,
            is_open=False
        )

        Cafe.objects.create(
            place_id='test-3',
            name='Cafe Three',
            cafe_type='coffee_shop',
            latitude=27.72,
            longitude=85.32,
            location={'type': 'Point', 'coordinates': [85.32, 27.72]},
            rating=None,
            review_count=0,
            is_open=True
        )

        response = self.client.get('/api/cafes/stats/')
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data['total_cafes'], 3)
        self.assertEqual(data['open_cafes'], 2)
        self.assertAlmostEqual(data['avg_rating'], 4.25)
        self.assertAlmostEqual(data['avg_review_count'], 66.7, places=1)
        self.assertEqual(data['type_counts']['coffee_shop'], 2)
        self.assertEqual(data['type_counts']['bakery'], 1)
        self.assertEqual(data['top_type_ranking'][0]['type'], 'coffee_shop')

    def test_nearby_cafes_by_distance(self):
        Cafe.objects.create(
            place_id='nearby-1',
            name='Nearby Cafe',
            cafe_type='coffee_shop',
            latitude=27.7172,
            longitude=85.3240,
            location={'type': 'Point', 'coordinates': [85.3240, 27.7172]},
            rating=4.4,
            review_count=50,
            is_open=True
        )

        Cafe.objects.create(
            place_id='far-1',
            name='Far Cafe',
            cafe_type='bakery',
            latitude=27.80,
            longitude=85.40,
            location={'type': 'Point', 'coordinates': [85.40, 27.80]},
            rating=3.5,
            review_count=10,
            is_open=True
        )

        response = self.client.get('/api/cafes/nearby/', {'lat': 27.7172, 'lng': 85.3240, 'radius': 1000})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['count'], 1)
        self.assertEqual(data['cafes'][0]['place_id'], 'nearby-1')

        # with big radius includes far cafe
        response2 = self.client.get('/api/cafes/nearby/', {'lat': 27.7172, 'lng': 85.3240, 'radius': 15000})
        self.assertEqual(response2.status_code, 200)
        data2 = response2.json()
        self.assertEqual(data2['count'], 2)
