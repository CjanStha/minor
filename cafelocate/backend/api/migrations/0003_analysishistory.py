from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_amenity'),
    ]

    operations = [
        migrations.CreateModel(
            name='AnalysisHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('latitude', models.FloatField()),
                ('longitude', models.FloatField()),
                ('cafe_type', models.CharField(max_length=50)),
                ('radius', models.IntegerField(default=500)),
                ('suitability_score', models.FloatField()),
                ('suitability_level', models.CharField(max_length=50)),
                ('recommended_cafe_type', models.CharField(blank=True, max_length=100)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='analysis_history', to='api.userprofile')),
            ],
            options={
                'db_table': 'analysis_history',
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='analysishistory',
            index=models.Index(fields=['user', 'cafe_type', 'created_at'], name='analysis_his_user_id_68efc2_idx'),
        ),
    ]
