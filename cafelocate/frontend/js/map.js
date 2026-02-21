// CafeLocate - Map Module
// Handles Leaflet map initialization, markers, and user interactions

class MapManager {
    constructor() {
        this.map = null;
        this.marker = null;
        this.circle = null;
        this.cafeMarkers = [];
        this.selectedLocation = null;
        this.selectedCafeType = null;
        this.analysisRadius = 500; // Default 500 meters
        this.init();
    }

    init() {
        if (!document.getElementById('map')) return;

        this.initializeMap();
        this.setupEventListeners();
        this.loadKathmanduBounds();
    }

    initializeMap() {
        // Initialize Leaflet map centered on Kathmandu
        this.map = L.map('map').setView([27.7172, 85.3240], 12);

        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18,
        }).addTo(this.map);

        // Set bounds to Kathmandu Valley (more inclusive)
        const kathmanduBounds = L.latLngBounds(
            [27.5, 85.1], // Southwest - more west and south
            [27.9, 85.6]  // Northeast - more east and north
        );
        this.map.setMaxBounds(kathmanduBounds);
        this.map.on('drag', () => {
            this.map.panInsideBounds(kathmanduBounds, { animate: false });
        });
    }

    setupEventListeners() {
        // Map click event
        this.map.on('click', (e) => {
            this.handleMapClick(e.latlng);
        });

        // Cafe type selector
        const cafeTypeSelect = document.getElementById('cafe-type-select');
        if (cafeTypeSelect) {
            cafeTypeSelect.addEventListener('change', (e) => {
                this.selectedCafeType = e.target.value;
                if (this.selectedLocation) {
                    this.analyzeLocation();
                }
            });
        }

        // Radius slider
        const radiusSlider = document.getElementById('radius-slider');
        const radiusValue = document.getElementById('radius-value');
        if (radiusSlider && radiusValue) {
            radiusSlider.addEventListener('input', (e) => {
                this.analysisRadius = parseInt(e.target.value);
                radiusValue.textContent = this.analysisRadius;
                // Update circle radius if marker exists
                if (this.circle && this.selectedLocation) {
                    this.circle.setRadius(this.analysisRadius);
                    // Re-analyze with new radius
                    if (this.selectedCafeType) {
                        this.analyzeLocation();
                    }
                }
            });
        }

        // Full report button
        const fullReportBtn = document.getElementById('full-report-btn');
        if (fullReportBtn) {
            fullReportBtn.addEventListener('click', () => {
                this.showFullReport();
            });
        }

        // Modal close
        const modalClose = document.querySelector('.modal-close');
        if (modalClose) {
            modalClose.addEventListener('click', () => {
                this.hideFullReport();
            });
        }

        // Close modal when clicking outside
        const modal = document.getElementById('full-report-modal');
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideFullReport();
                }
            });
        }
    }

    handleMapClick(latlng) {
        const { lat, lng } = latlng;

        // Remove existing marker and circle
        this.clearMarkerAndCircle();

        // Add new marker
        this.marker = L.marker([lat, lng]).addTo(this.map);
        this.marker.bindPopup(`<b>Location Selected</b><br>Lat: ${lat.toFixed(6)}<br>Lng: ${lng.toFixed(6)}`).openPopup();

        // Add 500m radius circle (using adjustable radius)
        this.circle = L.circle([lat, lng], {
            color: '#e74c3c',
            fillColor: '#e74c3c',
            fillOpacity: 0.1,
            radius: this.analysisRadius
        }).addTo(this.map);

        // Store selected location
        this.selectedLocation = { lat, lng };

        // Update coordinates display
        this.updateCoordinatesDisplay(lat, lng);

        // Analyze location if cafe type is selected
        if (this.selectedCafeType) {
            this.analyzeLocation();
        } else {
            this.showCafeTypePrompt();
        }
    }

    async analyzeLocation() {
        if (!this.selectedLocation || !this.selectedCafeType) return;

        const { lat, lng } = this.selectedLocation;

        // Show loading indicator
        this.showLoading(true);

        try {
            // Get nearby cafes
            const nearbyData = await window.apiManager.getNearbyCafes(lat, lng, this.analysisRadius);
            this.displayNearbyCafes(nearbyData.cafes);

            // Get full analysis
            const analysisData = await window.apiManager.getSuitabilityAnalysis(lat, lng, this.selectedCafeType, this.analysisRadius);
            this.displayAnalysisResults(analysisData);

        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError(window.apiManager.formatError(error));
        } finally {
            this.showLoading(false);
        }
    }

    displayNearbyCafes(cafes) {
        // Clear existing cafe markers
        this.clearCafeMarkers();

        // Add markers for each cafe
        cafes.forEach(cafe => {
            const marker = L.marker([cafe.latitude, cafe.longitude])
                .addTo(this.map)
                .bindPopup(`
                    <b>${cafe.name}</b><br>
                    Type: ${cafe.cafe_type}<br>
                    Rating: ${cafe.rating || 'N/A'} ⭐<br>
                    Reviews: ${cafe.review_count || 0}
                `);

            this.cafeMarkers.push(marker);
        });
    }

    displayAnalysisResults(data) {
        // Update suitability score
        const scoreEl = document.getElementById('suitability-score');
        if (scoreEl) {
            scoreEl.textContent = data.suitability.score;
            this.updateScoreCircle(data.suitability.score);
        }

        // Update prediction
        const predictionCard = document.getElementById('prediction-card');
        if (predictionCard && data.prediction) {
            predictionCard.querySelector('.prediction-type').textContent = data.prediction.predicted_type;
            predictionCard.querySelector('.prediction-confidence').textContent =
                `Confidence: ${(data.prediction.confidence * 100).toFixed(1)}%`;
        }

        // Update top 5 cafes
        const top5List = document.getElementById('top5-list');
        if (top5List && data.top5) {
            top5List.innerHTML = data.top5.map(cafe => `
                <div class="cafe-item">
                    <div class="cafe-name">${cafe.name}</div>
                    <div class="cafe-rating">${cafe.rating || 'N/A'} ⭐</div>
                </div>
            `).join('');
        }

        // Update detailed metrics
        this.updateMetrics(data.suitability);
    }

    updateScoreCircle(score) {
        const scoreCircle = document.querySelector('.score-circle');
        if (scoreCircle) {
            const percentage = (score / 100) * 360;
            scoreCircle.style.background = `conic-gradient(var(--success-color) 0% ${score}%, #e0e0e0 ${score}% 100%)`;
        }
    }

    updateMetrics(suitability) {
        const updates = {
            'competitor-count': suitability.competitor_count,
            'road-length': `${suitability.road_length_m}m`,
            'population-density': `${suitability.population_density}/km²`
        };

        Object.entries(updates).forEach(([id, value]) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        });
    }

    updateCoordinatesDisplay(lat, lng) {
        const coordsEl = document.getElementById('location-coords');
        if (coordsEl) {
            coordsEl.textContent = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
        }
    }

    showCafeTypePrompt() {
        // Highlight the cafe type selector
        const selectEl = document.getElementById('cafe-type-select');
        if (selectEl) {
            selectEl.style.borderColor = '#e74c3c';
            selectEl.style.boxShadow = '0 0 0 2px rgba(231, 76, 60, 0.2)';

            setTimeout(() => {
                selectEl.style.borderColor = '';
                selectEl.style.boxShadow = '';
            }, 3000);
        }
    }

    showLoading(show) {
        const loadingEl = document.getElementById('loading-indicator');
        if (loadingEl) {
            loadingEl.style.display = show ? 'block' : 'none';
        }
    }

    showError(message) {
        // Simple error display - could be enhanced with a toast notification
        alert(`Error: ${message}`);
    }

    clearMarkerAndCircle() {
        if (this.marker) {
            this.map.removeLayer(this.marker);
            this.marker = null;
        }
        if (this.circle) {
            this.map.removeLayer(this.circle);
            this.circle = null;
        }
    }

    clearCafeMarkers() {
        this.cafeMarkers.forEach(marker => {
            this.map.removeLayer(marker);
        });
        this.cafeMarkers = [];
    }

    clearMap() {
        this.clearMarkerAndCircle();
        this.clearCafeMarkers();
        this.selectedLocation = null;
        this.selectedCafeType = null;

        // Clear results display
        this.clearResultsDisplay();
    }

    clearResultsDisplay() {
        const elements = [
            'suitability-score',
            'prediction-card',
            'top5-list',
            'competitor-count',
            'road-length',
            'population-density',
            'location-coords'
        ];

        elements.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                if (id === 'suitability-score' || id.includes('count') || id.includes('length') || id.includes('density')) {
                    el.textContent = '-';
                } else if (id === 'prediction-card') {
                    el.querySelector('.prediction-type').textContent = '-';
                    el.querySelector('.prediction-confidence').textContent = '-';
                } else if (id === 'top5-list') {
                    el.innerHTML = '<p class="no-data">Click on the map to see results</p>';
                } else if (id === 'location-coords') {
                    el.textContent = '';
                }
            }
        });
    }

    loadKathmanduBounds() {
        // This could load GeoJSON boundaries for more precise bounds
        // For now, using simple rectangular bounds set in initializeMap()
    }

    showFullReport() {
        const modal = document.getElementById('full-report-modal');
        const reportContent = document.getElementById('report-content');

        if (!modal || !reportContent || !this.selectedLocation) return;

        // Generate comprehensive report
        const reportHTML = this.generateFullReport();
        reportContent.innerHTML = reportHTML;

        modal.style.display = 'block';
    }

    hideFullReport() {
        const modal = document.getElementById('full-report-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    generateFullReport() {
        if (!this.selectedLocation) return '<p>No location selected for analysis.</p>';

        const { lat, lng } = this.selectedLocation;
        const score = document.getElementById('suitability-score')?.textContent || '-';
        const competitors = document.getElementById('competitor-count')?.textContent || '-';
        const roadLength = document.getElementById('road-length')?.textContent || '-';
        const population = document.getElementById('population-density')?.textContent || '-';

        return `
            <div class="report-section">
                <h3>📍 Location Details</h3>
                <div class="report-grid">
                    <div class="report-item">
                        <strong>Coordinates:</strong> ${lat.toFixed(6)}, ${lng.toFixed(6)}
                    </div>
                    <div class="report-item">
                        <strong>Analysis Radius:</strong> ${this.analysisRadius} meters
                    </div>
                    <div class="report-item">
                        <strong>Café Type:</strong> ${this.selectedCafeType ? this.selectedCafeType.replace('_', ' ').toUpperCase() : 'Not selected'}
                    </div>
                </div>
            </div>

            <div class="report-section">
                <h3>📊 Suitability Analysis</h3>
                <div class="report-grid">
                    <div class="report-item">
                        <strong>Overall Score:</strong> ${score}/100
                    </div>
                    <div class="report-item">
                        <strong>Competitor Count:</strong> ${competitors} cafés
                    </div>
                    <div class="report-item">
                        <strong>Road Accessibility:</strong> ${roadLength}
                    </div>
                    <div class="report-item">
                        <strong>Population Density:</strong> ${population}
                    </div>
                </div>
            </div>

            <div class="report-section">
                <h3>🤖 AI Prediction Details</h3>
                <div id="prediction-details">
                    <p>Loading prediction details...</p>
                </div>
            </div>

            <div class="report-section">
                <h3>⭐ Nearby Competitors</h3>
                <div id="competitors-details">
                    <p>Loading competitor details...</p>
                </div>
            </div>

            <div class="report-insights">
                <h4>💡 Key Insights & Recommendations</h4>
                <ul>
                    <li><strong>Location Strength:</strong> ${this.getLocationStrength(parseInt(score))}</li>
                    <li><strong>Competition Level:</strong> ${this.getCompetitionLevel(parseInt(competitors))}</li>
                    <li><strong>Accessibility:</strong> ${this.getAccessibilityLevel(roadLength)}</li>
                    <li><strong>Market Potential:</strong> ${this.getMarketPotential(population)}</li>
                </ul>
            </div>
        `;
    }

    getLocationStrength(score) {
        if (score >= 80) return "Excellent location with high success potential";
        if (score >= 60) return "Good location with moderate success potential";
        if (score >= 40) return "Fair location, consider improvements";
        return "Poor location, high risk of failure";
    }

    getCompetitionLevel(count) {
        if (count < 5) return "Low competition - good opportunity";
        if (count < 15) return "Moderate competition - competitive market";
        return "High competition - saturated market";
    }

    getAccessibilityLevel(roadLength) {
        const meters = parseInt(roadLength) || 0;
        if (meters > 2000) return "Excellent road access";
        if (meters > 1000) return "Good road access";
        return "Limited road access";
    }

    getMarketPotential(population) {
        const density = parseInt(population.replace(',', '').replace('/km²', '')) || 0;
        if (density > 15000) return "High population density - strong market";
        if (density > 10000) return "Moderate population density - good market";
        return "Low population density - limited market";
    }
}

// Initialize map manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.mapManager = new MapManager();
});
