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
        this.analysisRadius = 500;
        this.initialized = false;
        this.lastAnalysisData = null;
        this.lastAmenitiesReport = null;
        this.lastPopulationData = null;
    }

    init() {
        // Guard: only initialize once, and only when the map div is visible
        const mapEl = document.getElementById('map');
        if (!mapEl || this.initialized) return;

        // Check map page is active
        const mapPage = document.getElementById('map-page');
        if (!mapPage || !mapPage.classList.contains('active')) return;

        this.initialized = true;
        this.initializeMap();
        this.setupEventListeners();
        this.loadDatasetStats();
    }

    initializeMap() {
        const mapEl = document.getElementById('map');
        if (!mapEl) return;

        // Destroy existing map instance if any (safety)
        if (this.map) {
            this.map.remove();
            this.map = null;
        }

        this.map = L.map('map').setView([27.7172, 85.3240], 13);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19,
        }).addTo(this.map);

        const kathmanduBounds = L.latLngBounds(
            [27.55, 85.10],
            [27.90, 85.55]
        );
        this.map.setMaxBounds(kathmanduBounds);
    }

    setupEventListeners() {
        if (!this.map) return;

        this.map.on('click', (e) => {
            this.handleMapClick(e.latlng);
        });

        const cafeTypeSelect = document.getElementById('cafe-type-select');
        if (cafeTypeSelect) {
            cafeTypeSelect.addEventListener('change', (e) => {
                this.selectedCafeType = e.target.value;
                if (this.selectedLocation && this.selectedCafeType) {
                    this.analyzeLocation();
                }
            });
        }

        const radiusSlider = document.getElementById('radius-slider');
        const radiusValue = document.getElementById('radius-value');
        if (radiusSlider && radiusValue) {
            radiusSlider.addEventListener('input', (e) => {
                this.analysisRadius = parseInt(e.target.value);
                radiusValue.textContent = this.analysisRadius;
                if (this.circle && this.selectedLocation) {
                    this.circle.setRadius(this.analysisRadius);
                }
            });

            radiusSlider.addEventListener('change', (e) => {
                if (this.selectedLocation && this.selectedCafeType) {
                    this.analyzeLocation();
                }
            });
        }

        const fullReportBtn = document.getElementById('full-report-btn');
        if (fullReportBtn) {
            fullReportBtn.addEventListener('click', () => {
                this.showFullReport();
            });
        }

        const downloadPdfBtn = document.getElementById('download-pdf-btn');
        if (downloadPdfBtn) {
            downloadPdfBtn.addEventListener('click', () => {
                this.downloadReportAsPdf();
            });
        }

        const modalClose = document.querySelector('.modal-close');
        if (modalClose) {
            modalClose.addEventListener('click', () => this.hideFullReport());
        }

        const modal = document.getElementById('full-report-modal');
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) this.hideFullReport();
            });
        }
    }

    handleMapClick(latlng) {
        const { lat, lng } = latlng;

        this.clearMarkerAndCircle();

        this.marker = L.marker([lat, lng]).addTo(this.map);
        this.marker.bindPopup(
            `<b>📍 Selected Location</b><br>Lat: ${lat.toFixed(6)}<br>Lng: ${lng.toFixed(6)}`
        ).openPopup();

        this.circle = L.circle([lat, lng], {
            color: '#6c5ce7',
            fillColor: '#6c5ce7',
            fillOpacity: 0.08,
            weight: 2,
            radius: this.analysisRadius
        }).addTo(this.map);

        this.selectedLocation = { lat, lng };
        this.updateCoordinatesDisplay(lat, lng);

        if (this.selectedCafeType) {
            this.analyzeLocation();
        } else {
            this.showCafeTypePrompt();
        }
    }

    async analyzeLocation() {
        if (!this.selectedLocation || !this.selectedCafeType) return;

        const { lat, lng } = this.selectedLocation;
        this.showLoading(true);

        try {
            if (!window.apiManager) {
                throw new Error('API manager not initialized');
            }

            const analysisData = await window.apiManager.getSuitabilityAnalysis(
                lat, lng, this.selectedCafeType, this.analysisRadius
            );

            this.lastAnalysisData = analysisData;
            this.displayAnalysisResults(analysisData);

            if (analysisData.top5) {
                this.displayNearbyCafes(analysisData.top5);
            }

        } catch (error) {
            console.error('Analysis failed:', error);
            this.showAnalysisError(error.message);
        } finally {
            this.showLoading(false);
        }
    }

    displayNearbyCafes(cafes) {
        this.clearCafeMarkers();

        cafes.forEach((cafe, index) => {
            if (!cafe.latitude || !cafe.longitude) return;

            const marker = L.circleMarker([cafe.latitude, cafe.longitude], {
                radius: 8,
                fillColor: '#fdcb6e',
                color: '#e17055',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(this.map);

            const ratingLine = cafe.rating ? `Rating: ⭐ ${cafe.rating}<br>` : '';
            marker.bindPopup(`
                <b>${cafe.name}</b><br>
                <small>Type: ${(cafe.cafe_type || '').replace('_', ' ')}</small><br>
                ${ratingLine}
                Reviews: ${cafe.review_count || 0}
            `);

            this.cafeMarkers.push(marker);
        });
    }

    displayAnalysisResults(data) {
        const suitability = data.suitability || {};
        const prediction = data.prediction || {};

        // Suitability score
        const scoreEl = document.getElementById('suitability-score');
        if (scoreEl) {
            scoreEl.textContent = suitability.score || '-';
            this.updateScoreCircle(suitability.score || 0);
        }

        // ML Prediction card
        const predictionTypeEl = document.querySelector('#prediction-card .prediction-type');
        const predictionConfEl = document.querySelector('#prediction-card .prediction-confidence');
        if (predictionTypeEl) {
            // Show recommended cafe type from AI recommendation
            predictionTypeEl.textContent = prediction.recommended_cafe_type || prediction.predicted_suitability || suitability.level || 'Unknown';
        }
        if (predictionConfEl) {
            // Remove suitability confidence display as requested
            predictionConfEl.textContent = '';
        }

        // Top 5 cafes
        const top5List = document.getElementById('top5-list');
        if (top5List && data.top5) {
            if (data.top5.length === 0) {
                top5List.innerHTML = '<p class="no-data">No cafes found in this area</p>';
            } else {
                top5List.innerHTML = data.top5.map(cafe => {
                    const ratingHtml = cafe.rating ? `<div class="cafe-rating">⭐ ${cafe.rating}</div>` : '';
                    return `
                    <div class="cafe-item">
                        <div class="cafe-name">${cafe.name}</div>
                        ${ratingHtml}
                    </div>
                `;
                }).join('');
            }
        }

        // Metrics - conditionally show based on actual values
        const competitorEl = document.getElementById('competitor-count');
        const roadEl = document.getElementById('road-length');
        const popEl = document.getElementById('population-density');

        const competitorValue = suitability.competitor_count ?? data.nearby_count;
        const roadValue = suitability.road_distance_m != null ? suitability.road_distance_m : suitability.road_length_m;
        const popValue = suitability.population_density;

        // Hide/show metric boxes based on whether they have values
        if (competitorEl) {
            const metricBox = competitorEl.closest('.metric');
            if (competitorValue != null) {
                competitorEl.textContent = competitorValue;
                if (metricBox) metricBox.style.display = 'block';
            } else {
                if (metricBox) metricBox.style.display = 'none';
            }
        }

        if (roadEl) {
            const metricBox = roadEl.closest('.metric');
            if (roadValue != null) {
                roadEl.textContent = roadValue + 'm';
                if (metricBox) metricBox.style.display = 'block';
            } else {
                if (metricBox) metricBox.style.display = 'none';
            }
        }

        if (popEl) {
            const metricBox = popEl.closest('.metric');
            if (popValue != null) {
                popEl.textContent = Number(popValue).toLocaleString() + '/km²';
                if (metricBox) metricBox.style.display = 'block';
            } else {
                if (metricBox) metricBox.style.display = 'none';
            }
        }
    }

    updateScoreCircle(score) {
        const scoreCircle = document.querySelector('.score-circle');
        if (!scoreCircle) return;

        let color = '#00b894'; // green
        if (score < 40) color = '#e17055';     // red
        else if (score < 70) color = '#fdcb6e'; // yellow

        scoreCircle.style.background = `conic-gradient(${color} 0% ${score}%, #e9ecef ${score}% 100%)`;
    }

    updateCoordinatesDisplay(lat, lng) {
        const coordsEl = document.getElementById('location-coords');
        if (coordsEl) {
            coordsEl.textContent = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
        }
    }

    showCafeTypePrompt() {
        const selectEl = document.getElementById('cafe-type-select');
        if (selectEl) {
            selectEl.style.borderColor = '#e17055';
            selectEl.style.boxShadow = '0 0 0 3px rgba(231,112,85,0.2)';
            selectEl.focus();
            setTimeout(() => {
                selectEl.style.borderColor = '';
                selectEl.style.boxShadow = '';
            }, 3000);
        }

        if (window.uiManager) {
            window.uiManager.showNotification('Please select a cafe type first!', 'warning');
        }
    }

    showLoading(show) {
        const loadingEl = document.getElementById('loading-indicator');
        if (loadingEl) {
            loadingEl.style.display = show ? 'block' : 'none';
        }
    }

    showAnalysisError(message) {
        const top5List = document.getElementById('top5-list');
        if (top5List) {
            top5List.innerHTML = `<p class="no-data" style="color:#e17055">Error: ${message || 'Analysis failed. Is the backend running?'}</p>`;
        }
        if (window.uiManager) {
            window.uiManager.showNotification('Analysis failed. Check backend connection.', 'error');
        }
    }

    async loadDatasetStats() {
        if (!window.apiManager) return;

        const statsEl = document.getElementById('dataset-stats');
        if (statsEl) {
            statsEl.innerHTML = '<p>Loading dataset insights...</p>';
        }

        try {
            const stats = await window.apiManager.getCafeDatasetStats();
            this.displayDatasetStats(stats);
        } catch (error) {
            console.warn('Unable to fetch dataset stats', error);
            if (statsEl) {
                statsEl.innerHTML = '<p style="color:#e17055">Unable to load dataset stats.</p>';
            }
        }
    }

    displayDatasetStats(stats) {
        const statsEl = document.getElementById('dataset-stats');
        if (!statsEl) return;

        if (!stats || typeof stats !== 'object') {
            statsEl.innerHTML = '<p style="color:#e17055">Dataset stats unavailable.</p>';
            return;
        }

        const typeList = stats.top_type_ranking || [];
        statsEl.innerHTML = `
            <div class="dataset-card-grid">
                <div class="dataset-card"><strong>Total Cafes:</strong><br>${this.formatNumber(stats.total_cafes || 0)}</div>
                <div class="dataset-card"><strong>Open Cafes:</strong><br>${this.formatNumber(stats.open_cafes || 0)}</div>
                <div class="dataset-card"><strong>Avg Rating:</strong><br>${stats.avg_rating !== null ? stats.avg_rating.toFixed(2) : 'N/A'}</div>
                <div class="dataset-card"><strong>Avg Reviews:</strong><br>${this.formatNumber(stats.avg_review_count || 0)}</div>
            </div>
            <div class="dataset-type-ranking">
                <strong>Top Cafe Types:</strong>
                <ul>${typeList.map(item => `<li>${item.type} — ${item.count}</li>`).join('')}</ul>
            </div>
        `;
    }

    formatNumber(value) {
        return Number(value || 0).toLocaleString('en-US');
    }

    clearMarkerAndCircle() {
        if (this.marker) { this.map.removeLayer(this.marker); this.marker = null; }
        if (this.circle) { this.map.removeLayer(this.circle); this.circle = null; }
    }

    clearCafeMarkers() {
        this.cafeMarkers.forEach(m => this.map.removeLayer(m));
        this.cafeMarkers = [];
    }

    clearMap() {
        if (this.map) {
            this.clearMarkerAndCircle();
            this.clearCafeMarkers();
        }
        this.selectedLocation = null;
        this.selectedCafeType = null;
        this.lastAnalysisData = null;
        this.clearResultsDisplay();
    }

    clearResultsDisplay() {
        const scoreEl = document.getElementById('suitability-score');
        if (scoreEl) scoreEl.textContent = '-';

        const predType = document.querySelector('#prediction-card .prediction-type');
        const predConf = document.querySelector('#prediction-card .prediction-confidence');
        if (predType) predType.textContent = '-';
        if (predConf) predConf.textContent = '-';

        const top5List = document.getElementById('top5-list');
        if (top5List) top5List.innerHTML = '<p class="no-data">Click on the map to see results</p>';

        // Hide all metric boxes
        ['competitor-count', 'road-length', 'population-density'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = '-';
                const metricBox = el.closest('.metric');
                if (metricBox) metricBox.style.display = 'none';
            }
        });

        const coordsEl = document.getElementById('location-coords');
        if (coordsEl) coordsEl.textContent = '';
    }

    showFullReport() {
        const modal = document.getElementById('full-report-modal');
        const reportContent = document.getElementById('report-content');
        if (!modal || !reportContent || !this.selectedLocation) {
            if (window.uiManager) {
                window.uiManager.showNotification('Pin a location first to generate a report.', 'warning');
            }
            return;
        }

        // Show loading state
        reportContent.innerHTML = '<div style="padding: 20px; text-align: center;"><p>⏳ Loading detailed report...</p></div>';
        modal.style.display = 'block';

        // Fetch amenities and population data
        this.fetchReportData().then(() => {
            reportContent.innerHTML = this.generateFullReport();
            // Attach event handlers for "See more" buttons after content is inserted
            this.attachAmenityHandlers();
        }).catch(error => {
            console.error('Error fetching report data:', error);
            reportContent.innerHTML = '<div style="padding: 20px; color: #e17055;"><p>⚠️ Error loading report data. Please try again.</p></div>';
        });
    }

    attachAmenityHandlers() {
        const buttons = document.querySelectorAll('.see-more-amenities');
        buttons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const type = btn.getAttribute('data-type');
                this.showFullAmenityList(type, btn);
            });
        });
    }

    showFullAmenityList(type, triggerButton) {
        if (!this.lastAmenitiesReport) return;
        const report = this.lastAmenitiesReport.amenities_report || {};
        const data = report[type];
        if (!data) return;

        // Build full list HTML with extra details (name, distance if available)
        const itemsHtml = data.amenities.map(a => {
            const name = a.name || 'Unnamed';
            const details = [];
            if (a.amenity_type) details.push(a.amenity_type);
            if (a.latitude && a.longitude) details.push(`${a.latitude.toFixed(6)}, ${a.longitude.toFixed(6)}`);
            if (a.distance != null) details.push(`${Math.round(a.distance)} m`);
            return `<li style="margin:6px 0"><strong>${name}</strong>${details.length?` — <span style="color:#666">${details.join(' · ')}</span>`:''}</li>`;
        }).join('');

        // Replace the parent <ul> content where the button was located
        const li = triggerButton.closest('li');
        if (li) {
            const parentUl = li.parentElement;
            if (parentUl) {
                parentUl.innerHTML = itemsHtml + `<li style="margin-top:8px"><button class="collapse-amenities" style="background:none;border:none;color:#0984e3;cursor:pointer;padding:0">Show less</button></li>`;
                const collapseBtn = parentUl.querySelector('.collapse-amenities');
                if (collapseBtn) collapseBtn.addEventListener('click', () => {
                    // Re-render the report to restore original truncated view
                    const reportContent = document.getElementById('report-content');
                    if (reportContent) reportContent.innerHTML = this.generateFullReport();
                    this.attachAmenityHandlers();
                });
            }
        }
    }

    async fetchReportData() {
        if (!this.selectedLocation) return;

        const { lat, lng } = this.selectedLocation;

        try {
            // Fetch amenities report (schools, hospitals, bus stops, cafes)
            const amenitiesResponse = await fetch(
                `http://localhost:8000/api/amenities-report/`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        lat: lat,
                        lng: lng,
                        radius: this.analysisRadius
                    })
                }
            );

            if (amenitiesResponse.ok) {
                this.lastAmenitiesReport = await amenitiesResponse.json();
            }

            // Fetch area population
            const populationResponse = await fetch(
                `http://localhost:8000/api/area-population/?lat=${lat}&lng=${lng}&radius=${this.analysisRadius}`
            );

            if (populationResponse.ok) {
                this.lastPopulationData = await populationResponse.json();
            }
        } catch (error) {
            console.error('Error fetching report data:', error);
            this.lastAmenitiesReport = null;
            this.lastPopulationData = null;
        }
    }

    hideFullReport() {
        const modal = document.getElementById('full-report-modal');
        if (modal) modal.style.display = 'none';
    }

    generateFullReport() {
        if (!this.selectedLocation) return '<p>No location selected.</p>';

        const { lat, lng } = this.selectedLocation;
        const score = document.getElementById('suitability-score')?.textContent || '-';
        const competitors = document.getElementById('competitor-count')?.textContent || '-';
        const roadLength = document.getElementById('road-length')?.textContent || '-';
        const population = document.getElementById('population-density')?.textContent || '-';
        const cafeTypeFormatted = this.selectedCafeType
            ? this.selectedCafeType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
            : 'Not selected';

        const prediction = this.lastAnalysisData?.prediction || {};
        const probabilities = prediction.all_probabilities || {};

        // Get amenities report and population data
        const amenitiesReport = this.lastAmenitiesReport?.amenities_report || {};
        const populationData = this.lastPopulationData || {};

        return `
            <div class="report-section">
                <h3>📍 Location Details</h3>
                <div class="report-grid">
                    <div class="report-item"><strong>Coordinates:</strong><br>${lat.toFixed(6)}, ${lng.toFixed(6)}</div>
                    <div class="report-item"><strong>Analysis Radius:</strong><br>${this.analysisRadius} meters</div>
                    <div class="report-item"><strong>Cafe Type:</strong><br>${cafeTypeFormatted}</div>
                </div>
            </div>

            <div class="report-section">
                <h3>📊 Suitability Analysis</h3>
                <div class="report-grid">
                    <div class="report-item"><strong>Overall Score:</strong><br>${score} / 100</div>
                    <div class="report-item"><strong>Competitors Nearby:</strong><br>${competitors}</div>
                    <div class="report-item"><strong>Road Accessibility:</strong><br>${roadLength}</div>
                    <div class="report-item"><strong>Population Density:</strong><br>${population}</div>
                    <div class="report-item"><strong>Total Population (in radius):</strong><br>${populationData && populationData.total_population ? Number(populationData.total_population).toLocaleString() : '0'}</div>
                </div>
            </div>

            ${populationData.total_population ? `
            <div class="report-section">
                <h3>👥 Population in Selected Area</h3>
                <div class="report-grid">
                    <div class="report-item">
                        <strong>Total Population:</strong><br>
                        <span style="font-size: 1.4em; color: #6c5ce7; font-weight: bold;">
                            ${Number(populationData.total_population).toLocaleString()}
                        </span>
                    </div>
                    <div class="report-item">
                        <strong>Affected Wards:</strong><br>
                        ${populationData.affected_ward_count || 0} ward${populationData.affected_ward_count !== 1 ? 's' : ''}
                    </div>
                </div>
                ${populationData.affected_wards && populationData.affected_wards.length > 0 ? `
                <div style="margin-top: 15px; padding: 10px; background: #f5f6fa; border-radius: 5px;">
                    <strong>Ward Details:</strong>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        ${populationData.affected_wards.map(ward => `
                            <li>Ward ${ward.ward_number}: ${Number(ward.population).toLocaleString()} population, ${ward.population_density.toFixed(0)}/km²</li>
                        `).join('')}
                    </ul>
                </div>
                ` : ''}
            </div>
            ` : ''}

            ${Object.keys(amenitiesReport).length > 0 ? `
            <div class="report-section">
                <h3>🏘️ Amenities in Selected Area</h3>
                <div class="report-grid">
                    ${Object.entries(amenitiesReport).map(([type, data]) => `
                        <div class="report-item">
                            <strong>${type.replace(/_/g, ' ').toUpperCase()}:</strong><br>
                            <span style="font-size: 1.3em; color: #00b894;">${data.count}</span>
                        </div>
                    `).join('')}
                </div>
                
                <div style="margin-top: 15px; padding: 10px; background: #f5f6fa; border-radius: 5px;">
                    <strong>Amenity Listings:</strong>
                    ${Object.entries(amenitiesReport).map(([type, data]) => {
                        if (data.count === 0) return '';
                        return `
                            <div style="margin: 10px 0;">
                                <strong>${type.replace(/_/g, ' ').toUpperCase()} (${data.count}):</strong>
                                <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                                    ${data.amenities.slice(0, 5).map(amenity => `
                                        <li>${amenity.name || 'Unnamed'}</li>
                                    `).join('')}
                                            ${data.count > 5 ? `<li><button class="see-more-amenities" data-type="${type}" style="background:none;border:none;color:#0984e3;cursor:pointer;padding:0;margin:0">See more (${data.count - 5})</button></li>` : ''}
                                </ul>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
            ` : ''}

            ${Object.keys(probabilities).length > 0 ? `
            <div class="report-section">
                <h3>🤖 ML Model Probabilities</h3>
                <div class="report-grid">
                    ${Object.entries(probabilities).map(([label, prob]) => `
                        <div class="report-item">
                            <strong>${label}:</strong><br>${(prob * 100).toFixed(1)}%
                        </div>
                    `).join('')}
                </div>
            </div>` : ''}

            <div class="report-insights">
                <h4>💡 Key Insights & Recommendations</h4>
                <ul>
                    <li><strong>Location Strength:</strong> ${this._getLocationStrength(parseInt(score))}</li>
                    <li><strong>Competition Level:</strong> ${this._getCompetitionLevel(competitors)}</li>
                    <li><strong>Market Potential:</strong> ${this._getMarketPotential(population)}</li>
                    <li><strong>Recommendation:</strong> ${parseInt(score) >= 60
                        ? '✅ This location shows good potential for a cafe business.'
                        : '⚠️ Consider alternative locations with less competition or better road access.'}</li>
                </ul>
            </div>
        `;
    }

    downloadReportAsPdf() {
        if (!this.selectedLocation) {
            if (window.uiManager) {
                window.uiManager.showNotification('Pin a location first to generate a report.', 'warning');
            }
            return;
        }

        // Generate the report HTML
        const reportHtml = this.generateFullReport();
        
        // Create a temporary container for PDF generation
        const element = document.createElement('div');
        element.style.padding = '20px';
        element.style.backgroundColor = '#fff';
        element.innerHTML = `
            <h1>CafeLocate - Location Analysis Report</h1>
            <p style="color: #666; margin-bottom: 20px;">Generated on ${new Date().toLocaleDateString()}</p>
            ${reportHtml}
        `;

        // Configure PDF generation options
        const opt = {
            margin: 10,
            filename: 'cafelocate-report.pdf',
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2 },
            jsPDF: { orientation: 'portrait', unit: 'mm', format: 'a4' }
        };

        // Generate and download PDF
        html2pdf().set(opt).from(element).save();
    }

    _getLocationStrength(score) {
        if (score >= 80) return "Excellent – high success potential";
        if (score >= 60) return "Good – moderate success potential";
        if (score >= 40) return "Fair – consider improvements";
        return "Poor – high risk, explore alternatives";
    }

    _getCompetitionLevel(count) {
        const n = parseInt(count) || 0;
        if (n < 5) return "Low competition – great opportunity";
        if (n < 15) return "Moderate competition – viable market";
        return "High competition – saturated market";
    }

    _getMarketPotential(population) {
        const density = parseInt((population || '').replace(/[^0-9]/g, '')) || 0;
        if (density > 15000) return "High population density – strong market";
        if (density > 8000) return "Moderate density – decent market";
        return "Low density – limited foot traffic expected";
    }
}

// Do NOT auto-init here — init() is called by authManager.showMapPage()
document.addEventListener('DOMContentLoaded', () => {
    window.mapManager = new MapManager();
});
