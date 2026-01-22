# ForestSathi

**ForestSathi** (meaning "Forest Friend" in Nepali) is a wildfire intelligence platform built specifically for Nepal. It uses machine learning trained on 190,000+ NASA satellite fire detections to predict fire risk for any location in the country.

## The Problem: Wildfires in Nepal

Nepal loses thousands of hectares of forest annually to wildfires. The country's unique geography creates three distinct fire-prone zones:

- **Terai Plains** (< 500m): Agricultural burning and dry grassland fires dominate the southern lowlands
- **Mahabharat Hills** (500m - 3,000m): Accumulated dry pine needles and slash-and-burn practices cause most fires in the mid-hills
- **High Himalayas** (> 3,000m): Strong seasonal winds and dry alpine shrubs fuel fires in remote mountain regions

Fire season peaks from February to May when dry conditions, accumulated biomass, and human activities create dangerous conditions. Most fires are human-caused, making prediction and awareness critical for prevention.

## What ForestSathi Does

- **Risk Prediction**: Enter any Nepal location to get AI-powered wildfire risk assessment (Low/Moderate/High)
- **Interactive Heatmap**: Visualize 190,000+ historical fire events across Nepal
- **Regional Intelligence**: Automatic classification into ecological zones with region-specific fire patterns, causes, and seasonal trends
- **Educational Insights**: Learn about fire behavior unique to each region

## Tech Stack & Libraries

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core programming language |
| **Flask 2.3+** | Lightweight web framework for serving the application |
| **Gunicorn 21+** | Production-grade WSGI HTTP server for deployment |

### Machine Learning & Data Processing
| Library | Purpose |
|---------|---------|
| **scikit-learn 1.3+** | Random Forest Classifier for fire risk prediction (98.72% accuracy) |
| **Pandas 2.0+** | Data manipulation and CSV processing for 190K+ fire records |
| **NumPy 1.24+** | Numerical computations and array operations |
| **pickle** | Model serialization (model.pkl, encoder.pkl, region_stats.pkl) |

### Geocoding & Location Services
| Library | Purpose |
|---------|---------|
| **Geopy 2.4+** | Convert location names to coordinates using Nominatim (OpenStreetMap) |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5/CSS3** | Responsive UI with dark theme design |
| **JavaScript (ES6)** | Client-side interactivity and API calls |
| **Leaflet.js 1.9** | Interactive map rendering with custom markers |
| **leaflet-heat 0.2** | Heatmap visualization layer for fire hotspots |
| **Inter Font** | Modern typography via Google Fonts |

### Data Sources
| Source | Description |
|--------|-------------|
| **NASA FIRMS** | MODIS/VIIRS satellite fire detection data |
| **ICIMOD Nepal** | Regional ecological zone classifications |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/prashantsubedii/ForestSathi.git
cd ForestSathi
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model (Optional)

The repository includes pre-trained model files (`model.pkl`, `encoder.pkl`, `region_stats.pkl`). If you want to retrain:

```bash
python train_model.py
```

### Step 5: Run the Application

```bash
python app_flask.py
```

Open your browser and go to **http://localhost:5000**

---

## Project Structure

```
ForestSathi/
├── app_flask.py                    # Flask web application
├── train_model.py                  # Model training script
├── templates/
│   ├── index.html                  # Main dashboard
│   └── about.html                  # About page
├── static/
│   └── logo.png                    # Application logo
├── model.pkl                       # Trained Random Forest model
├── encoder.pkl                     # Region label encoder
├── region_stats.pkl                # Regional fire statistics
├── forestsathi_training_data.csv   # Training dataset (190K+ records)
├── requirements.txt                # Python dependencies
├── Procfile                        # Deployment config (Render/Heroku)
└── render.yaml                     # Render deployment config
```

## Data Sources

- **[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)** — Fire Information for Resource Management System providing active fire detections via MODIS and VIIRS satellites
- **[ICIMOD](https://www.icimod.org/)** — International Centre for Integrated Mountain Development for regional ecological classifications

## Author

**Prashant Subedi**

## License

MIT
