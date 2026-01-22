"""
🌲 ForestSathi - Flask Application
Wildfire Intelligence Platform for Nepal
Based on NASA FIRMS VIIRS Satellite Data
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import json
from datetime import datetime
import hashlib

app = Flask(__name__)

# ======================= CONFIGURATION =======================
DATA_FILE = 'forestsathi_training_data.csv'
MODEL_FILE = 'model.pkl'
ENCODER_FILE = 'encoder.pkl'

# ======================= LOAD DATA & MODEL =======================
def load_resources():
    """Load model, encoder, and fire data"""
    global model, encoder, fire_data, heatmap_data, location_stats_cache
    
    location_stats_cache = {}
    
    # Load fire data
    if os.path.exists(DATA_FILE):
        fire_data = pd.read_csv(DATA_FILE)
        fire_data = fire_data.dropna(subset=['latitude', 'longitude', 'brightness'])
        
        # Parse dates
        fire_data['acq_date'] = pd.to_datetime(fire_data['acq_date'], errors='coerce')
        fire_data['year'] = fire_data['acq_date'].dt.year
        fire_data['month'] = fire_data['acq_date'].dt.month
        
        # Prepare heatmap data (sample for performance)
        sample_size = min(15000, len(fire_data))
        heatmap_sample = fire_data.sample(n=sample_size, random_state=42)
        
        # Normalize intensity
        max_brightness = heatmap_sample['brightness'].max()
        min_brightness = heatmap_sample['brightness'].min()
        
        heatmap_data = []
        for _, row in heatmap_sample.iterrows():
            intensity = (row['brightness'] - min_brightness) / (max_brightness - min_brightness)
            heatmap_data.append({
                'lat': round(row['latitude'], 4),
                'lng': round(row['longitude'], 4),
                'intensity': round(intensity, 2)
            })
        
        # Pre-compute district-level statistics for accurate predictions
        precompute_location_statistics()
    else:
        fire_data = None
        heatmap_data = []
    
    # Load model
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
    else:
        model = None
    
    # Load encoder
    if os.path.exists(ENCODER_FILE):
        with open(ENCODER_FILE, 'rb') as f:
            encoder = pickle.load(f)
    else:
        encoder = None

def precompute_location_statistics():
    """Pre-compute fire statistics for different grid cells across Nepal"""
    global location_stats_cache, fire_data
    
    if fire_data is None:
        return
    
    # Create grid cells (0.25 degree resolution ~ 25km)
    fire_data['grid_lat'] = (fire_data['latitude'] / 0.25).astype(int) * 0.25
    fire_data['grid_lon'] = (fire_data['longitude'] / 0.25).astype(int) * 0.25
    fire_data['grid_key'] = fire_data['grid_lat'].astype(str) + '_' + fire_data['grid_lon'].astype(str)
    
    # Compute statistics for each grid cell
    grid_stats = fire_data.groupby('grid_key').agg({
        'brightness': ['mean', 'max', 'std', 'count'],
        'frp': ['mean', 'max'],
        'latitude': 'mean',
        'longitude': 'mean',
        'year': lambda x: x.nunique(),
        'likely_cause': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
    }).reset_index()
    
    grid_stats.columns = ['grid_key', 'avg_brightness', 'max_brightness', 'std_brightness', 
                          'fire_count', 'avg_frp', 'max_frp', 'lat', 'lon', 
                          'years_with_fires', 'primary_cause']
    
    for _, row in grid_stats.iterrows():
        location_stats_cache[row['grid_key']] = {
            'avg_brightness': row['avg_brightness'],
            'max_brightness': row['max_brightness'],
            'std_brightness': row['std_brightness'] if pd.notna(row['std_brightness']) else 10,
            'fire_count': int(row['fire_count']),
            'avg_frp': row['avg_frp'],
            'max_frp': row['max_frp'],
            'years_with_fires': int(row['years_with_fires']),
            'primary_cause': row['primary_cause']
        }

# Initialize on startup
model = None
encoder = None
fire_data = None
heatmap_data = []
location_stats_cache = {}

# ======================= NEPAL DISTRICTS WITH ACCURATE DATA =======================
# Comprehensive district data with coordinates and fire risk characteristics
NEPAL_DISTRICTS = {
    'kathmandu': {'lat': 27.7172, 'lon': 85.3240, 'address': 'Kathmandu, Bagmati Province', 'zone': 'hills', 'urban': True, 'forest_cover': 0.15},
    'pokhara': {'lat': 28.2096, 'lon': 83.9856, 'address': 'Pokhara, Gandaki Province', 'zone': 'hills', 'urban': True, 'forest_cover': 0.35},
    'chitwan': {'lat': 27.5291, 'lon': 84.3542, 'address': 'Chitwan, Bagmati Province', 'zone': 'terai', 'urban': False, 'forest_cover': 0.55},
    'biratnagar': {'lat': 26.4525, 'lon': 87.2718, 'address': 'Biratnagar, Province No. 1', 'zone': 'terai', 'urban': True, 'forest_cover': 0.10},
    'lalitpur': {'lat': 27.6588, 'lon': 85.3247, 'address': 'Lalitpur, Bagmati Province', 'zone': 'hills', 'urban': True, 'forest_cover': 0.20},
    'bharatpur': {'lat': 27.6833, 'lon': 84.4333, 'address': 'Bharatpur, Bagmati Province', 'zone': 'terai', 'urban': True, 'forest_cover': 0.25},
    'birgunj': {'lat': 27.0104, 'lon': 84.8821, 'address': 'Birgunj, Province No. 2', 'zone': 'terai', 'urban': True, 'forest_cover': 0.08},
    'dharan': {'lat': 26.8065, 'lon': 87.2846, 'address': 'Dharan, Province No. 1', 'zone': 'terai', 'urban': True, 'forest_cover': 0.30},
    'butwal': {'lat': 27.7006, 'lon': 83.4483, 'address': 'Butwal, Lumbini Province', 'zone': 'terai', 'urban': True, 'forest_cover': 0.20},
    'hetauda': {'lat': 27.4167, 'lon': 85.0333, 'address': 'Hetauda, Bagmati Province', 'zone': 'terai', 'urban': False, 'forest_cover': 0.45},
    'janakpur': {'lat': 26.7271, 'lon': 85.9407, 'address': 'Janakpur, Province No. 2', 'zone': 'terai', 'urban': True, 'forest_cover': 0.05},
    'nepalgunj': {'lat': 28.0500, 'lon': 81.6167, 'address': 'Nepalgunj, Lumbini Province', 'zone': 'terai', 'urban': True, 'forest_cover': 0.15},
    'dhangadhi': {'lat': 28.6833, 'lon': 80.6000, 'address': 'Dhangadhi, Sudurpashchim Province', 'zone': 'terai', 'urban': True, 'forest_cover': 0.20},
    'mahendranagar': {'lat': 28.9667, 'lon': 80.1833, 'address': 'Mahendranagar, Sudurpashchim Province', 'zone': 'terai', 'urban': True, 'forest_cover': 0.25},
    'itahari': {'lat': 26.6667, 'lon': 87.2833, 'address': 'Itahari, Province No. 1', 'zone': 'terai', 'urban': True, 'forest_cover': 0.12},
    'kirtipur': {'lat': 27.6667, 'lon': 85.2833, 'address': 'Kirtipur, Bagmati Province', 'zone': 'hills', 'urban': True, 'forest_cover': 0.18},
    'bhaktapur': {'lat': 27.6710, 'lon': 85.4298, 'address': 'Bhaktapur, Bagmati Province', 'zone': 'hills', 'urban': True, 'forest_cover': 0.22},
    'damak': {'lat': 26.6667, 'lon': 87.7000, 'address': 'Damak, Province No. 1', 'zone': 'terai', 'urban': False, 'forest_cover': 0.35},
    'tulsipur': {'lat': 28.1333, 'lon': 82.3000, 'address': 'Tulsipur, Lumbini Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.40},
    'siddharthanagar': {'lat': 27.5000, 'lon': 83.4500, 'address': 'Siddharthanagar, Lumbini Province', 'zone': 'terai', 'urban': True, 'forest_cover': 0.15},
    'gorkha': {'lat': 28.0000, 'lon': 84.6333, 'address': 'Gorkha, Gandaki Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.55},
    'dolakha': {'lat': 27.7833, 'lon': 86.0667, 'address': 'Dolakha, Bagmati Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.50},
    'bardiya': {'lat': 28.3500, 'lon': 81.5000, 'address': 'Bardiya, Lumbini Province', 'zone': 'terai', 'urban': False, 'forest_cover': 0.60},
    'solukhumbu': {'lat': 27.7903, 'lon': 86.7139, 'address': 'Solukhumbu, Province No. 1', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.45},
    'mustang': {'lat': 28.9985, 'lon': 83.8473, 'address': 'Mustang, Gandaki Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.10},
    'manang': {'lat': 28.6660, 'lon': 84.0167, 'address': 'Manang, Gandaki Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.15},
    'jumla': {'lat': 29.2747, 'lon': 82.1838, 'address': 'Jumla, Karnali Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.35},
    'humla': {'lat': 29.9667, 'lon': 81.8833, 'address': 'Humla, Karnali Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.25},
    'dang': {'lat': 28.1167, 'lon': 82.3000, 'address': 'Dang, Lumbini Province', 'zone': 'terai', 'urban': False, 'forest_cover': 0.50},
    'surkhet': {'lat': 28.6000, 'lon': 81.6167, 'address': 'Surkhet, Karnali Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.55},
    'kaski': {'lat': 28.2639, 'lon': 83.9721, 'address': 'Kaski, Gandaki Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.48},
    'parsa': {'lat': 27.1333, 'lon': 84.8167, 'address': 'Parsa, Province No. 2', 'zone': 'terai', 'urban': False, 'forest_cover': 0.45},
    'nawalpur': {'lat': 27.6500, 'lon': 84.1167, 'address': 'Nawalpur, Gandaki Province', 'zone': 'terai', 'urban': False, 'forest_cover': 0.40},
    'rupandehi': {'lat': 27.5000, 'lon': 83.4500, 'address': 'Rupandehi, Lumbini Province', 'zone': 'terai', 'urban': True, 'forest_cover': 0.18},
    'kapilvastu': {'lat': 27.5500, 'lon': 83.0500, 'address': 'Kapilvastu, Lumbini Province', 'zone': 'terai', 'urban': False, 'forest_cover': 0.25},
    'palpa': {'lat': 27.8667, 'lon': 83.5500, 'address': 'Palpa, Lumbini Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.52},
    'syangja': {'lat': 28.1000, 'lon': 83.8833, 'address': 'Syangja, Gandaki Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.48},
    'tanahu': {'lat': 27.9333, 'lon': 84.2167, 'address': 'Tanahu, Gandaki Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.52},
    'lamjung': {'lat': 28.2833, 'lon': 84.3667, 'address': 'Lamjung, Gandaki Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.58},
    'baglung': {'lat': 28.2667, 'lon': 83.5833, 'address': 'Baglung, Gandaki Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.55},
    'myagdi': {'lat': 28.5000, 'lon': 83.5000, 'address': 'Myagdi, Gandaki Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.60},
    'parbat': {'lat': 28.3833, 'lon': 83.6833, 'address': 'Parbat, Gandaki Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.55},
    'makwanpur': {'lat': 27.4167, 'lon': 85.0333, 'address': 'Makwanpur, Bagmati Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.58},
    'sindhuli': {'lat': 27.2500, 'lon': 85.9667, 'address': 'Sindhuli, Bagmati Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.55},
    'ramechhap': {'lat': 27.5333, 'lon': 86.0833, 'address': 'Ramechhap, Bagmati Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.52},
    'kavrepalanchok': {'lat': 27.5333, 'lon': 85.5500, 'address': 'Kavrepalanchok, Bagmati Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.48},
    'sindhupalchok': {'lat': 27.9500, 'lon': 85.7000, 'address': 'Sindhupalchok, Bagmati Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.55},
    'nuwakot': {'lat': 27.9167, 'lon': 85.1667, 'address': 'Nuwakot, Bagmati Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.50},
    'rasuwa': {'lat': 28.1167, 'lon': 85.2833, 'address': 'Rasuwa, Bagmati Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.45},
    'dhading': {'lat': 27.8667, 'lon': 84.9167, 'address': 'Dhading, Bagmati Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.52},
    'banke': {'lat': 28.0500, 'lon': 81.6167, 'address': 'Banke, Lumbini Province', 'zone': 'terai', 'urban': False, 'forest_cover': 0.42},
    'kailali': {'lat': 28.6833, 'lon': 80.6000, 'address': 'Kailali, Sudurpashchim Province', 'zone': 'terai', 'urban': False, 'forest_cover': 0.38},
    'kanchanpur': {'lat': 28.9667, 'lon': 80.1833, 'address': 'Kanchanpur, Sudurpashchim Province', 'zone': 'terai', 'urban': False, 'forest_cover': 0.35},
    'doti': {'lat': 29.2500, 'lon': 80.9500, 'address': 'Doti, Sudurpashchim Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.55},
    'achham': {'lat': 29.0500, 'lon': 81.2500, 'address': 'Achham, Sudurpashchim Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.50},
    'bajhang': {'lat': 29.5333, 'lon': 81.1833, 'address': 'Bajhang, Sudurpashchim Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.40},
    'bajura': {'lat': 29.4500, 'lon': 81.4833, 'address': 'Bajura, Sudurpashchim Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.42},
    'dadeldhura': {'lat': 29.3000, 'lon': 80.5833, 'address': 'Dadeldhura, Sudurpashchim Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.48},
    'baitadi': {'lat': 29.5167, 'lon': 80.4167, 'address': 'Baitadi, Sudurpashchim Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.52},
    'darchula': {'lat': 29.8500, 'lon': 80.5500, 'address': 'Darchula, Sudurpashchim Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.38},
    'dailekh': {'lat': 28.8500, 'lon': 81.7167, 'address': 'Dailekh, Karnali Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.48},
    'jajarkot': {'lat': 28.7000, 'lon': 82.1833, 'address': 'Jajarkot, Karnali Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.55},
    'dolpa': {'lat': 29.0000, 'lon': 82.8667, 'address': 'Dolpa, Karnali Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.30},
    'mugu': {'lat': 29.5167, 'lon': 82.1000, 'address': 'Mugu, Karnali Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.32},
    'kalikot': {'lat': 29.1333, 'lon': 81.6167, 'address': 'Kalikot, Karnali Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.45},
    'rukum': {'lat': 28.6167, 'lon': 82.6167, 'address': 'Rukum, Karnali Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.52},
    'salyan': {'lat': 28.3833, 'lon': 82.1500, 'address': 'Salyan, Karnali Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.48},
    'rolpa': {'lat': 28.4333, 'lon': 82.6500, 'address': 'Rolpa, Lumbini Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.55},
    'pyuthan': {'lat': 28.1000, 'lon': 82.8667, 'address': 'Pyuthan, Lumbini Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.50},
    'arghakhanchi': {'lat': 27.9500, 'lon': 83.1333, 'address': 'Arghakhanchi, Lumbini Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.48},
    'gulmi': {'lat': 28.0833, 'lon': 83.2667, 'address': 'Gulmi, Lumbini Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.52},
    'ilam': {'lat': 26.9167, 'lon': 87.9333, 'address': 'Ilam, Province No. 1', 'zone': 'hills', 'urban': False, 'forest_cover': 0.58},
    'jhapa': {'lat': 26.6333, 'lon': 87.8833, 'address': 'Jhapa, Province No. 1', 'zone': 'terai', 'urban': False, 'forest_cover': 0.25},
    'morang': {'lat': 26.4525, 'lon': 87.2718, 'address': 'Morang, Province No. 1', 'zone': 'terai', 'urban': True, 'forest_cover': 0.18},
    'sunsari': {'lat': 26.6667, 'lon': 87.2833, 'address': 'Sunsari, Province No. 1', 'zone': 'terai', 'urban': True, 'forest_cover': 0.20},
    'saptari': {'lat': 26.6333, 'lon': 86.7333, 'address': 'Saptari, Province No. 2', 'zone': 'terai', 'urban': False, 'forest_cover': 0.15},
    'siraha': {'lat': 26.6500, 'lon': 86.2000, 'address': 'Siraha, Province No. 2', 'zone': 'terai', 'urban': False, 'forest_cover': 0.12},
    'dhanusha': {'lat': 26.7271, 'lon': 85.9407, 'address': 'Dhanusha, Province No. 2', 'zone': 'terai', 'urban': True, 'forest_cover': 0.08},
    'mahottari': {'lat': 26.8333, 'lon': 85.7833, 'address': 'Mahottari, Province No. 2', 'zone': 'terai', 'urban': False, 'forest_cover': 0.10},
    'sarlahi': {'lat': 27.0000, 'lon': 85.5667, 'address': 'Sarlahi, Province No. 2', 'zone': 'terai', 'urban': False, 'forest_cover': 0.15},
    'rautahat': {'lat': 27.0833, 'lon': 85.3000, 'address': 'Rautahat, Province No. 2', 'zone': 'terai', 'urban': False, 'forest_cover': 0.12},
    'bara': {'lat': 27.0667, 'lon': 85.0500, 'address': 'Bara, Province No. 2', 'zone': 'terai', 'urban': False, 'forest_cover': 0.20},
    'everest': {'lat': 27.9881, 'lon': 86.9250, 'address': 'Mount Everest Region, Province No. 1', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.25},
    'annapurna': {'lat': 28.5960, 'lon': 83.8203, 'address': 'Annapurna Region, Gandaki Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.35},
    'langtang': {'lat': 28.2133, 'lon': 85.5158, 'address': 'Langtang, Bagmati Province', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.40},
    'lumbini': {'lat': 27.4833, 'lon': 83.2833, 'address': 'Lumbini, Lumbini Province', 'zone': 'terai', 'urban': False, 'forest_cover': 0.20},
    'nagarkot': {'lat': 27.7172, 'lon': 85.5200, 'address': 'Nagarkot, Bagmati Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.55},
    'bandipur': {'lat': 27.9333, 'lon': 84.4167, 'address': 'Bandipur, Gandaki Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.50},
    'tansen': {'lat': 27.8667, 'lon': 83.5500, 'address': 'Tansen, Lumbini Province', 'zone': 'hills', 'urban': False, 'forest_cover': 0.52},
    'namche': {'lat': 27.8069, 'lon': 86.7140, 'address': 'Namche Bazaar, Province No. 1', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.35},
    'lukla': {'lat': 27.6869, 'lon': 86.7314, 'address': 'Lukla, Province No. 1', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.40},
    'taplejung': {'lat': 27.3500, 'lon': 87.6667, 'address': 'Taplejung, Province No. 1', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.55},
    'panchthar': {'lat': 27.1333, 'lon': 87.7833, 'address': 'Panchthar, Province No. 1', 'zone': 'hills', 'urban': False, 'forest_cover': 0.58},
    'terhathum': {'lat': 27.1333, 'lon': 87.5500, 'address': 'Terhathum, Province No. 1', 'zone': 'hills', 'urban': False, 'forest_cover': 0.55},
    'sankhuwasabha': {'lat': 27.3667, 'lon': 87.2333, 'address': 'Sankhuwasabha, Province No. 1', 'zone': 'himalaya', 'urban': False, 'forest_cover': 0.52},
    'bhojpur': {'lat': 27.1833, 'lon': 87.0500, 'address': 'Bhojpur, Province No. 1', 'zone': 'hills', 'urban': False, 'forest_cover': 0.50},
    'dhankuta': {'lat': 26.9833, 'lon': 87.3500, 'address': 'Dhankuta, Province No. 1', 'zone': 'hills', 'urban': False, 'forest_cover': 0.48},
    'khotang': {'lat': 27.0167, 'lon': 86.8500, 'address': 'Khotang, Province No. 1', 'zone': 'hills', 'urban': False, 'forest_cover': 0.55},
    'okhaldhunga': {'lat': 27.3167, 'lon': 86.5000, 'address': 'Okhaldhunga, Province No. 1', 'zone': 'hills', 'urban': False, 'forest_cover': 0.52},
    'udayapur': {'lat': 26.9333, 'lon': 86.5167, 'address': 'Udayapur, Province No. 1', 'zone': 'hills', 'urban': False, 'forest_cover': 0.45},
}

# ======================= HELPER FUNCTIONS =======================
def get_nepal_region(latitude):
    """Categorize location into Nepal's ecological zones"""
    if latitude < 27.5:
        return "Terai Plains", "terai", "🌾"
    elif latitude < 28.5:
        return "Mahabharat Range (Hills)", "hills", "⛰️"
    else:
        return "High Himalayas", "himalayas", "🏔️"

def get_region_info(region_name):
    """Get detailed info about a region"""
    info = {
        "Terai Plains": {
            "elevation": "Below 500m",
            "primary_cause": "Agricultural Residue Burning",
            "peak_season": "March - May",
            "characteristics": ["Agricultural burning", "Dense population", "Hot, dry conditions", "Grassland ecosystems"],
            "color": "#2ECC71"
        },
        "Mahabharat Range (Hills)": {
            "elevation": "500m - 3,000m",
            "primary_cause": "Dry Pine Needle Accumulation",
            "peak_season": "April - June",
            "characteristics": ["Chir pine forests", "Flammable needles", "Steep terrain", "Limited access"],
            "color": "#f39c12"
        },
        "High Himalayas": {
            "elevation": "Above 3,000m",
            "primary_cause": "Natural Dry Shrubs & High Winds",
            "peak_season": "March - May",
            "characteristics": ["Alpine shrubs", "Strong winds", "Remote terrain", "Limited firefighting"],
            "color": "#3498db"
        }
    }
    return info.get(region_name, info["Mahabharat Range (Hills)"])

def get_grid_key(latitude, longitude):
    """Get grid key for a location"""
    grid_lat = int(latitude / 0.25) * 0.25
    grid_lon = int(longitude / 0.25) * 0.25
    return f"{grid_lat}_{grid_lon}"

def get_location_specific_stats(latitude, longitude, radius_km=50):
    """Get actual fire statistics for a specific location from NASA FIRMS data"""
    global fire_data, location_stats_cache
    
    if fire_data is None:
        return None
    
    # Try grid-based cache first
    grid_key = get_grid_key(latitude, longitude)
    if grid_key in location_stats_cache:
        cached = location_stats_cache[grid_key]
        return {
            'fire_count': cached['fire_count'],
            'avg_brightness': cached['avg_brightness'],
            'max_brightness': cached['max_brightness'],
            'avg_frp': cached['avg_frp'],
            'max_frp': cached['max_frp'],
            'years_with_fires': cached['years_with_fires'],
            'primary_cause': cached['primary_cause'],
            'fire_density': cached['fire_count'] / 625  # fires per ~625 sq km grid (25x25)
        }
    
    # Fall back to radius-based search
    lat_range = radius_km / 111.0
    lng_range = radius_km / 85.0
    
    nearby = fire_data[
        (fire_data['latitude'] >= latitude - lat_range) &
        (fire_data['latitude'] <= latitude + lat_range) &
        (fire_data['longitude'] >= longitude - lng_range) &
        (fire_data['longitude'] <= longitude + lng_range)
    ]
    
    if len(nearby) == 0:
        return None
    
    area = np.pi * radius_km * radius_km
    
    return {
        'fire_count': len(nearby),
        'avg_brightness': nearby['brightness'].mean(),
        'max_brightness': nearby['brightness'].max(),
        'avg_frp': nearby['frp'].mean() if 'frp' in nearby.columns else 5.0,
        'max_frp': nearby['frp'].max() if 'frp' in nearby.columns else 15.0,
        'years_with_fires': nearby['year'].nunique() if 'year' in nearby.columns else 3,
        'primary_cause': nearby['likely_cause'].mode().iloc[0] if 'likely_cause' in nearby.columns and len(nearby) > 0 else 'Unknown',
        'fire_density': len(nearby) / area * 100  # fires per 100 sq km
    }

def calculate_risk_score(latitude, longitude, location_stats, district_info=None):
    """Calculate accurate risk score based on actual data"""
    base_score = 0.30  # Base risk level
    
    if location_stats is None:
        # No fire history - lower risk
        return 0.20, "Low", "No significant fire history detected in this area."
    
    # Factor 1: Fire frequency (0-0.25)
    fire_count = location_stats.get('fire_count', 0)
    if fire_count > 1000:
        frequency_score = 0.25
    elif fire_count > 500:
        frequency_score = 0.20
    elif fire_count > 200:
        frequency_score = 0.15
    elif fire_count > 100:
        frequency_score = 0.10
    elif fire_count > 50:
        frequency_score = 0.07
    else:
        frequency_score = 0.03
    
    # Factor 2: Fire intensity (avg brightness) (0-0.20)
    avg_brightness = location_stats.get('avg_brightness', 330)
    if avg_brightness > 350:
        intensity_score = 0.20
    elif avg_brightness > 342:
        intensity_score = 0.15
    elif avg_brightness > 335:
        intensity_score = 0.10
    elif avg_brightness > 330:
        intensity_score = 0.06
    else:
        intensity_score = 0.03
    
    # Factor 3: Fire radiative power (0-0.15)
    avg_frp = location_stats.get('avg_frp', 5)
    if avg_frp > 15:
        frp_score = 0.15
    elif avg_frp > 10:
        frp_score = 0.10
    elif avg_frp > 6:
        frp_score = 0.06
    else:
        frp_score = 0.03
    
    # Factor 4: Recurrence (years with fires) (0-0.15)
    years_with_fires = location_stats.get('years_with_fires', 1)
    if years_with_fires >= 5:
        recurrence_score = 0.15
    elif years_with_fires >= 4:
        recurrence_score = 0.12
    elif years_with_fires >= 3:
        recurrence_score = 0.08
    elif years_with_fires >= 2:
        recurrence_score = 0.05
    else:
        recurrence_score = 0.02
    
    # Factor 5: Seasonal adjustment (current month)
    current_month = datetime.now().month
    # Peak fire season in Nepal: March-May
    if current_month in [3, 4, 5]:
        seasonal_factor = 1.20
    elif current_month in [2, 6]:
        seasonal_factor = 1.08
    elif current_month in [11, 12, 1]:
        seasonal_factor = 0.88
    else:
        seasonal_factor = 0.82
    
    # Factor 6: Forest cover and urban adjustment
    forest_adjustment = 0
    if district_info:
        forest_cover = district_info.get('forest_cover', 0.3)
        if forest_cover > 0.55:
            forest_adjustment = 0.12
        elif forest_cover > 0.40:
            forest_adjustment = 0.08
        elif forest_cover > 0.25:
            forest_adjustment = 0.04
        
        if district_info.get('urban', False):
            forest_adjustment -= 0.10  # Urban areas have lower forest fire risk
    
    # Calculate total risk score
    raw_score = base_score + frequency_score + intensity_score + frp_score + recurrence_score + forest_adjustment
    final_score = min(0.95, max(0.08, raw_score * seasonal_factor))
    
    # Determine risk level
    if final_score >= 0.65:
        risk_level = "High"
    elif final_score >= 0.40:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    
    return final_score, risk_level, location_stats.get('primary_cause', 'Unknown')

def generate_risk_reason(risk_level, location_stats, region_name, latitude, longitude):
    """Generate detailed, location-specific risk reason"""
    
    fire_count = location_stats.get('fire_count', 0) if location_stats else 0
    avg_brightness = location_stats.get('avg_brightness', 330) if location_stats else 330
    primary_cause = location_stats.get('primary_cause', 'Unknown') if location_stats else 'Unknown'
    years_with_fires = location_stats.get('years_with_fires', 0) if location_stats else 0
    fire_density = location_stats.get('fire_density', 0) if location_stats else 0
    
    # Build location-specific reason
    reasons = []
    
    if fire_count > 0:
        reasons.append(f"Analysis of {fire_count:,} historical fire events from NASA VIIRS satellite data")
    
    if years_with_fires > 1:
        reasons.append(f"with fires recorded across {years_with_fires} different years")
    
    if fire_density > 2:
        reasons.append(f"shows high fire density ({fire_density:.1f} fires per 100 km²)")
    elif fire_density > 0.5:
        reasons.append(f"indicates moderate fire activity ({fire_density:.1f} fires per 100 km²)")
    
    if avg_brightness > 345:
        reasons.append("with high-intensity fire patterns detected")
    elif avg_brightness > 338:
        reasons.append("showing moderate intensity fire activity")
    
    if primary_cause and primary_cause != 'Unknown':
        reasons.append(f"Primary ignition source: {primary_cause}")
    
    # Add region-specific context
    region_context = {
        "Terai Plains": "Agricultural practices and dry seasonal conditions in the Terai region contribute to elevated fire risk during dry months.",
        "Mahabharat Range (Hills)": "Chir pine forests with accumulated dry needles create significant fuel load in the hill region, especially during pre-monsoon months.",
        "High Himalayas": "Alpine vegetation and strong winds during dry season combined with limited firefighting access create fire-prone conditions in the Himalayan zone."
    }
    
    if region_name in region_context:
        reasons.append(region_context[region_name])
    
    if not reasons:
        reasons.append("Risk assessment based on regional fire patterns and NASA satellite monitoring data.")
    
    return " ".join(reasons)

def predict_risk(latitude, longitude, region_name, district_info=None):
    """Predict fire risk for a location using actual NASA FIRMS data"""
    global model, encoder, fire_data
    
    # Get location-specific statistics from actual data
    location_stats = get_location_specific_stats(latitude, longitude, radius_km=30)
    
    # Calculate risk score based on actual data
    risk_score, risk_level, primary_cause = calculate_risk_score(
        latitude, longitude, location_stats, district_info
    )
    
    # Generate detailed reason
    reason = generate_risk_reason(risk_level, location_stats, region_name, latitude, longitude)
    
    # Try to use trained model for additional refinement
    if model is not None and encoder is not None and location_stats:
        try:
            # Get actual location-specific values for model prediction
            avg_brightness = location_stats.get('avg_brightness', 335)
            
            # Encode region
            region_encoded = encoder.transform([region_name])[0]
            
            features = np.array([[
                latitude,
                longitude,
                avg_brightness,
                0.40,  # scan
                0.37,  # track
                region_encoded
            ]])
            
            # Get model prediction probability
            proba = model.predict_proba(features)[0]
            model_confidence = max(proba)
            
            # Blend model prediction with data-driven score
            blended_score = (risk_score * 0.65) + (model_confidence * 0.35)
            risk_score = blended_score
            
        except Exception as e:
            print(f"Model prediction error: {e}")
    
    return risk_level, risk_score, reason

def get_nearby_fire_history(latitude, longitude, radius_km=50):
    """Get actual fire history near a specific location from NASA FIRMS data"""
    global fire_data
    
    if fire_data is None:
        return None
    
    try:
        lat_range = radius_km / 111.0
        lng_range = radius_km / 85.0
        
        nearby = fire_data[
            (fire_data['latitude'] >= latitude - lat_range) &
            (fire_data['latitude'] <= latitude + lat_range) &
            (fire_data['longitude'] >= longitude - lng_range) &
            (fire_data['longitude'] <= longitude + lng_range)
        ].copy()
        
        actual_radius = radius_km
        
        if len(nearby) == 0:
            # Expand search
            lat_range *= 2
            lng_range *= 2
            actual_radius = radius_km * 2
            nearby = fire_data[
                (fire_data['latitude'] >= latitude - lat_range) &
                (fire_data['latitude'] <= latitude + lat_range) &
                (fire_data['longitude'] >= longitude - lng_range) &
                (fire_data['longitude'] <= longitude + lng_range)
            ].copy()
        
        if len(nearby) == 0:
            return {
                'yearly_counts': {},
                'monthly_distribution': {},
                'recent_fires': [],
                'stats': {
                    'total_nearby_fires': 0,
                    'avg_brightness': 0,
                    'avg_frp': 0,
                    'primary_cause': 'No fire history detected',
                    'peak_month': 'N/A',
                    'fire_density_per_100sqkm': 0
                },
                'search_radius_km': actual_radius
            }
        
        # Count fires per year
        yearly_counts = {}
        if 'year' in nearby.columns:
            yearly_counts = nearby.groupby('year').size().to_dict()
            # Convert keys to int for JSON serialization
            yearly_counts = {int(k): int(v) for k, v in yearly_counts.items()}
        
        # Count fires per month (for seasonal pattern)
        monthly_counts = {}
        peak_month = 'April'
        if 'month' in nearby.columns:
            monthly_counts = nearby.groupby('month').size().to_dict()
            monthly_counts = {int(k): int(v) for k, v in monthly_counts.items()}
            if monthly_counts:
                peak_month_num = max(monthly_counts, key=monthly_counts.get)
                month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 
                               6: 'June', 7: 'July', 8: 'August', 9: 'September', 
                               10: 'October', 11: 'November', 12: 'December'}
                peak_month = month_names.get(peak_month_num, 'April')
        
        # Get recent fire events
        recent_fires = []
        if 'acq_date' in nearby.columns:
            recent_df = nearby.nlargest(5, 'acq_date')[
                ['latitude', 'longitude', 'brightness', 'acq_date', 'frp', 'likely_cause']
            ]
            for _, row in recent_df.iterrows():
                recent_fires.append({
                    'latitude': round(row['latitude'], 4),
                    'longitude': round(row['longitude'], 4),
                    'brightness': round(row['brightness'], 1),
                    'acq_date': row['acq_date'].strftime('%Y-%m-%d') if pd.notna(row['acq_date']) else 'Unknown',
                    'frp': round(row['frp'], 2) if pd.notna(row['frp']) else 0,
                    'likely_cause': row['likely_cause'] if pd.notna(row['likely_cause']) else 'Unknown'
                })
        
        # Calculate statistics
        area_sq_km = np.pi * actual_radius * actual_radius
        fire_density = (len(nearby) / area_sq_km) * 100  # per 100 sq km
        
        stats = {
            'total_nearby_fires': int(len(nearby)),
            'avg_brightness': round(nearby['brightness'].mean(), 1),
            'max_brightness': round(nearby['brightness'].max(), 1),
            'avg_frp': round(nearby['frp'].mean(), 2) if 'frp' in nearby.columns else 0,
            'max_frp': round(nearby['frp'].max(), 2) if 'frp' in nearby.columns else 0,
            'primary_cause': nearby['likely_cause'].mode().iloc[0] if 'likely_cause' in nearby.columns and len(nearby) > 0 else 'Unknown',
            'peak_month': peak_month,
            'fire_density_per_100sqkm': round(fire_density, 2),
            'years_analyzed': int(nearby['year'].nunique()) if 'year' in nearby.columns else 1,
            'first_record': str(int(nearby['year'].min())) if 'year' in nearby.columns else 'Unknown',
            'last_record': str(int(nearby['year'].max())) if 'year' in nearby.columns else 'Unknown'
        }
        
        return {
            'yearly_counts': yearly_counts,
            'monthly_distribution': monthly_counts,
            'recent_fires': recent_fires,
            'stats': stats,
            'search_radius_km': actual_radius
        }
        
    except Exception as e:
        print(f"Error getting fire history: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_ignition_probability(latitude, longitude, location_stats, region_name):
    """Calculate probability of fire ignition for a location"""
    
    if location_stats is None:
        return {
            'probability': 12,
            'level': 'Low',
            'factors': ['Insufficient historical data - area shows minimal fire activity']
        }
    
    base_prob = 15
    factors = []
    
    # Historical fire frequency factor
    fire_count = location_stats.get('fire_count', 0)
    fire_density = location_stats.get('fire_density', 0)
    
    if fire_density > 3:
        prob_adjustment = 28
        factors.append(f"High historical fire density: {fire_density:.1f} fires per 100 km²")
    elif fire_density > 1.5:
        prob_adjustment = 18
        factors.append(f"Moderate fire density: {fire_density:.1f} fires per 100 km²")
    elif fire_density > 0.5:
        prob_adjustment = 10
        factors.append(f"Low fire density: {fire_density:.1f} fires per 100 km²")
    else:
        prob_adjustment = 3
        factors.append("Minimal historical fire activity in this area")
    
    base_prob += prob_adjustment
    
    # Fire intensity factor
    avg_brightness = location_stats.get('avg_brightness', 330)
    if avg_brightness > 348:
        base_prob += 15
        factors.append(f"High intensity fires detected - avg brightness {avg_brightness:.0f}K")
    elif avg_brightness > 340:
        base_prob += 10
        factors.append(f"Moderate fire intensity - avg brightness {avg_brightness:.0f}K")
    elif avg_brightness > 333:
        base_prob += 5
        factors.append(f"Standard fire intensity patterns - {avg_brightness:.0f}K avg")
    
    # Recurrence factor
    years_with_fires = location_stats.get('years_with_fires', 0)
    if years_with_fires >= 5:
        base_prob += 12
        factors.append(f"Consistent annual fire recurrence ({years_with_fires} years)")
    elif years_with_fires >= 3:
        base_prob += 7
        factors.append(f"Regular fire pattern detected ({years_with_fires} years)")
    elif years_with_fires >= 2:
        base_prob += 4
        factors.append(f"Some recurrence detected ({years_with_fires} years)")
    
    # FRP factor
    avg_frp = location_stats.get('avg_frp', 5)
    if avg_frp > 12:
        base_prob += 8
        factors.append(f"High fire radiative power ({avg_frp:.1f} MW)")
    elif avg_frp > 7:
        base_prob += 4
        factors.append(f"Moderate fire energy release ({avg_frp:.1f} MW)")
    
    # Seasonal factor
    current_month = datetime.now().month
    if current_month in [3, 4, 5]:
        base_prob = int(base_prob * 1.25)
        factors.append("⚠️ Currently in peak fire season (March-May)")
    elif current_month in [2, 6]:
        base_prob = int(base_prob * 1.10)
        factors.append("Approaching/leaving fire season")
    elif current_month in [7, 8, 9]:
        base_prob = int(base_prob * 0.70)
        factors.append("Monsoon season - reduced fire risk")
    elif current_month in [11, 12, 1]:
        base_prob = int(base_prob * 0.85)
        factors.append("Post-monsoon/winter - moderate conditions")
    
    # Regional factor
    if region_name == "Terai Plains":
        base_prob += 6
        factors.append("Terai region: agricultural burning common")
    elif region_name == "Mahabharat Range (Hills)":
        base_prob += 10
        factors.append("Hill region: pine needle fuel accumulation")
    elif region_name == "High Himalayas":
        base_prob += 4
        factors.append("Alpine zone: dry shrubs and wind exposure")
    
    # Cap probability
    final_prob = min(92, max(5, base_prob))
    
    # Determine level
    if final_prob >= 55:
        level = 'High'
    elif final_prob >= 30:
        level = 'Moderate'
    else:
        level = 'Low'
    
    return {
        'probability': final_prob,
        'level': level,
        'factors': factors
    }

def get_location_heatmap(latitude, longitude, radius_km=75):
    """Get heatmap data for area around a specific location"""
    global fire_data
    
    if fire_data is None:
        return []
    
    try:
        lat_range = radius_km / 111.0
        lng_range = radius_km / 85.0
        
        nearby = fire_data[
            (fire_data['latitude'] >= latitude - lat_range) &
            (fire_data['latitude'] <= latitude + lat_range) &
            (fire_data['longitude'] >= longitude - lng_range) &
            (fire_data['longitude'] <= longitude + lng_range)
        ]
        
        if len(nearby) == 0:
            return []
        
        sample_size = min(3000, len(nearby))
        sample = nearby.sample(n=sample_size, random_state=42) if len(nearby) > sample_size else nearby
        
        max_brightness = sample['brightness'].max()
        min_brightness = sample['brightness'].min()
        brightness_range = max_brightness - min_brightness if max_brightness != min_brightness else 1
        
        result = []
        for _, row in sample.iterrows():
            intensity = (row['brightness'] - min_brightness) / brightness_range
            result.append({
                'lat': round(row['latitude'], 4),
                'lng': round(row['longitude'], 4),
                'intensity': round(intensity, 2)
            })
        
        return result
    except Exception as e:
        print(f"Error getting location heatmap: {e}")
        return []

def geocode_location(location_name):
    """Geocode a location in Nepal with fallback to local database"""
    search_term = location_name.lower().strip().replace(' ', '')
    
    # Check district database first
    for key, data in NEPAL_DISTRICTS.items():
        if key in search_term or search_term in key:
            return {
                'latitude': data['lat'],
                'longitude': data['lon'],
                'address': data['address'],
                'district_info': data
            }
    
    # Try partial matches
    for key, data in NEPAL_DISTRICTS.items():
        if any(part in key for part in search_term.split()) or any(part in search_term for part in key.split()):
            return {
                'latitude': data['lat'],
                'longitude': data['lon'],
                'address': data['address'],
                'district_info': data
            }
    
    # Try Nominatim
    try:
        geolocator = Nominatim(user_agent="forestsathi_wildfire_intelligence_nepal")
        search_query = f"{location_name}, Nepal"
        location = geolocator.geocode(search_query, timeout=15)
        
        if location:
            if 26.3 <= location.latitude <= 30.5 and 80.0 <= location.longitude <= 88.2:
                return {
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'address': location.address,
                    'district_info': None
                }
        return None
    except (GeocoderTimedOut, GeocoderServiceError, Exception):
        return None

# ======================= ROUTES =======================
@app.route('/')
def home():
    """Main page"""
    stats = {}
    if fire_data is not None:
        stats = {
            'total_fires': len(fire_data),
            'terai_fires': len(fire_data[fire_data['nepal_region'].str.contains('Terai', na=False)]),
            'hills_fires': len(fire_data[fire_data['nepal_region'].str.contains('Mahabharat', na=False)]),
            'himalaya_fires': len(fire_data[fire_data['nepal_region'].str.contains('Himalayas', na=False)]),
            'model_loaded': model is not None,
            'data_period': f"{int(fire_data['year'].min())}-{int(fire_data['year'].max())}" if 'year' in fire_data.columns else 'Unknown'
        }
    return render_template('index.html', stats=stats)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/search', methods=['POST'])
def search_location():
    """API endpoint for location search"""
    data = request.get_json()
    location_name = data.get('location', '')
    
    if not location_name:
        return jsonify({'error': 'Please enter a location name'}), 400
    
    # Geocode
    location_data = geocode_location(location_name)
    
    if not location_data:
        return jsonify({'error': 'Location not found in Nepal. Try a different search term.'}), 404
    
    lat = location_data['latitude']
    lon = location_data['longitude']
    district_info = location_data.get('district_info')
    
    # Get region
    region_name, region_class, region_emoji = get_nepal_region(lat)
    region_info = get_region_info(region_name)
    
    # Get location-specific statistics
    location_stats = get_location_specific_stats(lat, lon, radius_km=30)
    
    # Predict risk using actual data
    risk_level, confidence, reason = predict_risk(lat, lon, region_name, district_info)
    
    # Calculate ignition probability
    ignition_data = calculate_ignition_probability(lat, lon, location_stats, region_name)
    
    # Get actual fire history
    fire_history = get_nearby_fire_history(lat, lon, radius_km=50)
    
    # Get location-specific heatmap data
    location_heatmap = get_location_heatmap(lat, lon, radius_km=75)
    
    return jsonify({
        'success': True,
        'location': {
            'name': location_name.title(),
            'address': location_data['address'],
            'latitude': lat,
            'longitude': lon
        },
        'region': {
            'name': region_name,
            'class': region_class,
            'emoji': region_emoji,
            'info': region_info
        },
        'prediction': {
            'risk_level': risk_level,
            'confidence': round(confidence * 100, 1),
            'reason': reason
        },
        'ignition': ignition_data,
        'fire_history': fire_history,
        'location_heatmap': location_heatmap
    })

@app.route('/api/heatmap')
def get_heatmap_data():
    """API endpoint for heatmap data"""
    return jsonify(heatmap_data)

@app.route('/api/stats')
def get_stats():
    """API endpoint for statistics"""
    if fire_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    stats = {
        'total_fires': int(len(fire_data)),
        'data_period': f"{int(fire_data['year'].min())}-{int(fire_data['year'].max())}" if 'year' in fire_data.columns else 'Unknown',
        'regions': {}
    }
    
    for region in ['Terai Plains', 'Mahabharat Range (Hills)', 'High Himalayas']:
        region_data = fire_data[fire_data['nepal_region'].str.contains(region.split()[0], na=False)]
        if len(region_data) > 0:
            stats['regions'][region] = {
                'count': int(len(region_data)),
                'avg_brightness': round(region_data['brightness'].mean(), 1),
                'avg_frp': round(region_data['frp'].mean(), 2),
                'percentage': round(len(region_data) / len(fire_data) * 100, 1)
            }
    
    return jsonify(stats)

# ======================= ERROR HANDLERS =======================
@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ======================= MAIN =======================
if __name__ == '__main__':
    print("🌲 ForestSathi - Loading resources...")
    load_resources()
    print(f"   📊 Fire data: {'Loaded' if fire_data is not None else 'Not found'}")
    if fire_data is not None:
        print(f"   📈 Total records: {len(fire_data):,}")
        print(f"   📅 Data period: {int(fire_data['year'].min())}-{int(fire_data['year'].max())}")
        print(f"   🗺️ Grid cells cached: {len(location_stats_cache):,}")
    print(f"   🤖 Model: {'Loaded' if model is not None else 'Using data-driven fallback'}")
    print(f"   🔥 Heatmap points: {len(heatmap_data):,}")
    print("\n🚀 Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
