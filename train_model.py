"""
🌲 ForestSathi Model Training Script
Trains a Random Forest model from the CSV data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_forestsathi_model():
    print("🌲 ForestSathi Model Training")
    print("=" * 50)
    
    # Load data
    print("\n📊 Loading training data...")
    df = pd.read_csv('forestsathi_training_data.csv')
    print(f"   Loaded {len(df):,} records")
    
    # Clean data
    print("\n🧹 Cleaning data...")
    df = df.dropna(subset=['latitude', 'longitude', 'brightness', 'nepal_region'])
    print(f"   {len(df):,} records after cleaning")
    
    # Feature engineering
    print("\n⚙️ Engineering features...")
    
    # Encode regions
    region_encoder = LabelEncoder()
    df['region_encoded'] = region_encoder.fit_transform(df['nepal_region'])
    
    # Create risk labels based on brightness and FRP
    # High brightness (>340) and high FRP indicate more intense fires
    df['risk_level'] = np.where(
        (df['brightness'] > 340) | (df['frp'] > 10),
        1,  # High risk
        0   # Lower risk
    )
    
    # Select features
    features = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'region_encoded']
    X = df[features].values
    y = df['risk_level'].values
    
    print(f"   Features: {features}")
    print(f"   High risk samples: {sum(y):,} ({sum(y)/len(y)*100:.1f}%)")
    print(f"   Low risk samples: {len(y)-sum(y):,} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    # Split data
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Testing samples: {len(X_test):,}")
    
    # Train model
    print("\n🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy*100:.2f}%")
    
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    # Feature importance
    print("\n🎯 Feature Importance:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"   {feat}: {imp*100:.1f}%")
    
    # Save model and encoder
    print("\n💾 Saving model and encoder...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(region_encoder, f)
    
    # Save region statistics for the app
    region_stats = df.groupby('nepal_region').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'brightness': 'mean',
        'frp': 'mean',
        'risk_level': 'mean'
    }).to_dict('index')
    
    with open('region_stats.pkl', 'wb') as f:
        pickle.dump(region_stats, f)
    
    print("   ✅ model.pkl saved")
    print("   ✅ encoder.pkl saved")
    print("   ✅ region_stats.pkl saved")
    
    print("\n" + "=" * 50)
    print("🎉 Training complete! Model accuracy: {:.2f}%".format(accuracy * 100))
    
    return model, region_encoder, accuracy

if __name__ == "__main__":
    train_forestsathi_model()
