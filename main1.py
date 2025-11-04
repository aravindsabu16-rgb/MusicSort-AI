import os
import tempfile
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =================== Files ===================
MODEL_FILE = "trained_model.pkl"
ENCODER_FILE = "label_encoder.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
CSV_FILE = "features_3_sec.csv"

# ================== GENRE MAPPING ==================
GENRE_MAPPING = {
    'hiphop': 'hiphop', 'rap': 'hiphop', 'trap': 'hiphop',
    'rock': 'rock', 'metal': 'rock', 'punk': 'rock', 'alternative': 'rock',
    'pop': 'pop', 'disco': 'pop', 'dance': 'pop',
    'jazz': 'jazz', 'blues': 'jazz', 'swing': 'jazz',
    'classical': 'classical', 'opera': 'classical',
    'electronic': 'electronic', 'edm': 'electronic', 'techno': 'electronic', 'house': 'electronic',
    'country': 'country', 'folk': 'country', 'bluegrass': 'country',
    'reggae': 'reggae', 'dub': 'reggae',
    'r&b': 'rnb', 'soul': 'rnb', 'funk': 'rnb'
}

# ================== IMPROVED TRAIN MODEL ==================
def train_model():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Dataset file not found: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    
    # Map to broader genres
    df["broad_genre"] = df["label"].map(lambda x: GENRE_MAPPING.get(x.lower(), 'unknown'))
    df = df[df["broad_genre"] != 'unknown']
    
    if df.empty:
        print("❌ No valid genres found after mapping")
        return
    
    le = LabelEncoder()
    df["encoded_label"] = le.fit_transform(df["broad_genre"])

    # Use available features
    available_features = []
    feature_candidates = ["tempo", "zero_crossing_rate_mean", "rms_mean",
                         "spectral_centroid_mean", "rolloff_mean", "mfcc1_mean"]
    
    for feature in feature_candidates:
        if feature in df.columns:
            available_features.append(feature)
    
    if not available_features:
        print("❌ No matching features found in dataset")
        return
        
    print(f"✅ Using features: {available_features}")
    
    X = df[available_features]
    y = df["encoded_label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation for better accuracy estimate
    cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    
    print("✅ Training Complete!")
    print(f"📊 Test Accuracy: {accuracy:.2%}")
    print(f"📊 Cross-validation Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
    
    # Show genre distribution
    print("\n🎵 Genre Distribution:")
    genre_counts = df["broad_genre"].value_counts()
    for genre, count in genre_counts.items():
        print(f"   {genre}: {count} samples")

    # Save model and components
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, ENCODER_FILE)
    joblib.dump(available_features, FEATURE_NAMES_FILE)
    
    print("\n✅ Model saved - Ready for predictions!")
    print("💡 You can now use this model to predict ANY song")

# =========== AUDIO FEATURE EXTRACTION ===========
def extract_features(file_path, feature_names):
    try:
        y, sr = librosa.load(file_path, duration=30)
        
        features = []
        
        for feature_name in feature_names:
            if feature_name == "tempo":
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features.append(float(tempo) if not np.isnan(tempo) and tempo > 0 else 120.0)
            elif feature_name == "zero_crossing_rate_mean":
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                features.append(float(zcr))
            elif feature_name == "rms_mean":
                rms = np.mean(librosa.feature.rms(y=y))
                features.append(float(rms))
            elif feature_name == "spectral_centroid_mean":
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                features.append(float(spectral_centroid))
            elif feature_name == "rolloff_mean":
                spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                features.append(float(spectral_rolloff))
            elif feature_name == "mfcc1_mean":
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc1 = np.mean(mfccs[0])
                features.append(float(mfcc1))
        
        return np.array(features, dtype=float).reshape(1, -1)
        
    except Exception as e:
        print("❌ Feature extraction error:", e)
        return None

# ================== PREDICT GENRE ==================
def predict_song(path):
    if not os.path.exists(MODEL_FILE):
        print("⚠️ Please train model first (select 'T')")
        return

    model = joblib.load(MODEL_FILE)
    le = joblib.load(ENCODER_FILE)
    available_features = joblib.load(FEATURE_NAMES_FILE)

    feats = extract_features(path, available_features)
    if feats is None:
        return
    
    feats_df = pd.DataFrame(feats, columns=available_features)
    pred = model.predict(feats_df)[0]
    genre = le.inverse_transform([pred])[0]

    print(f"\n🎵 Predicted Genre: {genre.upper()}")
    
    probabilities = model.predict_proba(feats_df)[0]
    genres = le.classes_
    
    print("\n📊 Confidence Scores:")
    for genre_name, prob in zip(genres, probabilities):
        print(f"   {genre_name.upper()}: {prob:.2%}")

# ===================== CLI ===========================
if __name__ == "__main__":
    print("🎵 Music Genre Classifier")
    print("=" * 30)
    
    print("\n💡 Instructions:")
    print("   • Train once with 'T'")
    print("   • Predict many songs with 'P'")
    print("   • No need to retrain for each prediction!")
    
    mode = input("\nTrain (T) or Predict (P)? ").strip().lower()

    if mode == "t":
        print("\n🚀 Training model (one time process)...")
        train_model()

    elif mode == "p":
        path = input("\nEnter audio file path: ").strip().strip('"').strip("'")
        
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
        else:
            print("\n🔍 Analyzing song...")
            predict_song(path)

    else:
        print("❌ Invalid choice")
        