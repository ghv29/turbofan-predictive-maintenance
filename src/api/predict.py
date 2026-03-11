import pickle
import numpy as np
import pandas as pd 
import os 

# ==============================================#
# Loading model once when API starts            #
# ==============================================#
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
           os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, 'outputs', 'models', 
                          'rf_fd001_capped.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'outputs', 'models', 
                              'feature_cols.pkl')

print(f"Looking for model at: {MODEL_PATH}")
try:
    with open(MODEL_PATH, 'rb') as f:
        model =pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        FEATURE_COLS = pickle.load(f)
    print("Model loaded successfully")
    print(f"Expecting {len(FEATURE_COLS)} features")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    FEATURE_COLS = None



# ===============================================
# Health status
# ==============================================

def get_health_status(rul):
    if rul <= 30:
        return {
            'status' : 'CRITICAL',
            'color' : 'red',
            'message': 'Schedule maintenance immediately',
            'urgency': 'Immediate action required'
        }
    elif rul<=60:
        return {
            'status' : 'WARNING',
            'color' : 'orange',
            'message': 'Schedule maintenance this week',
            'urgency': 'Action required within 7 days'
        }
    elif rul <=90:
        return{
            'status': 'MONITOR',
            'color' : 'yellow',
            'message': 'Monitor closely',
            'urgency': 'Review within 30 days'
        }
    else: 
        return{
            'status': 'HEALTHY',
            'color' : 'green',
            'message': 'Engine operating normally',
            'urgency': 'No action required'
        }

# ================================================
# Main prediction function
# ================================================

def predict_rul(sensor_data: dict) -> dict:
    try:
        # Build feature dictionary
        features = {}
        
        for col in FEATURE_COLS:
            if col in sensor_data:
                # Direct match — use provided value
                features[col] = sensor_data[col]
            elif '_rollmean' in col:
                # Rolling mean not provided
                # Use raw sensor value as approximation
                base_sensor = col.replace('_rollmean', '')
                features[col] = sensor_data.get(base_sensor, 0)
            elif '_rollstd' in col:
                # Rolling std not provided
                # Single reading → no variation → 0
                features[col] = 0
            else:
                # Any other missing feature → 0
                features[col] = 0

        #converting to DataFrame
        df = pd.DataFrame([features])[FEATURE_COLS]

        #predict
        predicted_rul = float(model.predict(df)[0])
        predicted_rul = max(0, round(predicted_rul, 1))

        #Get health status
        health = get_health_status(predicted_rul)

        return{
            'success': True,
            'predicted_RUL': predicted_rul,
            'health_status':health['status'],
            'color': health['color'],
            'message': health['message'],
            'urgency': health['urgency']
        }
    except Exception as e:
        return{
            'success': False,
            'error': str(e)
        }