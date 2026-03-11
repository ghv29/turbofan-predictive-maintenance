import numpy as np 
from flask import Flask, request, jsonify
from predict import predict_rul
from datetime import datetime

# =======================================
# Initialize Flask app
# =======================================

app = Flask(__name__)

#Model performance metrics for /model-info 

MODEL_METRICS = { 
    'model_type': 'Random Forest (GridSearchCV optimized)',
    'dataset': 'NASA CMAPSS FD001', 
    'n_estimators': 200,
    'RUL_cap': 125,
    'metrics': {
        'RMSE': 15.51,
        'MAE' : 10.89,
        'R2' : 0.858
    },
    'features': 44,
    'training': 16504
}

# ======================================
# Endpoint 1: Health Check
# GET /health
# ======================================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'API is running',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_loaded': True
    }), 200

# ======================================
# Endpoint 2: Model Information
# Get /model-info
# ======================================

@app.route('/model-info', methods=["GET"])
def model_info():
    return jsonify(MODEL_METRICS), 200

# ======================================
# Endpoint 3: Predict RUL
# POST /predict
# ======================================

@app.route('/predict', methods=["POST"])
def predict():
    #Get JSON data from request
    data = request.get_json()

    #Validate input
    if not data:
        return jsonify({
            'success' : False,
            'error': 'No data provided'
        }), 400
    
    #Required Sensors
    required = ['s2', 's3', 's4', 's7', 's8', 
                's9', 's11', 's12', 's13', 's14', 
                's15', 's17', 's20', 's21']
    
    missing = [s for s in required if s not in data]
    if missing:
        return jsonify({
            'success': False,
            'error': f'Missing sensors: {missing}'
        }), 400
    
    #Making prediction
    result = predict_rul(data)

    # Adding metadata
    result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result['engine_id'] = data.get('engine_id', 'unknown')
    result['cycle'] = data.get('cycle', 'unknown')

    return jsonify(result), 200

# ========================================
# Endpoint 4: Batch predict
# POST /predict/batch
# ========================================
    
@app.route('/predict/batch', methods = ['POST'])
def predict_batch():
    data = request.get_json()

    if not data or 'engines' not in data:
        return jsonify({
            'success': False,
            'error': "Please provide engines list"
        }), 400
    
    results = []
    for engine_data in data['engines']: 
        result = predict_rul(engine_data)
        result['engine_id'] = engine_data.get('engine_id')
        result['cycle'] = engine_data.get('cycle')
        results.append(result)

    # Summary statistics
    ruls = [r['predicted_RUL'] for r in results if r['success']]

    summary = {
        'total_engines': len(results),
        'critical' : sum(1 for r in results if r.get('health_status') == 'CRITICAL'), 
        'warning' : sum(1 for r in results if r.get('health_status') == 'WARNING'),
        'monitor' : sum( 1 for r in results if r.get('health_status') == 'MONITOR'),
        'healthy' : sum( 1 for r in results if r.get('health_status') == 'HEALTHY'),
        'avg_RUL' : round(np.mean(ruls), 1) if ruls else 0
    }

    return jsonify({
        'success' : True,
        'summary': summary,
        'predictions': results, 
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }), 200

# ===================================
# Run the app
# ===================================

if __name__ == '__main__': 
    print("\n Turbofan Predictive Maintenance API")
    print("=" * 40)
    print("Endpoints:")
    print("  GET  /health       → API health check")
    print("  GET  /model-info   → Model details")
    print("  POST /predict      → Single prediction")
    print("  POST /predict/batch → Batch prediction")
    print("=" * 40)
    app.run(debug=True, port=5000)