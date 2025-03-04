from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the scaler (use the same scaler used during training)
scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        inputs = np.array([[
            data['terminal_voltage'],
            data['terminal_current'],
            data['temperature'],
            data['charge_current'],
            data['charge_voltage'],
            data['time'],
            data['capacity'],
            data['cycle']
        ]])

        # Scale the input data (ensure same scaling as training data)
        inputs_scaled = scaler.transform(inputs)

        # Predict SOH
        soh = model.predict(inputs_scaled)[0]

        # Estimate the distance based on SOH
        estimated_distance = (soh / 100) * 400  # Assuming 400 km at 100% SOH

        return jsonify({'soh': round(soh, 2), 'distance': round(estimated_distance, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
