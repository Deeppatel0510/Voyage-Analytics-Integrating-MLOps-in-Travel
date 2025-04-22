import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from pyngrok import ngrok

# Load the trained Random Forest model and scaler
random_forest_model = pickle.load(open("random_forest.pkl", "rb"))
scaler = pickle.load(open("scaling.pkl", "rb"))

# Features required for prediction
features_ordering = [
    "from_Florianopolis (SC)", "from_Sao_Paulo (SP)", "from_Salvador (BH)",
    "from_Brasilia (DF)", "from_Rio_de_Janeiro (RJ)", "from_Campo_Grande (MS)",
    "from_Aracaju (SE)", "from_Natal (RN)", "from_Recife (PE)",
    "destination_Florianopolis (SC)", "destination_Sao_Paulo (SP)",
    "destination_Salvador (BH)", "destination_Brasilia (DF)",
    "destination_Rio_de_Janeiro (RJ)", "destination_Campo_Grande (MS)",
    "destination_Aracaju (SE)", "destination_Natal (RN)", "destination_Recife (PE)",
    "flightType_economic", "flightType_firstClass", "flightType_premium",
    "agency_Rainbow", "agency_CloudFy", "agency_FlyingDrops",
    "week_no", "week_day", "day"
]

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# HTML Form for user input
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Flight Price Prediction</title>
</head>
<body>
    <h2>Flight Price Prediction</h2>
    <form action="/predict" method="post">
        <label>From:</label>
        <select name="from_location">
            <option value="from_Sao_Paulo (SP)">Sao Paulo (SP)</option>
            <option value="from_Rio_de_Janeiro (RJ)">Rio de Janeiro (RJ)</option>
            <option value="from_Campo_Grande (MS)">Campo Grande (MS)</option>
            <option value="from_Florianopolis (SC)">Florianopolis (SC)</option>
            <option value="from_Salvador (BH)">Salvador (BH)</option>
            <option value="from_Brasilia (DF)">Brasilia (DF)</option>
            <option value="from_Aracaju (SE)">Aracaju (SE)</option>
            <option value="from_Natal (RN)">Natal (RN)</option>
            <option value="from_Recife (PE)">Recife (PE)</option>
        </select>
        <br><br>

        <label>Destination:</label>
        <select name="destination">
            <option value="destination_Sao_Paulo (SP)">Sao Paulo (SP)</option>
            <option value="destination_Rio_de_Janeiro (RJ)">Rio de Janeiro (RJ)</option>
            <option value="destination_Campo_Grande (MS)">Campo Grande (MS)</option>
            <option value="destination_Florianopolis (SC)">Florianopolis (SC)</option>
            <option value="destination_Salvador (BH)">Salvador (BH)</option>
            <option value="destination_Brasilia (DF)">Brasilia (DF)</option>
            <option value="destination_Aracaju (SE)">Aracaju (SE)</option>
            <option value="destination_Natal (RN)">Natal (RN)</option>
            <option value="destination_Recife (PE)">Recife (PE)</option>
        </select>
        <br><br>

        <label>Flight Type:</label>
        <select name="flight_type">
            <option value="flightType_economic">Economic</option>
            <option value="flightType_firstClass">First Class</option>
            <option value="flightType_premium">Premium</option>
        </select>
        <br><br>

        <label>Agency:</label>
        <select name="agency">
            <option value="agency_Rainbow">Rainbow</option>
            <option value="agency_CloudFy">CloudFy</option>
            <option value="agency_FlyingDrops">FlyingDrops</option>
        </select>
        <br><br>

        <label>Week No:</label>
        <input type="number" name="week_no" min="1" required><br><br>

        <label>Week Day:</label>
        <input type="number" name="week_day" min="1" max="52" required><br><br>

        <label>Day:</label>
        <input type="number" name="day" min="1" max="31" required><br><br>

        <button type="submit">Predict</button>
    </form>

    {% if price %}
    <h3>Predicted Flight Price: ${{ price }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template_string(HTML_FORM)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()
        input_features = {col: 0 for col in features_ordering}

        # Assign 1 to selected categorical features
        for key in ["from_location", "destination", "flight_type", "agency"]:
            if form_data[key] in input_features:
                input_features[form_data[key]] = 1

        # Assign numerical values
        input_features["week_no"] = int(form_data["week_no"])
        input_features["week_day"] = int(form_data["week_day"])
        input_features["day"] = int(form_data["day"])

        # Convert to DataFrame and scale
        df = pd.DataFrame([input_features])
        df_scaled = scaler.transform(df)

        # Predict
        prediction = random_forest_model.predict(df_scaled)[0]

        return render_template_string(HTML_FORM, price=round(prediction, 2))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)