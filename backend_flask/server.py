from flask import Flask, request, jsonify
from digit_predictor import DigitPredictor
import pandas as pd
import numpy as np

app = Flask(__name__)

predictor = DigitPredictor()


@app.route("/")
def home():
    return "Digit Prediction API is running."


# ============================================================
#  FIXED API: Accepts ONE sample, expands to 3-row window
# ============================================================
@app.route("/predict_digit", methods=["POST"])
def predict_digit():
    data = request.json

    # You will receive:
    #  { "ax":0.5, "ay":1.2, "az":9.8, "gx":0, "gy":0, "gz":0 }

    # Convert single reading → 3 repeated rows (context window size = 3)
    df = pd.DataFrame([data, data, data])   # 3 rows

    # Run prediction
    pred = predictor.predict_digit(df)

    return jsonify({"digit": int(pred) + 1})


# ============================================================
#  MULTI-DIGIT PASSWORD API (fixed to repeat rows)
# ============================================================
@app.route("/predict_sequence", methods=["POST"])
def predict_sequence():
    data = request.json   # contains "sequences":[ {...}, {...}, ... ]

    preds = []
    for seq in data["sequences"]:
        df = pd.DataFrame([seq, seq, seq])  # make fake 3-row window
        d = predictor.predict_digit(df)
        preds.append(int(d) + 1)

    final_password = "".join(str(d) for d in preds)

    return jsonify({
        "digits": preds,
        "password": final_password
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
