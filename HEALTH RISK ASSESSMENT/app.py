from flask import Flask, request, jsonify, render_template, send_file
import joblib
import pandas as pd
import numpy as np
import os
from openai import OpenAI
from datetime import datetime, timedelta
import io
import csv

# optional PDF libs
try:
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ------------------------------
# Paths to model and encoders
# ------------------------------
BASE_DIR = "C:/Users/visha/OneDrive/문서/Desktop/Health -risk-app/backend"

MODEL_PATH = os.path.join(BASE_DIR, "health_risk_model (3).pkl").replace("\\", "/")
FEATURE_ENCODERS_PATH = os.path.join(BASE_DIR, "feature_encoders.pkl").replace("\\", "/")
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, "target_encoder.pkl").replace("\\", "/")

# ------------------------------
# Load model and encoders
# ------------------------------
model = joblib.load(MODEL_PATH)
feature_encoders = joblib.load(FEATURE_ENCODERS_PATH)
target_encoder = joblib.load(TARGET_ENCODER_PATH)

# ------------------------------
# Features trained with
# ------------------------------
TRAINED_FEATURES = [
    "age","gender","blood_type","medical_condition","doctor","hospital",
    "insurance_provider","billing_amount","room_number","admission_type",
    "medication","admission_year","admission_month","admission_day",
    "discharge_year","discharge_month","discharge_day","stay_length"
]

# ------------------------------
# Risk label mapping
# ------------------------------
RISK_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__, template_folder="templates")

# ------------------------------
# OpenAI client (reads from ENV key)
# ------------------------------
client = OpenAI()

# === Dashboard Additions ===
prediction_logs = []  # in-memory storage for dashboard entries

# ------------------------------
# Helper: convert NumPy types
# ------------------------------
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ------------------------------
# Home route
# ------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ------------------------------
# Predict route
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = datetime.utcnow()
        data = request.get_json()

        input_data = {f: 0 for f in TRAINED_FEATURES}
        for f in data:
            if f in input_data:
                input_data[f] = data[f]

        # Convert numeric fields
        numeric_cols = [
            "age", "billing_amount", "room_number", "admission_year",
            "admission_month", "admission_day", "discharge_year",
            "discharge_month", "discharge_day", "stay_length"
        ]
        for col in numeric_cols:
            if col in input_data:
                try:
                    input_data[col] = float(input_data[col])
                except Exception:
                    input_data[col] = 0.0

        # Encode categorical fields
        for col in feature_encoders:
            if col in input_data:
                encoder = feature_encoders[col]
                value = input_data[col]
                try:
                    if value in encoder.classes_:
                        input_data[col] = int(encoder.transform([value])[0])
                    else:
                        input_data[col] = -1
                except Exception:
                    input_data[col] = -1

        df = pd.DataFrame([[input_data[f] for f in TRAINED_FEATURES]], columns=TRAINED_FEATURES)
        prediction_num = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        prediction_label = RISK_LABELS[int(prediction_num)]

        # 🧠 Basic AI Explanation
        prompt = f"""
        You are a medical AI assistant.
        A patient has these details: {data}.
        The ML model predicted: {prediction_label}.
        Explain briefly (3-4 sentences) why this risk level may apply, 
        in simple terms for a patient to understand.
        """
        try:
            ai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            explanation = ai_response.choices[0].message.content.strip()
        except Exception:
            explanation = "Explanation currently unavailable."

        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000.0

        # Store log
        try:
            log_entry = {
                "time": datetime.utcnow(),
                "prediction": prediction_label,
                "probabilities": convert_numpy(probabilities),
                "latency": float(latency_ms),
                "age": data.get("age", None),
                "gender": data.get("gender", None),
                "condition": data.get("medical_condition", None)
            }
            prediction_logs.append(log_entry)
            if len(prediction_logs) > 100:
                prediction_logs.pop(0)
        except Exception:
            pass

        return jsonify({
            "prediction": prediction_label,
            "probabilities": convert_numpy(probabilities),
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# ------------------------------
# 🧩 Agentic AI Decision Route
# ------------------------------
@app.route("/agent_decision", methods=["POST"])
def agent_decision():
    """
    This endpoint provides actionable, context-aware recommendations
    from an agentic AI layer based on model prediction + explanation.
    """
    try:
        req = request.get_json()
        prediction = req.get("prediction")
        explanation = req.get("explanation")
        input_data = req.get("input_data", {})

        # Construct contextual prompt for the agent
        agent_prompt = f"""
        You are an Agentic Health AI advisor.
        The system predicted: {prediction}.
        Explanation from model: {explanation}.
        Patient info: {input_data}.
        Based on this, give one short, actionable recommendation (1–2 sentences)
        for what the patient should do next — e.g., lifestyle advice,
        medical consultation, or self-monitoring guidance.
        Keep it positive, human-like, and easy to understand.
        """
        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": agent_prompt}],
            max_tokens=100
        )
        advice = ai_response.choices[0].message.content.strip()

        return jsonify({"advice": advice})

    except Exception as e:
        return jsonify({"advice": "Agentic AI advice unavailable."})

# ------------------------------
# Dashboard Data Route
# ------------------------------
@app.route("/dashboard-data", methods=["GET"])
def dashboard_data():
    try:
        now = datetime.utcnow()
        last_7_days = now - timedelta(days=7)
        last_24h = now - timedelta(hours=24)

        total_predictions = len(prediction_logs)
        preds_7d = [p for p in prediction_logs if p["time"] >= last_7_days]
        preds_24h = [p for p in prediction_logs if p["time"] >= last_24h]

        high_7d = sum(1 for p in preds_7d if p["prediction"] == "High Risk")
        medium_7d = sum(1 for p in preds_7d if p["prediction"] == "Medium Risk")
        low_7d = sum(1 for p in preds_7d if p["prediction"] == "Low Risk")

        high_risk_percent_7d = (high_7d / len(preds_7d) * 100) if preds_7d else 0.0
        avg_latency_24h = float(np.mean([p["latency"] for p in preds_24h])) if preds_24h else 0.0

        recent = []
        for p in reversed(prediction_logs[-10:]):
            recent.append({
                "time": p["time"].strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": p["prediction"],
                "probabilities": p["probabilities"],
                "latency": round(float(p["latency"]), 2),
                "age": p.get("age"),
                "gender": p.get("gender"),
                "condition": p.get("condition")
            })

        return jsonify({
            "total_predictions": total_predictions,
            "predictions_7d": len(preds_7d),
            "high_risk_percent_7d": round(high_risk_percent_7d, 2),
            "avg_latency_24h": round(avg_latency_24h, 2),
            "risk_counts_7d": {
                "low": low_7d,
                "medium": medium_7d,
                "high": high_7d
            },
            "recent_predictions": recent
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# ------------------------------
# Export routes (CSV, Excel, PDF)
# ------------------------------
def logs_to_dataframe():
    rows = []
    for p in prediction_logs:
        rows.append({
            "time": p["time"].strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": p["prediction"],
            "prob_low": p["probabilities"][0] if len(p["probabilities"])>0 else None,
            "prob_medium": p["probabilities"][1] if len(p["probabilities"])>1 else None,
            "prob_high": p["probabilities"][2] if len(p["probabilities"])>2 else None,
            "latency_ms": round(float(p.get("latency") or 0.0), 2),
            "age": p.get("age"),
            "gender": p.get("gender"),
            "condition": p.get("condition")
        })
    return pd.DataFrame(rows)

@app.route("/export/csv")
def export_csv():
    try:
        df = logs_to_dataframe()
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(io.BytesIO(buf.getvalue().encode('utf-8')),
                         mimetype="text/csv",
                         download_name="dashboard_export.csv",
                         as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/export/excel")
def export_excel():
    try:
        df = logs_to_dataframe()
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='predictions')
        buf.seek(0)
        return send_file(buf,
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                         download_name="dashboard_export.xlsx",
                         as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/export/pdf")
def export_pdf():
    try:
        df = logs_to_dataframe()
        if REPORTLAB_AVAILABLE:
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=letter)
            styles = getSampleStyleSheet()
            flow = [Paragraph("Health Risk Dashboard Export", styles['Title']), Spacer(1, 12)]
            data = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
            table = Table(data, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('FONT', (0,0), (-1,0), 'Helvetica-Bold')
            ]))
            flow.append(table)
            doc.build(flow)
            buf.seek(0)
            return send_file(buf, mimetype='application/pdf',
                             download_name="dashboard_export.pdf",
                             as_attachment=True)
        else:
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            csv_buf.seek(0)
            return send_file(io.BytesIO(csv_buf.getvalue().encode('utf-8')),
                             mimetype="text/csv",
                             download_name="dashboard_export.csv",
                             as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------
# Run Flask
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
