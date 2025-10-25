import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import torch
import os, csv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level up from src
MODELS_DIR = os.path.join(BASE_DIR, "models")

pca = joblib.load(os.path.join(MODELS_DIR, "pca_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_model.pkl"))
clf_model = joblib.load(os.path.join(MODELS_DIR, "classifier_model.pkl"))

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)
CSV_FILE = "registros.csv"
# -------------------------------
# 2. FUNCIÓN DE EMBEDDINGS
# -------------------------------
def get_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = bert_model(**tokens)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# -------------------------------
# 3. FUNCIÓN DE PREDICCIÓN
# -------------------------------
def predecir_top3(descripcion):
    embedding = get_embeddings([descripcion])
    embedding_scaled = scaler.transform(pca.transform(embedding))
    print(len(embedding[0]))
    print(len(embedding_scaled[0]))
    # Probabilidades
    if hasattr(clf_model, 'predict_proba'):
        probas = clf_model.predict_proba(embedding_scaled)[0]
        clases = clf_model.classes_
    else:
        scores = clf_model.decision_function(embedding_scaled)[0]
        exp_scores = np.exp(scores - np.max(scores))
        probas = exp_scores / exp_scores.sum()
        clases = clf_model.classes_
    
    top_idx = np.argsort(probas)[::-1] 
    resultados = [
        {"componentes": clases[i].strip(), "probabilidad": float(probas[i]), "embedding": str(embedding)}
        for i in top_idx
    ]
    return resultados

# -------------------------------
# 4. API FLASK
# -------------------------------
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        descripcion = data.get('description', '')
        if not descripcion:
            return jsonify({'error': 'Falta el campo "description"'}), 400
        
        resultados = predecir_top3(descripcion)
        return resultados
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save_register', methods=['POST'])
def save_register():
    data = request.get_json()

    # Prepare data fields
    fecha = datetime.now().strftime("%Y-%m-%d")
    sintoma = data.get("sintoma", "")
    componente = data.get("componente", "")
    reparacion = data.get("reparacion", "")

    # Check if file exists
    file_exists = os.path.isfile(CSV_FILE)

    # Determine the next ID
    next_id = 1
    if file_exists:
        try:
            with open(CSV_FILE, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                if rows:
                    last_id = int(rows[-1]["ID"])
                    next_id = last_id + 1
        except Exception:
            # If any issue reading ID, reset to 1
            next_id = 1

    # Write to CSV
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        fieldnames = ["ID", "Fecha", "Síntoma", "Componente", "Reparación"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header if file not exists
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "ID": next_id,
            "Fecha": fecha,
            "Síntoma": sintoma,
            "Componente": componente,
            "Reparación": reparacion
        })

    return jsonify({"status": "success", "ID": next_id}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
