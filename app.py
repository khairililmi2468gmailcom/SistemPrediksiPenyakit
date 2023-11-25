from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)
app.template_folder = 'mbahPenyakitWeb'

# Load the trained model
loaded_model = joblib.load("rf_model.joblib")

# Mapping untuk nama penyakit
penyakit_mapping = {
    0: '(vertigo) Paroymsal Positional Vertigo',
    1: 'AIDS',
    2: 'Acne',
    3: 'Alcoholic hepatitis',
    4: 'Allergy',
    5: 'Arthritis',
    6: 'Bronchial Asthma',
    7: 'Cervical spondylosis',
    8: 'Chicken pox',
    9: 'Chronic cholestasis',
    10: 'Common Cold',
    11: 'Dengue',
    12: 'Diabetes',
    13: 'Dimorphic hemmorhoids(piles)',
    14: 'Drug Reaction',
    15: 'Fungal infection',
    16: 'GERD',
    17: 'Gastroenteritis',
    18: 'Heart attack',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    23: 'Hypertension',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    26: 'Hypothyroidism',
    27: 'Impetigo',
    28: 'Jaundice',
    29: 'Malaria',
    30: 'Migraine',
    31: 'Osteoarthristis',
    32: 'Paralysis (brain hemorrhage)',
    33: 'Peptic ulcer disease',
    34: 'Pneumonia',
    35: 'Psoriasis',
    36: 'Tuberculosis',
    37: 'Typhoid',
    38: 'Urinary tract infection',
    39: 'Varicose veins',
    40: 'Hepatitis A'
}

@app.route('/')
def home():
    # Set `prediction` ke `None` saat halaman diperbarui
    prediction = None
    return render_template('index2.html', prediction=prediction)
@app.route('/prediksi')
def prediksi():
    # Set `prediction` ke `None` saat halaman diperbarui
    prediction = None
    return render_template('prediksi.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Dapatkan data yang diinputkan oleh pengguna
        features = []
        features = request.get_json().get('features', [])
        print("Received features:", features)
        # for i in range(1, 49):
        #     feature_name = f'feature{i}'
        #     feature_value = float(request.json[feature_name])
        #     features.append(feature_value)

        # Lakukan prediksi dengan model Anda
        prediction = loaded_model.predict([features])[0]
        predicted_penyakit = penyakit_mapping[prediction]

        return jsonify({'prediction': predicted_penyakit, 'error': None})
    except Exception as e:
        return jsonify({'prediction': None, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0')

