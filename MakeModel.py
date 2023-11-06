# -*- coding: utf-8 -*-
"""Project Akhir AI.ipynb

"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from googletrans import Translator

pip install googletrans==4.0.0-rc1

"""##Input Dataset Training dan Testing dari Disease Dataset"""

# Kamus untuk menerjemahkan nama penyakit
kamus_penyakit = {
    "itching": "gatal",
    "skin_rash": "ruam kulit",
    "nodal_skin_eruptions": "nodal skin erupsi",
    "continuous_sneezing": "terus menerus bersin",
    "shivering": "gemetaran",
    "chills": "panas dingin",
    "joint_pain": "nyeri sendi",
    "stomach_pain": "sakit perut",
    "acidity": "keasaman",
    "ulcers_on_tongue": "bisul di lidah",
    "muscle_wasting": "pengecilan otot",
    "vomiting": "muntah",
    "burning_micturition": "pembakaran kemih",
    "spotting_ urination": "bercak buang air kecil",
    "fatigue": "kelelahan",
    "weight_gain": "berat penambahan",
    "anxiety": "kecemasan",
    "cold_hands_and_feets": "tangan dan kaki dingin",
    "mood_swings": "suasana hati ayunan",
    "weight_loss": "berat penurunan",
    "restlessness": "kegelisahan",
    "lethargy": "kelesuan",
    "patches_in_throat": "patch in throat",
    "irregular_sugar_level": "kadar gula tidak teratur",
    "cough": "batuk",
    "high_fever": "demam tinggi",
    "sunken_eyes": "cekung mata",
    "breathlessness": "sesak napas",
    "sweating": "berkeringat",
    "dehydration": "dehidrasi",
    "indigestion": "gangguan pencernaan",
    "headache": "sakit kepala",
    "yellowish_skin": "kulit kekuningan",
    "dark_urine": "urin gelap",
    "nausea": "mual",
    "loss_of_appetite": "kehilangan selera makan",
    "pain_behind_the_eyes": "rasa sakit di belakang mata",
    "back_pain": "sakit punggung",
    "constipation": "sembelit",
    "abdominal_pain": "sakit perut",
    "diarrhoea": "diare",
    "mild_fever": "ringan demam",
    "yellow_urine": "kuning urin",
    "yellowing_of_eyes": "mata menguning",
    "acute_liver_failure": "kegagalan hati akut",
    "fluid_overload": "kelebihan cairan",
    "swelling_of_stomach": "pembengkakan perut",
    "swelled_lymph_nodes": "pembengkakan kelenjar getah bening",
    "malaise": "rasa tidak enak",
    "blurred_and_distorted_vision": "penglihatan kabur dan terdistorsi",
    "phlegm": "dahak",
    "throat_irritation": "tenggorokan iritasi",
    "redness_of_eyes": "kemerahan mata",
    "sinus_pressure": "tekanan sinus",
    "runny_nose": "pilek",
    "congestion": "penyumbatan",
    "chest_pain": "nyeri dada",
    "weakness_in_limbs": "kelemahan di anggota badan",
    "fast_heart_rate": "detak jantung cepat",
    "pain_during_bowel_movements": "nyeri selama gerakan usus",
    "pain_in_anal_region": "nyeri di anal wilayah",
    "bloody_stool": "bangku berdarah",
    "irritation_in_anus": "iritasi di anus",
    "neck_pain": "sakit leher",
    "dizziness": "pusing",
    "cramps": "kram",
    "bruising": "memar",
    "obesity": "kegemukan",
    "swollen_legs": "kaki bengkak",
    "swollen_blood_vessels": "pembuluh darah bengkak",
    "puffy_face_and_eyes": "wajah dan mata bengkak",
    "enlarged_thyroid": "pembesaran tiroid",
    "brittle_nails": "kuku rapuh",
    "swollen_extremities": "bengkak ekstremitas",
    "excessive_hunger": "berlebihan kelaparan",
    "extra_marital_contacts": "extra_marital_contacts",
    "drying_and_tingling_lips": "mengeringkan dan kesemutan bibir",
    "slurred_speech": "cadel ucapan",
    "knee_pain": "sakit lutut",
    "hip_joint_pain": "pinggul sendi nyeri",
    "muscle_weakness": "kelemahan otot",
    "stiff_neck": "leher kaku",
    "swelling_joints": "pembengkakan sendi",
    "movement_stiffness": "gerakan kekakuan",
    "spinning_movements": "gerakan berputar",
    "loss_of_balance": "kehilangan keseimbangan",
    "unsteadiness": "kegoyangan",
    "weakness_of_one_body_side": "kelemahan sisi satu tubuh",
    "loss_of_smell": "kehilangan bau",
    "bladder_discomfort": "kandung kemih ketidaknyamanan",
    "foul_smell_of urine": "urin berbau busuk",
    "continuous_feel_of_urine": "rasa air urin yang terus-menerus",
    "passage_of gases": "bagian of gas",
    "internal_itching": "internal gatal",
    "toxic_look_(typhos)": "tampilan toksik (kesalahan ketik)",
    "depression": "depresi",
    "irritability": "sifat lekas marah",
    "muscle_pain": "nyeri otot",
    "altered_sensorium": "diubah sensorium",
    "red_spots_over_body": "bintik merah di atas tubuh",
    "belly_pain": "sakit perut",
    "abnormal_menstruation": "abnormal menstruasi",
    "dischromic _patches": "patches dischromic",
    "watering_from_eyes": "penyiraman dari mata",
    "increased_appetite": "peningkatan nafsu makan",
    "polyuria": "poliuria",
    "family_history": "sejarah keluarga",
    "mucoid_sputum": "mukoid dahak",
    "rusty_sputum": "berkarat dahak",
    "lack_of_concentration": "kurang konsenterasi",
    "visual_disturbances": "gangguan visual",
    "receiving_blood_transfusion": "menerima transfusi darah",
    "receiving_unsterile_injections": "menerima suntikan tidak steril",
    "coma": "koma",
    "stomach_bleeding": "perut pendarahan",
    "distention_of_abdomen": "distensi perut",
    "history_of_alcohol_consumption": "riwayat konsumsi alkohol",
    "fluid_overload": "kelebihan cairan",
    "blood_in_sputum": "darah dalam dahak",
    "prominent_veins_on_calf": "vena menonjol di betis",
    "palpitations": "jantung berdebar",
    "painful_walking": "menyakitkan berjalan",
    "pus_filled_pimples": "pus filled jerawat",
    "blackheads": "komedo",
    "scurring": "berlarian",
    "skin_peeling": "pengelupasan kulit",
    "silver_like_dusting": "debu seperti perak",
    "small_dents_in_nails": "penyok kecil di kuku",
    "inflammatory_nails": "kuku inflamasi",
    "blister": "lepuh",
    "red_sore_around_nose": "red sore around nose",
    "yellow_crust_ooze": "kuning kerak ooze",
    "prognosis": "prognosa"
}


# Membaca dataset dari file CSV
df = pd.read_csv("Testing.csv")

# Mengganti nama kolom berdasarkan kamus
df.rename(columns=kamus_penyakit, inplace=True)

# Menyimpan dataset yang telah diubah ke dalam file CSV
df.to_csv("dataTestingPre.csv", index=False)

# Kamus untuk menerjemahkan nama penyakit
kamus_penyakit = {
    "itching": "gatal",
    "skin_rash": "ruam kulit",
    "nodal_skin_eruptions": "nodal skin erupsi",
    "continuous_sneezing": "terus menerus bersin",
    "shivering": "gemetaran",
    "chills": "panas dingin",
    "joint_pain": "nyeri sendi",
    "stomach_pain": "sakit perut",
    "acidity": "keasaman",
    "ulcers_on_tongue": "bisul di lidah",
    "muscle_wasting": "pengecilan otot",
    "vomiting": "muntah",
    "burning_micturition": "pembakaran kemih",
    "spotting_ urination": "bercak buang air kecil",
    "fatigue": "kelelahan",
    "weight_gain": "berat penambahan",
    "anxiety": "kecemasan",
    "cold_hands_and_feets": "tangan dan kaki dingin",
    "mood_swings": "suasana hati ayunan",
    "weight_loss": "berat penurunan",
    "restlessness": "kegelisahan",
    "lethargy": "kelesuan",
    "patches_in_throat": "patch in throat",
    "irregular_sugar_level": "kadar gula tidak teratur",
    "cough": "batuk",
    "high_fever": "demam tinggi",
    "sunken_eyes": "cekung mata",
    "breathlessness": "sesak napas",
    "sweating": "berkeringat",
    "dehydration": "dehidrasi",
    "indigestion": "gangguan pencernaan",
    "headache": "sakit kepala",
    "yellowish_skin": "kulit kekuningan",
    "dark_urine": "urin gelap",
    "nausea": "mual",
    "loss_of_appetite": "kehilangan selera makan",
    "pain_behind_the_eyes": "rasa sakit di belakang mata",
    "back_pain": "sakit punggung",
    "constipation": "sembelit",
    "abdominal_pain": "sakit perut",
    "diarrhoea": "diare",
    "mild_fever": "ringan demam",
    "yellow_urine": "kuning urin",
    "yellowing_of_eyes": "mata menguning",
    "acute_liver_failure": "kegagalan hati akut",
    "fluid_overload": "kelebihan cairan",
    "swelling_of_stomach": "pembengkakan perut",
    "swelled_lymph_nodes": "pembengkakan kelenjar getah bening",
    "malaise": "rasa tidak enak",
    "blurred_and_distorted_vision": "penglihatan kabur dan terdistorsi",
    "phlegm": "dahak",
    "throat_irritation": "tenggorokan iritasi",
    "redness_of_eyes": "kemerahan mata",
    "sinus_pressure": "tekanan sinus",
    "runny_nose": "pilek",
    "congestion": "penyumbatan",
    "chest_pain": "nyeri dada",
    "weakness_in_limbs": "kelemahan di anggota badan",
    "fast_heart_rate": "detak jantung cepat",
    "pain_during_bowel_movements": "nyeri selama gerakan usus",
    "pain_in_anal_region": "nyeri di anal wilayah",
    "bloody_stool": "bangku berdarah",
    "irritation_in_anus": "iritasi di anus",
    "neck_pain": "sakit leher",
    "dizziness": "pusing",
    "cramps": "kram",
    "bruising": "memar",
    "obesity": "kegemukan",
    "swollen_legs": "kaki bengkak",
    "swollen_blood_vessels": "pembuluh darah bengkak",
    "puffy_face_and_eyes": "wajah dan mata bengkak",
    "enlarged_thyroid": "pembesaran tiroid",
    "brittle_nails": "kuku rapuh",
    "swollen_extremities": "bengkak ekstremitas",
    "excessive_hunger": "berlebihan kelaparan",
    "extra_marital_contacts": "extra_marital_contacts",
    "drying_and_tingling_lips": "mengeringkan dan kesemutan bibir",
    "slurred_speech": "cadel ucapan",
    "knee_pain": "sakit lutut",
    "hip_joint_pain": "pinggul sendi nyeri",
    "muscle_weakness": "kelemahan otot",
    "stiff_neck": "leher kaku",
    "swelling_joints": "pembengkakan sendi",
    "movement_stiffness": "gerakan kekakuan",
    "spinning_movements": "gerakan berputar",
    "loss_of_balance": "kehilangan keseimbangan",
    "unsteadiness": "kegoyangan",
    "weakness_of_one_body_side": "kelemahan sisi satu tubuh",
    "loss_of_smell": "kehilangan bau",
    "bladder_discomfort": "kandung kemih ketidaknyamanan",
    "foul_smell_of urine": "urin berbau busuk",
    "continuous_feel_of_urine": "rasa air urin yang terus-menerus",
    "passage_of gases": "bagian of gas",
    "internal_itching": "internal gatal",
    "toxic_look_(typhos)": "tampilan toksik (kesalahan ketik)",
    "depression": "depresi",
    "irritability": "sifat lekas marah",
    "muscle_pain": "nyeri otot",
    "altered_sensorium": "diubah sensorium",
    "red_spots_over_body": "bintik merah di atas tubuh",
    "belly_pain": "sakit perut",
    "abnormal_menstruation": "abnormal menstruasi",
    "dischromic _patches": "patches dischromic",
    "watering_from_eyes": "penyiraman dari mata",
    "increased_appetite": "peningkatan nafsu makan",
    "polyuria": "poliuria",
    "family_history": "sejarah keluarga",
    "mucoid_sputum": "mukoid dahak",
    "rusty_sputum": "berkarat dahak",
    "lack_of_concentration": "kurang konsenterasi",
    "visual_disturbances": "gangguan visual",
    "receiving_blood_transfusion": "menerima transfusi darah",
    "receiving_unsterile_injections": "menerima suntikan tidak steril",
    "coma": "koma",
    "stomach_bleeding": "perut pendarahan",
    "distention_of_abdomen": "distensi perut",
    "history_of_alcohol_consumption": "riwayat konsumsi alkohol",
    "fluid_overload": "kelebihan cairan",
    "blood_in_sputum": "darah dalam dahak",
    "prominent_veins_on_calf": "vena menonjol di betis",
    "palpitations": "jantung berdebar",
    "painful_walking": "menyakitkan berjalan",
    "pus_filled_pimples": "pus filled jerawat",
    "blackheads": "komedo",
    "scurring": "berlarian",
    "skin_peeling": "pengelupasan kulit",
    "silver_like_dusting": "debu seperti perak",
    "small_dents_in_nails": "penyok kecil di kuku",
    "inflammatory_nails": "kuku inflamasi",
    "blister": "lepuh",
    "red_sore_around_nose": "red sore around nose",
    "yellow_crust_ooze": "kuning kerak ooze",
    "prognosis": "prognosa"
}


# Membaca dataset dari file CSV
df = pd.read_csv("Training.csv")

# Mengganti nama kolom berdasarkan kamus
df.rename(columns=kamus_penyakit, inplace=True)

# Menyimpan dataset yang telah diubah ke dalam file CSV
df.to_csv("dataTrainingPre.csv", index=False)

Train_data=pd.read_csv("dataTrainingPre.csv")
Test_data=pd.read_csv("dataTestingPre.csv")

train_df=pd.DataFrame(Train_data)
train_df.head(15)
# train_df=train_df.drop(['Unnamed: 133'],axis=1)
train_df.head()

"""##menambah variabel target dan fitur dalam kumpulan data

"""

list(train_df.columns)

test_df=pd.DataFrame(Test_data)
test_df.head()

"""## Menghapus Parameter yang tidak perlu"""

y_train=train_df['prognosa']
x_train=train_df.drop(['rasa tidak enak',
 'penglihatan kabur dan terdistorsi',
 'dahak',
 'tenggorokan iritasi',
 'kemerahan mata',
 'tekanan sinus',
 'pilek',
 'penyumbatan',
 'nyeri dada',
 'kelemahan di anggota badan',
 'detak jantung cepat',
 'nyeri selama gerakan usus',
 'nyeri di anal wilayah',
 'bangku berdarah',
 'iritasi di anus',
 'sakit leher',
 'pusing',
 'kram',
 'memar',
 'kegemukan',
 'kaki bengkak',
 'pembuluh darah bengkak',
 'wajah dan mata bengkak',
 'pembesaran tiroid',
 'kuku rapuh',
 'swollen_extremeties',
 'berlebihan kelaparan',
 'extra_marital_contacts',
 'mengeringkan dan kesemutan bibir',
 'cadel ucapan',
 'sakit lutut',
 'pinggul sendi nyeri',
 'kelemahan otot',
 'leher kaku',
 'pembengkakan sendi',
 'gerakan kekakuan',
 'gerakan berputar',
 'kehilangan keseimbangan',
 'kegoyangan',
 'kelemahan sisi satu tubuh',
 'kehilangan bau',
 'kandung kemih ketidaknyamanan',
 'urin berbau busuk',
 'rasa air urin yang terus-menerus',
 'passage_of_gases',
 'internal gatal',
 'tampilan toksik (kesalahan ketik)',
 'depresi',
 'sifat lekas marah',
 'nyeri otot',
 'diubah sensorium',
 'bintik merah di atas tubuh',
 'sakit perut.2',
 'abnormal menstruasi',
 'patches dischromic',
 'penyiraman dari mata',
 'peningkatan nafsu makan',
 'poliuria',
 'sejarah keluarga',
 'mukoid dahak',
 'berkarat dahak',
 'kurang konsenterasi',
 'gangguan visual',
 'menerima transfusi darah',
 'menerima suntikan tidak steril',
 'koma',
 'perut pendarahan',
 'distensi perut',
 'riwayat konsumsi alkohol',
 'fluid_overload.1',
 'darah dalam dahak',
 'vena menonjol di betis',
 'jantung berdebar',
 'menyakitkan berjalan',
 'pus filled jerawat',
 'komedo',
 'berlarian',
 'pengelupasan kulit',
 'debu seperti perak',
 'penyok kecil di kuku',
 'kuku inflamasi',
 'lepuh',
 'red sore around nose',
 'kuning kerak ooze',
 'prognosa'],axis=1)
y_train.unique()

print(x_train.shape)

test_df.isna().sum()

x_test=test_df.drop(['rasa tidak enak',
 'penglihatan kabur dan terdistorsi',
 'dahak',
 'tenggorokan iritasi',
 'kemerahan mata',
 'tekanan sinus',
 'pilek',
 'penyumbatan',
 'nyeri dada',
 'kelemahan di anggota badan',
 'detak jantung cepat',
 'nyeri selama gerakan usus',
 'nyeri di anal wilayah',
 'bangku berdarah',
 'iritasi di anus',
 'sakit leher',
 'pusing',
 'kram',
 'memar',
 'kegemukan',
 'kaki bengkak',
 'pembuluh darah bengkak',
 'wajah dan mata bengkak',
 'pembesaran tiroid',
 'kuku rapuh',
 'swollen_extremeties',
 'berlebihan kelaparan',
 'extra_marital_contacts',
 'mengeringkan dan kesemutan bibir',
 'cadel ucapan',
 'sakit lutut',
 'pinggul sendi nyeri',
 'kelemahan otot',
 'leher kaku',
 'pembengkakan sendi',
 'gerakan kekakuan',
 'gerakan berputar',
 'kehilangan keseimbangan',
 'kegoyangan',
 'kelemahan sisi satu tubuh',
 'kehilangan bau',
 'kandung kemih ketidaknyamanan',
 'urin berbau busuk',
 'rasa air urin yang terus-menerus',
 'passage_of_gases',
 'internal gatal',
 'tampilan toksik (kesalahan ketik)',
 'depresi',
 'sifat lekas marah',
 'nyeri otot',
 'diubah sensorium',
 'bintik merah di atas tubuh',
 'sakit perut.2',
 'abnormal menstruasi',
 'patches dischromic',
 'penyiraman dari mata',
 'peningkatan nafsu makan',
 'poliuria',
 'sejarah keluarga',
 'mukoid dahak',
 'berkarat dahak',
 'kurang konsenterasi',
 'gangguan visual',
 'menerima transfusi darah',
 'menerima suntikan tidak steril',
 'koma',
 'perut pendarahan',
 'distensi perut',
 'riwayat konsumsi alkohol',
 'fluid_overload.1',
 'darah dalam dahak',
 'vena menonjol di betis',
 'jantung berdebar',
 'menyakitkan berjalan',
 'pus filled jerawat',
 'komedo',
 'berlarian',
 'pengelupasan kulit',
 'debu seperti perak',
 'penyok kecil di kuku',
 'kuku inflamasi',
 'lepuh',
 'red sore around nose',
 'kuning kerak ooze',
 'prognosa'],axis=1)
y_test=test_df['prognosa']
print(x_train.columns)

print(len(test_df['prognosa'].unique()))

x_train.shape
x_test.shape
print(x_train[:0])

"""## mengonversi label kelas penyakit menjadi bentuk angka yang sesuai untuk pelatihan model."""

le=LabelEncoder()
dummy=le.fit_transform(train_df['prognosa'])
y_train=pd.DataFrame(dummy)

"""## Memasukkan indeks penyakitnya"""

print(le.classes_)

y_train.head()
print(y_train)

dummy=LabelEncoder().fit_transform(test_df['prognosa'])
y_test=pd.DataFrame(dummy)
y_test.head()
len(y_test)

"""## Pembuatan Model Neural Network (ANN)"""

model=Sequential([
    Dense(units=200,activation='relu'),
    Dense(units=150,activation='relu'),
    Dense(units=100, activation='relu'),
    Dense(units=42,activation='softmax')
])

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy'])
# model.summary()

""" ## Pelatihan Model Neural Network model.fit()"""

history=model.fit(x_train,y_train,epochs=10,batch_size=128,validation_data=(x_test,y_test))

prediction=model.predict(x_test)
prediction[1]

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

pred=np.argmax(prediction, axis=1)
original=y_test[0]
print(original[41])

print(y_test.shape[0])

"""## menghitung confision matrix dengan menggunakan jumlah prediksi yang benar dibagi dengan jumlah total data pengujian."""

confusion = confusion_matrix(original,pred)

fig = px.imshow(confusion, labels=dict(x="Predicted Value", y="Actual Vlaue"),text_auto=True, title='Confusion Matrix')
fig.update_layout(title_x=0.5)
fig.show()

count=0
for i in range(41):
    if(pred[i]-original[i]!=0):
        count+=1

Test_accuracy=(42-count)/42*100
for i in range(10):
    print(f"Predicted: {pred[i]}, Actual: {original[i]}")

print("Test Data Accuracy using ANN=",Test_accuracy,"%")

"""## Model Random Forest"""

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

rf_predictions = clf.predict(x_test)
print(len(rf_predictions))

"""## mengukur performa model Random Forest dengan menghitung mean absolute error (MAE), mean squared error (MSE), dan R-squared pada data pengujian"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, rf_predictions)
mse = mean_squared_error(y_test, rf_predictions)
r2 = r2_score(y_test, rf_predictions)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

count=0
for i in range(41):
    if(rf_predictions[i]-original[i]!=0):
        count+=1

rf_Test_accuracy=(42-count)/42*100

print(f"Random Forest Test Accuracy:",Test_accuracy,'%')

"""## Saving the Model"""

from joblib import dump,load

filename = 'rf_model.joblib'
model = clf
dump(model, filename)

ml_model=load('rf_model.joblib')

inp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# Reshape the 1D array into a 2D array with 1 row and 48 columns
inp = inp.reshape(1,-1)
# Make the prediction
prediksi = ml_model.predict(inp)
print(prediksi)

prediksiPenyakit = le.classes_[prediksi[0]]
print("Penyakit yang diderita:", prediksiPenyakit)
