from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
MODEL = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        ipk = float(request.form["IPK"])
        absensi = float(request.form["Jumlah_Absensi"])
        waktu_belajar = float(request.form["Waktu_Belajar_Jam"])
        rasio = float(request.form["Rasio_Absensi"])
        ipk_x_study = float(request.form["IPK_x_Study"])

        data = pd.DataFrame([{
            "IPK": ipk,
            "Jumlah_Absensi": absensi,
            "Waktu_Belajar_Jam": waktu_belajar,
            "Rasio_Absensi": rasio,
            "IPK_x_Study": ipk_x_study
        }])

        pred = MODEL.predict(data)[0]

        # cek apakah model mendukung predict_proba
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(data)[0][int(pred)] * 100
            proba_text = f"{proba:.2f}%"
        else:
            proba_text = "N/A"

        hasil = "Lulus ✅" if pred == 1 else "Tidak Lulus ❌"

        return render_template(
            "index.html",
            prediction_text=f"Hasil Prediksi: {hasil} (Probabilitas: {proba_text})"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
