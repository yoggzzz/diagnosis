from flask import Flask, render_template, request
import joblib,os
import pandas as pd
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
data_train = pd.read_excel("Book1.xlsx")


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(data_train['Gejala'])


app = Flask(__name__)


@app.route('/')
def index():
    data_gejala = pd.DataFrame(data_train)
    data_gejala = data_gejala.drop(['Penyakit'], axis=1)
    data_gejala.drop_duplicates(inplace=True, keep='first', subset=['Gejala'])
    data_gejala.reset_index(drop=True, inplace=True)
    gejala = data_gejala['Gejala'].str.title()

    return render_template('index.html', array=gejala, jumlah=len(gejala))

@app.route('/hasil', methods = ["POST"])
def predict():
    model = load("gigi_clf.pkl")
    data_gejala = pd.DataFrame(data_train)
    data_gejala = data_gejala.drop('Penyakit', axis=1)
    data_gejala.drop_duplicates(inplace=True, keep='first', subset=['Gejala'])
    data_gejala.reset_index(drop=True, inplace=True)
    gejala = data_gejala['Gejala'].str.title()

    array = request.form.getlist('check')
    jml_array = len(array)
    if (jml_array > 0):
        teks = ', '.join([str(elem) for elem in array])
        teks_vector = vectorizer.transform([teks])
        prediction = model.predict(teks_vector)


        # deskripsi
        data_penyakit = pd.DataFrame(data_train)
        newdf = data_penyakit.drop_duplicates(subset = ['Penyakit'], keep = 'last').reset_index(drop = True)
        newdf.drop(['Gejala', 'Kategori', 'No'],axis='columns', inplace=True)
        index = newdf.index[newdf['Penyakit'] == prediction[0]].tolist()
        id = index[0]
        id = id+1
        id = str(id)
        url_penyakit = "https://dwarayoga.site/penyakit.php?id="+id
        deskripsi = requests.get(url_penyakit).json()
        teks_penyakit = deskripsi['deskripsi']

        url_perawatan = "https://dwarayoga.site/perawatan.php?id="+id
        perawatan = requests.get(url_perawatan).json()
        teks_perawatan = perawatan['nama_perawatan']
        return render_template("index.html", hasil_prediksi=prediction[0], array=gejala, jumlah=len(gejala), input_data=array, deskripsi=teks_penyakit, perawatan="Rekomendasi Perawatan : "+teks_perawatan)
    else:
        return render_template("index.html", error = "Tidak ada masukan", array=gejala, jumlah = len(gejala))
        

def load(file):
    load = joblib.load(open(os.path.join(file), "rb"))
    return load

if __name__ == '__main__':
    app.run(debug=True)