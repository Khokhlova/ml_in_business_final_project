import dill
import pandas as pd
import os

dill._dill._reverse_typemap['ClassType'] = type
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    print(model)


model_path = "models/classifier_pipeline.dill"
load_model(model_path)


@app.route("/", methods=["GET"])
def general():
    return """Прогноз выживаемости для пациента с сердечной недостаточностью. Используйте 'http://<address>/predict' чтобы POST"""


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")

    if flask.request.method == "POST":

        age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time = "", "", "", "", "", "", "", "", "", "", "", ""
        request_json = flask.request.get_json()
        if request_json["age"]:
            age = request_json['age']

        if request_json["anaemia"]:
            anaemia = request_json['anaemia']

        if request_json["creatinine_phosphokinase"]:
            creatinine_phosphokinase = request_json['creatinine_phosphokinase']

        if request_json["diabetes"]:
            diabetes = request_json['diabetes']

        if request_json["ejection_fraction"]:
            ejection_fraction = request_json['ejection_fraction']

        if request_json["high_blood_pressure"]:
            high_blood_pressure = request_json['high_blood_pressure']

        if request_json["platelets"]:
            platelets = request_json['platelets']

        if request_json["serum_creatinine"]:
            serum_creatinine = request_json['serum_creatinine']

        if request_json["serum_sodium"]:
            serum_sodium = request_json['serum_sodium']

        if request_json["sex"]:
            sex = request_json['sex']

        if request_json["smoking"]:
            smoking = request_json['smoking']

        if request_json["time"]:
            time = request_json['time']
        logger.info(
            f'{dt} Data: age={age}, anaemia={anaemia}, creatinine_phosphokinase={creatinine_phosphokinase}, diabetes={diabetes}, ejection_fraction={ejection_fraction}, high_blood_pressure={high_blood_pressure}, platelets={platelets}, serum_creatinine={serum_creatinine}, serum_sodium={serum_sodium}, sex={sex}, smoking={smoking}, time={time}')
        try:
            preds = model.predict_proba(pd.DataFrame({"age": [age],
                                                      "anaemia": [anaemia],
                                                      "creatinine_phosphokinase": [creatinine_phosphokinase],
                                                      "diabetes": [diabetes],
                                                      "ejection_fraction": [ejection_fraction],
                                                      "high_blood_pressure": [high_blood_pressure],
                                                      "platelets": [platelets],
                                                      "serum_creatinine": [serum_creatinine],
                                                      "serum_sodium": [serum_sodium],
                                                      "sex": [sex],
                                                      "smoking": [smoking],
                                                      "time": [time]}))
        except AttributeError as e:
            logger.warning(f'{dt} Exception: {str(e)}')
            data['predictions'] = str(e)
            data['success'] = False
            return flask.jsonify(data)

        data["predictions"] = preds[:, 1][0]
        data["success"] = True

    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
