from flask import Flask, request, render_template
import numpy as np

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_weight', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('predict_weight.html')
    else:
        data = CustomData(
            GAINED=request.form.get('gained'),
            VISITS=request.form.get('visits'),
            MAGE=request.form.get('mage'),
            FAGE=request.form.get('fage'),
            TOTALP=request.form.get('totalp'),
            BDEAD=request.form.get('bdead'),
            TERMS=request.form.get('terms'),
            WEEKS=request.form.get('weeks'),
            CIGNUM=request.form.get('cignum'),
            DRINKNUM=request.form.get('drinknum'),
            SEX=request.form.get('sex'),
            MARITAL=request.form.get('marital'),
            RACEMOM=request.form.get('racemom'),
            RACEDAD=request.form.get('raedad'),
            HISPMOM=request.form.get('hispmom'),
            HISPDAD=request.form.get('hispdad'),
            ANEMIA=request.form.get('anemia'),
            CARDIAC=request.form.get('cardiac'),
            ACLUNG=request.form.get('aclung'),
            DIABETES=request.form.get('diabetes'),
            HERPES=request.form.get('hepes'),
            HYDRAM=request.form.get('hydra'),
            HEMOGLOB=request.form.get('hemoglob'),
            HYPERCH=request.form.get('hyperch'),
            HYPERPR=request.form.get('hyperpr'),
            ECLAMP=request.form.get('eclamp'),
            CERVIX=request.form.get('cervix'),
            PINFANT=request.form.get('pinfant'),
            PRETERM=request.form.get('preterm'),
            RENAL=request.form.get('renal'),
            RHSEN=request.form.get('rhsen'),
            UTERINE=request.form.get('uterine'),
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        score = round(float(results[0]), 1) if isinstance(results, (list, np.ndarray)) else round(float(results), 1)

        return render_template('predict_weight.html', results=score)

    return render_template("index.html", results=None)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
