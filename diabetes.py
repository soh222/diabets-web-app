import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from flask import Flask, render_template
from tensorflow import keras

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

bootstrap5 = Bootstrap5(app)

class LabForm(FlaskForm):
    preg = StringField('# Pregnancies', validators=[DataRequired()])
    glucose = StringField('Glucose', validators=[DataRequired()])
    blood = StringField('Blood pressure', validators=[DataRequired()])
    skin = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin', validators=[DataRequired()])
    bmi = StringField('BMI', validators=[DataRequired()])
    dpf = StringField('DPF Score', validators=[DataRequired()])
    age = StringField('Age', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        # 환자 데이터를 가져와서 테스트를 위한 X_test 폼에 넣습니다
        X_test = np.array([[float(form.preg.data),
                            float(form.glucose.data),
                            float(form.blood.data),
                            float(form.skin.data),
                            float(form.insulin.data),
                            float(form.bmi.data),
                            float(form.dpf.data),
                            float(form.age.data)]])
        print(X_test.shape)
        print(X_test)

        # 당뇨병 데이터를 읽어옵니다
        data = pd.read_csv('./diabetes.csv', sep=',')

        # 가져온 데이터에서 X와 y를 추출합니다
        X = data.values[:, 0:8]
        y = data.values[:, 8]

        # MinMaxScaler를 사용하여 스케일러 객체를 맞춥니다(fit)
        scaler = MinMaxScaler()
        scaler.fit(X)

        # 예측을 위해 데이터를 최소-최대 스케일링(min max scale) 합니다
        X_test = scaler.transform(X_test)

        # 모델 로드
        # 주의: keras가 import되어 있어야 합니다. (앞선 코드 참고)
        from tensorflow import keras
        model = keras.models.load_model('pima_model.keras')

        # 모델 평가 (예측 수행)
        prediction = model.predict(X_test)
        res = prediction[0][0]
        res = np.round(res, 2)
        res = (float)(np.round(res * 100))

        return render_template('result.html', res=res)
    
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run()