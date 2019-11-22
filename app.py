import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import math
app = Flask(__name__)
model = pickle.load(open('pima.pickle.dat', 'rb'))
X_test= pd.read_pickle("./index.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    (item_id, store_id)=int_features
    #test with 5320, 34 -> 0.64109534
    i=X_test.query("item_id==@item_id and shop_id==@store_id ").index.values[0]
    '''
    for i in range(len(X_test)):
        if X_test.iloc[i]["item_id"]==item_id and X_test.iloc[i]["shop_id"]==34:
            break
    '''
    prediction = model.predict(X_test.iloc[i:i+1])
    output = math.ceil(prediction[0])

    return render_template('index.html', prediction_text='Sales next month for item: {}, \
        store:{} should be  {}'.format(item_id, store_id, output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)