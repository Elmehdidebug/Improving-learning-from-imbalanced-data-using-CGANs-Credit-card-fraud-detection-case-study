#!/usr/bin/env python
# coding: utf-8
# %%

# %%


from flask import Flask, render_template,request
import plotly
import plotly.graph_objs as go
import plotly.express as px
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import sys
import pickle
np.set_printoptions(threshold=sys.maxsize)
import json
app = Flask(__name__)



# %%


@app.route('/')
def index():
    return render_template('index.html')


# %%


@app.route('/stats')
def stats():
    lableBar = plotBar('Class', {"title": "Transaction Class Distribution", "x_axis": "class", "y_axis": "Frequency"})
    features=randomForestSelectorRanges(data_RUS.drop(['Class'],axis=1) ,data_RUS['Class'] , 2, 29,{"title": "Feature importance", "x_axis": "Number of features", "y_axis": "Metrics"})
    tsnePlot = scatterplot(tsne[tsne['Y_gans'] == 0].values, tsne[tsne['Y_gans'] == 1].values,{"title": "Tsne Plot", "x_axis": "First Component", "y_axis": "Second Component"})
    lableBarh = plotBarh([0.030, 0.060, 0.068, 0.080, 0.090, 0.100, 0.15, 0.200], ['V7' ,'V3' ,'V11' ,'V4' ,'V12' ,'V17' ,'V10' ,'V14'], {"title": "feature importance", "x_axis": "Relative importance", "y_axis": "Features"})
    return render_template('stats.html', plot=lableBar, feature_metrics = features, featureImportance=lableBarh, tsne = tsnePlot)


# %%


@app.route('/result', methods=['POST'])
def result():
    
    """
    features=[float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    """
    features = [request.form.get('v3') ,request.form.get('v4')  ,request.form.get('v7') 
                               ,request.form.get('v10') ,request.form.get('v11') ,request.form.get('v12') 
                               ,request.form.get('v14') ,request.form.get('v17') ]
    final_features = [np.array(features)]
    if model.predict(final_features) == 0:
        proba = round(model.predict_proba(final_features)[0][0]*100, 2)
        pred = "NO Fraud"
    elif model.predict(final_features) == 1:
        proba = round(model.predict_proba(final_features)[0][1]*100, 2)
        pred = "Fraud"
    else :
        proba = "Error"
        pred = "Error"
    return render_template('predict.html', proba=proba, pred=pred)


# %%


@app.route('/predict')
def predict():
    return render_template('predict.html',  proba=None, pred=None, predict=None)


# %%

@app.route('/predict_file', methods=['POST'])
def predict_file():
    transaction= request.files['fichier']
    transac_df= pd.read_csv(transaction)
    
    predict = model.predict(transac_df)
    
    return render_template('predict.html',predict=predict )

# %%

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


# %%





# %%





# %%




