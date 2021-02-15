import pandas as pd
import numpy as np
import sys
import pickle
np.set_printoptions(threshold=sys.maxsize)
import json
app = Flask(__name__)

from sklearn.metrics import classification_report,accuracy_score, confusion_matrix,f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, balanced_accuracy_score, fbeta_score, recall_score, precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, chi2 , SelectPercentile , f_classif
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
# %%


data = pd.read_csv('creditcard.csv')
data_RUS=pd.read_csv('data_RUS.csv')
tsne= pd.read_csv('tsneData.csv')
# %%


def plotBar(column,layout):
    trace = [
        go.Bar(
            x = ['NonFraud', 'Fraud'],
            y = data[column].value_counts()
        )
    ]
    
    layout = go.Layout(title = layout['title'],
                   xaxis_title=layout['x_axis'],
                   yaxis_title=layout['y_axis'],
                    width=600,
                    height=600)
    fig = go.Figure(data = trace, layout = layout)
    bar = {'trace':trace, 'layout':layout}
    graphJSON = json.dumps(bar, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


# %%


def plotBarh(Xs, Ys, layouts):
    trace = [
        go.Bar(
            x=Xs,
            y=Ys,
            orientation='h')      
    ]
    
    layout = go.Layout(title = layouts['title'],
                   xaxis_title=layouts['x_axis'],
                   yaxis_title=layouts['y_axis'],
                    height=600)
    fig = go.Figure(data = trace, layout = layout)
    bar = {'trace':trace, 'layout':layout}
    graphJSON = json.dumps(bar, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


# %%



def scatterplot(y0, y1,layouts):
    trace = go.Scatter(
            x=y0[:,0],
            y=y0[:,1],
            name='No Fraud',
            mode='markers',
            marker_color='rgba(152, 0, 0, .8)' 
        )
    
    trace2 = go.Scatter(
            x=y1[:,0],
            y=y1[:,1],
            name='Fraud',
            mode='markers',
            marker_color='rgba(12, 150, 150, .8)'  
        )
    
    
    layout = go.Layout(title = layouts['title'],
                   xaxis_title=layouts['x_axis'],
                   yaxis_title=layouts['y_axis'],
                    height=600)
    fig = go.Figure(data = [trace, trace2], layout = layout)
    bar = {'trace':trace, 'layout':layout}
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


# %%


def getheatMap(dataSet, layout):
    layout = go.Layout(title = layout['title'],
                   xaxis_title=layout['x_axis'],
                   yaxis_title=layout['y_axis'],
                    height=1000)
    
    trace = go.Heatmap(
                    z=dataSet.values,
                    x=dataSet.columns,
                    y=dataSet.columns,
                   colorscale='Viridis')
    
    fig = go.Figure(data = trace, layout = layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# %%

def randomForestSelectorRanges(X, y, min, max,layouts):
    model = RandomForestClassifier(random_state=0, max_depth=10).fit(X, y)
    features = X.columns
    importances = model.feature_importances_
    thresholds = []
    praucs = []
    accuracys = []
    recalls = []
    precisions = []
    for i in range(min, max):
        indices = np.argsort(importances)[-i:]
        foreset_variable = [features[i] for i in indices]
        x_train , x_test,y_train, y_test = train_test_split(X[foreset_variable ], y, test_size=0.3, random_state=42, stratify=y)
        clf = LogisticRegression(random_state=0, C=0.1).fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        probs = clf.predict_proba(x_test)
        probs_rfs = probs[:,1]
        precision, recall, _ = precision_recall_curve(y_test, probs_rfs)
        prauc = auc(recall, precision)
        thresholds.append(i)
        praucs.append(prauc)
        accuracys.append(accuracy_score(y_pred, y_test))
        recalls.append(recall_score(y_pred, y_test))
        precisions.append(precision_score(y_pred, y_test))
    trace = go.Scatter(
            x=thresholds,
            y=praucs,
            name='praucs',
            mode='lines+markers',
            marker_color='rgba(152, 0, 0, .8)' 
        )
    
    trace2 = go.Scatter(
            x=thresholds,
            y=accuracys,
            name='accuracy',
            mode='lines+markers',
            marker_color='rgba(12, 150, 150, .8)'  
        )
    trace3 = go.Scatter(
            x=thresholds,
            y=recalls,
            name='recall',
            mode='lines+markers',
            marker_color='rgba(15, 150, 0, .8)' 
        )
    
    trace4 = go.Scatter(
            x=thresholds,
            y=precisions,
            name='precision',
            mode='lines+markers',
            marker_color='rgba(120, 150, 15, .5)'  
        )
    
    
    layout = go.Layout(title = layouts['title'],
                   xaxis_title=layouts['x_axis'],
                   yaxis_title=layouts['y_axis'],
                    height=600)
    fig = go.Figure(data = [trace, trace2,trace3, trace4], layout = layout)
    bar = {'trace':trace, 'layout':layout}
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

# %%

# on prend le meme nombre de données frauduleuses et normales
fraud_df = data.loc[data['Class'] == 1]
non_fraud_df = data.loc[data['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
new_df = normal_distributed_df.sample(frac=1, random_state=42)


# %%


model=pickle.load(open('model.pkl','rb'))
