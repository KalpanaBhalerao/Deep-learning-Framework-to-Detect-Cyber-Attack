#importing libraries

import pickle
from fastapi.responses import StreamingResponse
import joblib
import numpy as np
import pandas as pd
from joblib import dump, load
import sklearn
import sklearn.preprocessing
from sklearn import metrics
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
import sys
import os
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D 
from utilities import  ConvertToLinearOutput, SequentialModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from fastapi import FastAPI, File, HTTPException, Body, UploadFile

globalModel = any

app = FastAPI()

#Function to min-max normalize
def normalize(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy()  # do not touch the original df
    for feature_name in cols:
        if df[feature_name].dtype in ['int64', 'float64']:  # Check if the column is numerical
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            if max_value > min_value:
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

#One-hot encoding
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop([each], axis=1)  
    return df
columns = ['protocol_type','service','flag']



confusionMatrixPath = "analytics/confusionMatrix.png"

import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(confusionMatrixPath)


trainColumns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'subclass']

testColumns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'subclass']

@app.post("/Upload_DataSets")
async def Upload_DataSets(trainingFile:UploadFile = File(...) , testingFile:UploadFile = File(...)):
    with open("train.csv","wb") as f:
        content = await trainingFile.read()
        f.write(content) 
    with open("test.csv","wb") as f:
        content = await testingFile.read()
        f.write(content)
    return {'message' : "Files Uploaded Successfully"}
    

@app.post("/Train_CNN_Model")
async def Train_CNN_Model():
    global trainColumns
    global testColumns
    global columns
    global globalModel

    df_train = pd.read_csv('train.csv')
    df_train.columns = trainColumns
    
    df_train = one_hot(df_train,columns)
    tmp = df_train.pop('subclass') 
    new_df_train = normalize(df_train,df_train.columns)
    new_df_train['class'] = tmp
    y_train = new_df_train['class']
    df_test_X = new_df_train.drop('class', axis=1)
    
    train_X, test_X, train_y, test_y = train_test_split(df_test_X, y_train, test_size=0.2, shuffle=True , random_state=42)
    
    Model = SequentialModel(layers=
        {Convolution1D(64, kernel_size=122, activation="relu",input_shape=(122, 1)),
         MaxPooling1D(5,padding='same'),
         BatchNormalization(),
         Flatten(),
         Dropout(0.5),
         Dense(5),
         Activation('softmax') }) 
    Model.fit(train_X, train_y)
    pred = Model.predict(test_X)
    score = metrics.accuracy_score(test_y.values, pred)  # Evaluate accuracy
    with open('CNNmodel.pkl', 'wb') as model_file:
        pickle.dump({"model": Model}, model_file)

    cm = confusion_matrix(test_y.values,pred)
    plot_confusion_matrix(cm, normalize    = False, target_names = list(AttackEncodings.values()),
                      title        = "Confusion Matrix")
    return {"message":"The Model is trained ","accuracyScore":score}


AttackEncodings = {'processtable': 1, 'land': 2, 'neptune': 3, 'satan': 4, 'warezmaster': 5, 'back': 6, 'buffer_overflow': 7, 'snmpgetattack': 8, 'warezclient': 9, 'teardrop': 10, 'mailbomb': 11, 'normal': 12, 'multihop': 13, 'ps': 14, 'httptunnel': 15, 'imap': 16, 'xsnoop': 17, 'rootkit': 18, 'loadmodule': 19, 'portsweep': 20, 'pod': 21, 'perl': 22, 'nmap': 23, 'guess_passwd': 24, 'spy': 25, 'ftp_write': 26, 'ipsweep': 27, 'snmpguess': 28, 'xlock': 29, 'smurf': 30, 'saint': 31, 'apache2': 32, 'mscan': 33}

@app.get("/download-model")
async def download_model():
    model_path = "CNNmodel.pkl"
    
    # Check if the file exists
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # Use StreamingResponse to efficiently serve the file
    return StreamingResponse(open(model_path, "rb"), media_type="application/octet-stream", headers={"Content-Disposition": f"attachment;filename={model_path}"})

@app.post("/ParseCSVString")
async def ParseCSVString(csvString :str):
    data_dict = dict(zip(trainColumns, csvString.split(',')))
    df = pd.DataFrame([data_dict])
    if df["subclass"].isnull().any() or df["subclass"].iloc[0] == "" :
        raise Exception("the subclass can not be null or empty")
    return df

@app.post("/Test_CNN_Model")
async def Test_CNN_Model(csvString :str):
    global trainColumns
    global testColumns
    global columns
    df_test = await ParseCSVString(csvString)
    shape = df_test.shape
    print(shape)
    df_batch = pd.read_csv('train.csv')
    df_batch.columns = trainColumns
    df_test.columns = trainColumns 
    df_test = pd.concat([df_test ,df_batch],axis=0)
    df_test = one_hot(df_test,columns)
    tmp = df_test.pop('subclass') 
    new_df_test = normalize(df_test,df_test.columns)
    new_df_test['class'] = tmp
    finalTestDf = new_df_test.iloc[:1]
    
    finalTrainDf = new_df_test.iloc[shape[0]+1:new_df_test.shape[0]]
    y_train = finalTrainDf['class'] # dependant var 
    df_test_X = finalTrainDf.drop('class', axis=1) # Independant var 
    actual = finalTestDf['class'].values
    finalTestDf = finalTestDf.drop('class', axis=1)
    print(finalTestDf.head())
    with open('CNNmodel.pkl', 'rb') as model_file:
        loaded_data = pickle.load(model_file)
        Model = loaded_data["model"]
        
        pred = ConvertToLinearOutput(Model.predict(finalTestDf.values),actual[0])
        
    return {"Predicted": pred, "Actual":actual[0] }

if __name__ == "__main__":
    import uvicorn
    print("Server Running At : http://localhost:8000/docs")
    uvicorn.run(app, host="localhost", port=8000)

