import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
# from keras.utils import plot_model
# from keras import regularizers
# from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.model_selection import KFold

marrow = pd.read_csv("marrow_revised.csv", header = None)
# marrow_names = pd.read_csv('marrow_revised_names.csv')
 
y = marrow.iloc[0,]
y = y.values
y = np.int64(y)

x = marrow.iloc[1:421,:]
x = x.T
x = x.values
# divide by max of each metabolite (each feature is in [0,1])
x[x == 0] = 1000
x = np.divide(x, np.max(x, axis = 0, keepdims = True))

n = 6

def loocv(X,Y,k, n):
    model=Sequential([
        Dense(3,input_shape=(k,),kernel_initializer='he_uniform',bias_initializer='zeros'),
        Activation('relu'),
        Dense(1,kernel_initializer='he_uniform'),
        Activation('sigmoid'),
    ])

    #sgd=SGD(lr=0.03,momentum=0)
    adm=Adam(lr=0.005,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False)
    model.compile(optimizer=adm,
        loss='binary_crossentropy',
        metrics=['accuracy'])

    probs=np.zeros((X.shape[0],1))
    predictions=np.zeros((X.shape[0],1))

    kf = KFold(n_splits=n, shuffle = True)
    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index], X[test_index]
        train_y = Y[train_index]
        model.fit(train_x,train_y,epochs=2000,verbose=0,shuffle=False)
        probs[test_index] =model.predict(test_x)
        predictions_test=model.predict_classes(test_x)
        predictions[test_index] = predictions_test

    predictions=np.int64(predictions)
    #accuracy
    accuracy=np.mean(np.squeeze(predictions)==Y)


    return probs,predictions,accuracy

k = 1

col_names=[["run"+str(i)+"prob","run"+str(i)+"predictions"] for i in range(1,n+1)]
col_names=[i for l in col_names for i in l]
results=pd.DataFrame(np.zeros((18,len(col_names))),columns=col_names)


# Succinyl + Noise
x_succinyl = x[None, :, 392]
accuracies = []
for i in range(1,n+1):
    probs, predictions, accuracy = loocv(x_succinyl, y, k, n)
    accuracies.append(accuracy)
    results[['run' +str(i) +'prob','run'+str(i)+'predictions' ]] = np.array([probs, predictions]).T
results.to_csv('succinyl_runs.csv')
accuracies = pd.DataFrame(accuracies)
accuracies.to_csv('x_succinyl_accuracies.csv')

# Glutaryl + Noise
x_glutaryl = x[None,:, 395]
accuracies = []
for i in range(1,n+1):
    probs, predictions, accuracy = loocv(x_glutaryl, y, k,n)
    accuracies.append(accuracy)
    results[['run' +str(i) +'prob','run'+str(i)+'predictions' ]] = np.array([probs, predictions]).T
results.to_csv('glutaryl_runs.csv')
accuracies = pd.DataFrame(accuracies)
accuracies.to_csv('x_glutaryl_accuracies.csv')

# Lipoamide + Noise
x_lipoamide = x[None,:, 284]
accuracies = []
for i in range(1,n+1):
    probs, predictions, accuracy = loocv(x_lipoamide, y, k, n)
    accuracies.append(accuracy)
    results[['run' +str(i) +'prob','run'+str(i)+'predictions' ]] = np.array([probs, predictions]).T
results.to_csv('lipoamide_runs.csv')
accuracies = pd.DataFrame(accuracies)
accuracies.to_csv('x_lipoamide_accuracies.csv')
