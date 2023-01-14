import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from NeuralNetwork import Net
from LinearRegression import LR


data = pd.read_csv("SolarPrediction.csv")
X = data.drop(["UNIXTime","Radiation"],axis=1)
Y = pd.DataFrame(data.loc[:,"Radiation"])
X['TSR_Minute'] = pd.to_datetime(X['TimeSunRise']).dt.minute
X['TSS_Minute'] = pd.to_datetime(X['TimeSunSet']).dt.minute
X['TSS_Hour'] = np.where(pd.to_datetime(X['TimeSunSet']).dt.hour==18, 1, 0)
    
X['Month'] = pd.to_datetime(X['Data']).dt.month
X['Day'] = pd.to_datetime(X['Data']).dt.day
X['Hour'] = pd.to_datetime(X['Time']).dt.hour
X['Minute'] = pd.to_datetime(X['Time']).dt.minute
X['Second'] = pd.to_datetime(X['Time']).dt.second
X = X.drop(['Data', 'Time', 'TimeSunRise', 'TimeSunSet'], axis=1)
X['WindDirection(Degrees)_bin'] = np.digitize(X['WindDirection(Degrees)'], np.arange(0.0, 1.0, 0.02).tolist())
X['TSS_Minute_bin'] = np.digitize(X['TSS_Minute'], np.arange(0.0, 288.0, 12).tolist())
X['Humidity_bin'] = np.digitize(X['Humidity'], np.arange(32, 3192, 128).tolist())
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

lr = LR("Batch",0.001,5000)
model = lr.fit(X_train, y_train,X_test,y_test,"Batch")
lr.plot_loss("Batch")

net = Net(iterations=200,learning_rate=0.0001)
model = net.fit(X_train, y_train,X_test,y_test,"SGD",batch_size=128)
RMSE, MSE, MAE = net.evaluate(X_test,y_test)
print("RMSE Error on Test Data: {}".format(RMSE))
print("MSE Error on Test Data: {}".format(MSE))
print("MAE Error on Test Data: {}".format(MAE))
net.plot_loss("SGD")
print("W1:",net.params["W1"])
print("W2:",net.params["W2"])
print("W3:",net.params["W3"])
print("W4:",net.params["W4"])
print("b1:",net.params["b1"])
print("b2:",net.params["b2"])
print("b3:",net.params["b3"])
print("b4:",net.params["b4"])

with open("OptimalWeights.txt",'w') as f:
    f.write("W1: ")
    np.savetxt(f,net.params["W1"])
    f.write("\n")
    f.write("W2: ")
    np.savetxt(f,net.params["W2"])
    f.write("\n")
    f.write("W3: ")
    np.savetxt(f,net.params["W3"])
    f.write("\n")
    f.write("W4: ")
    np.savetxt(f,net.params["W4"])
    f.write("\n")
    f.write("b1: ")
    np.savetxt(f,net.params["b1"])
    f.write("\n")
    f.write("b2: ")
    np.savetxt(f,net.params["b2"])
    f.write("\n")
    f.write("b3: ")
    np.savetxt(f,net.params["b3"])
    f.write("\n")
    f.write("b4: ")
    np.savetxt(f,net.params["b4"])
    f.write("\n")