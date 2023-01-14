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