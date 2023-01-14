import numpy as np
import matplotlib.pyplot as plt

class Net():
        
    def __init__(self, layers=[16,64,48,32,1], learning_rate=0.01, iterations=50):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.loss_test = []
        self.loss_train_mae = []
        self.loss_test_mae = []
        self.loss_train_rmse = []
        self.loss_test_rmse = []
        self.layers = layers
        self.X = None
        self.y = None
                
    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(1)
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1]) * np.sqrt(2/self.layers[0])
        self.params['b1']  =np.random.randn(self.layers[1],)* np.sqrt(2/self.layers[0])
        self.params['W2'] = np.random.randn(self.layers[1],self.layers[2]) * np.sqrt(2/self.layers[1])
        self.params['b2'] = np.random.randn(self.layers[2],) * np.sqrt(2/self.layers[1])
        self.params["W3"] = np.random.randn(self.layers[2], self.layers[3]) * np.sqrt(2/self.layers[2])
        self.params['b3']  =np.random.randn(self.layers[3],)* np.sqrt(2/self.layers[2])
        self.params['W4'] = np.random.randn(self.layers[3],self.layers[4]) * np.sqrt(2/self.layers[3])
        self.params['b4'] = np.random.randn(self.layers[4],)* np.sqrt(2/self.layers[3])
    
    def relu(self,Z):
        '''
        The ReLu activation function is to performs a threshold
        operation to each input element where values less 
        than zero are set to zero.
        '''
        return np.maximum(0,Z)

    def dRelu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def mse_loss(self,y, yhat):

        loss = np.mean(np.power(yhat-y, 2));
        return loss
    def rmse_loss(self,y, yhat):
    
        loss = np.sqrt(np.mean(np.power(yhat-y, 2)))
        return loss
    def mae_loss(self,y, yhat):
        
        loss = np.mean(np.abs(yhat-y))
        return loss
    def forward_propagation(self,optimizer="Batch",randomKey=0):
        if(optimizer == "Batch"):
            Z1 = self.X.dot(self.params['W1']) + self.params['b1']
            A1 = self.relu(Z1)
            Z2 = A1.dot(self.params['W2']) + self.params['b2']
            A2 = self.relu(Z2)
            Z3 = A2.dot(self.params['W3']) + self.params['b3']
            A3 = self.relu(Z3)
            Z4 = A3.dot(self.params['W4']) + self.params['b4']
            yhat = self.relu(Z4)
            loss = self.mse_loss(self.y,yhat)
        elif(optimizer == "SGD"):
            Z1 = self.X[randomKey,:].dot(self.params['W1']) + self.params['b1']
            A1 = self.relu(Z1)
            Z2 = A1.dot(self.params['W2']) + self.params['b2']
            A2 = self.relu(Z2)
            Z3 = A2.dot(self.params['W3']) + self.params['b3']
            A3 = self.relu(Z3)
            Z4 = A3.dot(self.params['W4']) + self.params['b4']
            yhat = self.relu(Z4)
            loss = self.mse_loss(self.y[randomKey,:],yhat)
    
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['Z3'] = Z3
        self.params['Z4'] = Z4
        self.params['A1'] = A1
        self.params['A2'] = A2
        self.params['A3'] = A3

        return yhat,loss

    def back_propagation(self,yhat,optimizer="Batch",randomKey=0):
        '''
        Computes the derivatives and update weights and bias according.
        '''
        if(optimizer == "Batch"):
            dl_wrt_yhat = (yhat-self.y)
            dl_wrt_z4 = dl_wrt_yhat * self.dRelu(self.params["Z4"])
            dl_wrt_w4 = self.params['A3'].T.dot(dl_wrt_z4)
            dl_wrt_b4 = np.sum(dl_wrt_z4, axis=0, keepdims=True)

            dl_wrt_A3 = dl_wrt_z4.dot(self.params['W4'].T)
            dl_wrt_z3 = dl_wrt_A3 * self.dRelu(self.params['Z3'])
            dl_wrt_w3 = self.params['A2'].T.dot(dl_wrt_z3)
            dl_wrt_b3 = np.sum(dl_wrt_z3, axis=0, keepdims=True)
            
            dl_wrt_A2 = dl_wrt_z3.dot(self.params['W3'].T)
            dl_wrt_z2 = dl_wrt_A2 * self.dRelu(self.params['Z2'])
            dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
            dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)
            
            dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
            dl_wrt_z1 = dl_wrt_A1 * self.dRelu(self.params['Z1'])
            dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
            dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

            dl_wrt_w1 = dl_wrt_w1.clip(min=-6.5,max=6.5)
            dl_wrt_w2 = dl_wrt_w2.clip(min=-6.5,max=6.5)
            dl_wrt_w3 = dl_wrt_w3.clip(min=-6.5,max=6.5)
            dl_wrt_w4 = dl_wrt_w4.clip(min=-6.5,max=6.5)
            dl_wrt_b1 = dl_wrt_b1.clip(min=-6.5,max=6.5)
            dl_wrt_b2 = dl_wrt_b2.clip(min=-6.5,max=6.5)
            dl_wrt_b3 = dl_wrt_b3.clip(min=-6.5,max=6.5)
            dl_wrt_b4 = dl_wrt_b4.clip(min=-6.5,max=6.5)
            
        elif(optimizer == "SGD"):
            
            dl_wrt_yhat = (yhat-self.y[randomKey,:])
            dl_wrt_z4 = dl_wrt_yhat * self.dRelu(self.params["Z4"])
            dl_wrt_w4 = self.params['A3'].T.dot(dl_wrt_z4)
            dl_wrt_b4 = np.sum(dl_wrt_z4, axis=0, keepdims=True)

            dl_wrt_A3 = dl_wrt_z4.dot(self.params['W4'].T)
            dl_wrt_z3 = dl_wrt_A3 * self.dRelu(self.params['Z3'])
            dl_wrt_w3 = self.params['A2'].T.dot(dl_wrt_z3)
            dl_wrt_b3 = np.sum(dl_wrt_z3, axis=0, keepdims=True)
            
            dl_wrt_A2 = dl_wrt_z3.dot(self.params['W3'].T)
            dl_wrt_z2 = dl_wrt_A2 * self.dRelu(self.params['Z2'])
            dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
            dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)
            
            dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
            dl_wrt_z1 = dl_wrt_A1 * self.dRelu(self.params['Z1'])
            dl_wrt_w1 = self.X[randomKey,:].T.dot(dl_wrt_z1)
            dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

            dl_wrt_w1 = dl_wrt_w1.clip(min=-6.5,max=6.5)
            dl_wrt_w2 = dl_wrt_w2.clip(min=-6.5,max=6.5)
            dl_wrt_w3 = dl_wrt_w3.clip(min=-6.5,max=6.5)
            dl_wrt_w4 = dl_wrt_w4.clip(min=-6.5,max=6.5)
            dl_wrt_b1 = dl_wrt_b1.clip(min=-6.5,max=6.5)
            dl_wrt_b2 = dl_wrt_b2.clip(min=-6.5,max=6.5)
            dl_wrt_b3 = dl_wrt_b3.clip(min=-6.5,max=6.5)
            dl_wrt_b4 = dl_wrt_b4.clip(min=-6.5,max=6.5)
            
                
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['W3'] = self.params['W3'] - self.learning_rate * dl_wrt_w3
        self.params['W4'] = self.params['W4'] - self.learning_rate * dl_wrt_w4
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2
        self.params['b3'] = self.params['b3'] - self.learning_rate * dl_wrt_b3
        self.params['b4'] = self.params['b4'] - self.learning_rate * dl_wrt_b4

    def fit(self, X, y,X_test,y_test,optimizer="Batch",batch_size=32):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.X = X
        self.y = y
        self.init_weights()
        
        if(optimizer == "SGD"):
            for i in range(self.iterations):
                if((i+1)%50 == 0):
                    self.learning_rate/=2
                for _ in range(self.X.shape[0]//batch_size):
                    randomKey = np.random.randint(0,self.X.shape[0],size=batch_size)
                    yhat, loss = self.forward_propagation("SGD",randomKey)
                    self.back_propagation(yhat,"SGD",randomKey)
                yhat, loss = self.forward_propagation()
                self.back_propagation(yhat)
                self.loss.append(loss)
                preds_train = np.array(self.predict(self.X))
                true_outputs_train = np.array(self.y)
                loss_train_mae = self.mae_loss(true_outputs_train,preds_train)
                loss_train_rmse = self.rmse_loss(true_outputs_train,preds_train)
                self.loss_train_mae.append(loss_train_mae)
                self.loss_train_rmse.append(loss_train_rmse)
                preds = np.array(self.predict(X_test))
                true_outputs = np.array(y_test)
                loss_test = self.mse_loss(true_outputs,preds)
                loss_test_mae = self.mae_loss(true_outputs,preds)
                loss_test_rmse = self.rmse_loss(true_outputs,preds)
                self.loss_test_mae.append(loss_test_mae)
                self.loss_test_rmse.append(loss_test_rmse)
                self.loss_test.append(loss_test)
                print("Epoch {}: MSE Training Loss {} - MSE Testing Loss {} - MAE Testing Loss {} - RMSE Testing Loss {}".format(i+1, loss,loss_test,loss_test_mae,loss_test_rmse))
        else:
            for i in range(self.iterations):
                if((i+1)%100==0):
                    if((i+1) <= 400):
                        self.learning_rate /=2
                yhat, loss = self.forward_propagation()
                self.back_propagation(yhat)
                self.loss.append(loss)
                preds_train = np.array(self.predict(self.X))
                true_outputs_train = np.array(self.y)
                loss_train_mae = self.mae_loss(true_outputs_train,preds_train)
                loss_train_rmse = self.rmse_loss(true_outputs_train,preds_train)
                self.loss_train_mae.append(loss_train_mae)
                self.loss_train_rmse.append(loss_train_rmse)
                preds = np.array(self.predict(X_test))
                true_outputs = np.array(y_test)
                loss_test = self.mse_loss(true_outputs,preds)
                loss_test_mae = self.mae_loss(true_outputs,preds)
                loss_test_rmse = self.rmse_loss(true_outputs,preds)
                self.loss_test_mae.append(loss_test_mae)
                self.loss_test_rmse.append(loss_test_rmse)
                self.loss_test.append(loss_test)
                print("Epoch {}: MSE Training Loss {} - MSE Testing Loss {} - MAE Testing Loss {} - RMSE Testing Loss {}".format(i+1, loss,loss_test,loss_test_mae,loss_test_rmse))

    def predict(self, X):
        '''
        Predicts on a test data
        '''
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.relu(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        A3 = self.relu(Z3)
        Z4 = A3.dot(self.params['W4']) + self.params['b4']
        pred = self.relu(Z4)
        return np.round(pred) 
    
    def evaluate(self, x_test, y_test):
        preds = np.array(self.predict(x_test))
        true_outputs = np.array(y_test)
        return (self.rmse_loss(true_outputs,preds),self.mse_loss(true_outputs,preds), self.mae_loss(true_outputs,preds))


    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.subplots(1,3)
        plt.subplot(1,3,1)
        plt.plot(self.loss,label='Training Loss')
        plt.plot(self.loss_test,label='Testing Loss')
        plt.legend(loc='best')
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training and Testing MSE Loss")
        plt.subplot(1,3,2)
        plt.plot(self.loss_train_mae,label='Training Loss')
        plt.plot(self.loss_test_mae,label='Testing Loss')
        plt.legend(loc='best')
        plt.xlabel("Epoch")
        plt.ylabel("MAE Loss")
        plt.title("Training and Testing MAE Loss")
        plt.subplot(1,3,3)
        plt.plot(self.loss_train_rmse,label='Training Loss')
        plt.plot(self.loss_test_rmse,label='Testing Loss')
        plt.legend(loc='best')
        plt.xlabel("Epoch")
        plt.ylabel("RMSE Loss")
        plt.title("Training and Testing RMSE Loss")
        plt.show()    
        
    