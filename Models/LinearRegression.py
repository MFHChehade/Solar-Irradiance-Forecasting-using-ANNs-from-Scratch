import numpy as np
import matplotlib.pyplot as plt

class LR():
        
    def __init__(self, optimizer, learning_rate=0.01, iterations=50):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.loss_test = []
        self.loss_train_mae = []
        self.loss_test_mae = []
        self.loss_train_rmse = []
        self.loss_test_rmse = []
        self.X = None
        self.y = None
                
    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(1)
        self.params["W1"] = np.random.randn(self.X.shape[1], 1)
        self.params['b1']  =np.random.randn(1,1)

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
        '''
        Performs the forward propagation
        '''
        if(optimizer == "Batch"):
            yhat = self.X.dot(self.params['W1']) + self.params['b1']
            loss = self.mse_loss(self.y,yhat)
        elif(optimizer == "SGD"):
            yhat = self.X[randomKey,:].dot(self.params['W1']) + self.params['b1']
            loss = self.mse_loss(self.y[randomKey,:],yhat)

        return yhat,loss

    def back_propagation(self,yhat,optimizer="Batch",randomKey=0):
        '''
        Computes the derivatives and update weights and bias according.
        '''
        if(optimizer == "Batch"):
            gradient_wrt_b = (np.sum(yhat-self.y))/(self.X.shape[0])
            gradient_wrt_W = np.dot(self.X.T,yhat-self.y)/(self.X.shape[0])

            # gradient_wrt_b = gradient_wrt_b.clip(min=-6.5,max=6.5)
            # gradient_wrt_W = gradient_wrt_W.clip(min=-6.5,max=6.5)

        elif(optimizer == "SGD"):
            
            gradient_wrt_b = np.sum(yhat-self.y[randomKey,:])/(randomKey.size)
            gradient_wrt_W = np.dot(self.X[randomKey,:].T,yhat-self.y[randomKey,:])/(randomKey.size)

            gradient_wrt_b = gradient_wrt_b.clip(min=-6.5,max=6.5)
            gradient_wrt_W = gradient_wrt_W.clip(min=-6.5,max=6.5)
            
                
        self.params['W1'] = self.params['W1'] - self.learning_rate * gradient_wrt_W
        self.params['b1'] = self.params['b1'] - self.learning_rate * gradient_wrt_b

    def fit(self, X, y,X_test,y_test,optimizer="Batch",batch_size=32):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.X = X
        self.y = y
        self.init_weights()
        
        if(optimizer == "SGD"):
            for i in range(self.iterations):
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
        pred = X.dot(self.params['W1']) + self.params['b1']
        return np.round(pred) 
    
    def evaluate(self, x_test, y_test):
        preds = np.array(self.predict(x_test))
        true_outputs = np.array(y_test)
        return (self.rmse_loss(true_outputs,preds),self.mse_loss(true_outputs,preds), self.mae_loss(true_outputs,preds))


    def plot_loss(self,optimizer):
        '''
        Plots the loss curve
        '''
        plt.subplots(1,3)
        plt.suptitle("Using {} optimizer with a learning rate of {}".format(optimizer,self.learning_rate), fontsize=14)
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
        
    