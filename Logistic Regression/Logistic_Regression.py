import numpy as np
class LogisticRegression():
    def __init__(self,iterations=100,alpha=0.0115):
        self.iterations = iterations
        self.alpha = alpha
        self.J = []
        self.J_val = []
        self.output = 0
        self.o_val = 0
        self.regularization = False

    def sigmoid(self,a):
        return 1/(1+np.exp(-a))

    def split(self,dataset,value):
        part1 = dataset[:int(len(dataset)*(1-value))]
        part2 = dataset[len(part1):]
        return part1, part2

    def fit(self,X,y,regularization = False,validation = 0.1):
        self.validation = validation
        self.regularization = regularization

        self.X , self.x_val = self.split(X,self.validation)
        self.y , self.y_val = self.split(y.reshape(y.shape[0],1),self.validation)

        self.m = self.X.shape[1]
        self.W = np.random.randn(self.m,1)

        self.m_val = self.x_val.shape[1]
        self.W_val = np.random.randn(self.m_val,1)

        assert self.X.shape[0] == self.y.shape[0] and self.x_val.shape[0] == self.y_val.shape[0]

        for i in range(self.iterations):
            a = self.X.dot(self.W)
            a_val = self.x_val.dot(self.W_val)

            self.output = self.sigmoid(a)
            self.o_val = self.sigmoid(a_val)

            assert self.output.shape == self.y.shape and self.o_val.shape == self.y_val.shape

            self.cost()
            self.update()

    def cost(self,Lambda=0):
        if self.regularization is False:
            J = sum(self.y * np.log(self.output) + (1-self.y)*np.log(1-self.output))
            self.J.append(*(1/-self.m)*J)

            # Validation

            J_val = sum(self.y_val * np.log(self.o_val) + (1-self.y_val)*np.log(1-self.o_val))
            self.J_val.append(*(1/-self.m_val)*J_val)

        else :
            J = sum(self.y * np.log(self.output) + (1-self.y)*np.log(1-self.output)) + Lambda * sum(np.power(W,2))
            self.J.append(*(1/-self.m)*J)

            # Validation

            J_val = sum(self.y_val * np.log(self.o_val) + (1-self.y_val)*np.log(1-self.o_val)) + Lambda * sum(np.power(self.W_val,2))
            self.J_val.append(*(1/-self.m_val)*J)


    def update(self):
        dw = np.dot(self.X.T,self.output-self.y)
        self.W = self.W - (self.alpha * dw)

        # Validation

        dw_val = np.dot(self.x_val.T, self.o_val - self.y_val)
        self.W_val = self.W_val - (self.alpha * dw_val)

    def predict(self,x,threshold=0.5):
        prediction = self.sigmoid(np.dot(x,self.W))
        prediction[prediction>threshold] = 1
        prediction[prediction<=threshold] = 0
        return prediction

    def confusion_matrix(self,predictions,y):
        assert predictions.shape[0] == y.shape[0]
        self.TP, self.TN, self.FN, self.FP = 0,0,0,0

        for i in range(len(predictions)):
            if predictions[i]==1 and y[i]==1:self.TP+=1
            elif predictions[i]==1 and y[i]==0:self.FP+=1
            elif predictions[i]==0 and y[i]==1:self.FN+=1
            elif predictions[i]==0 and y[i]==0:self.TN+=1


    def precision(self):
        return self.TP/(self.TP+self.FP)

    def recall(self):
        return self.TP/(self.TP+self.FN)

    def accuracy(self):
        return (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)

    def F1_score(self):
        return (2*self.precision()*self.recall())/(self.precision() + self.recall())

    def plot(self):
        x = np.linspace(0,self.iterations,self.iterations)
        y = np.asarray(self.J)
        y_val = np.asarray(self.J_val)
        assert x.shape == y.shape and x.shape == y_val.shape
        plt.plot(x,y)
        plt.plot(x,y_val)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
