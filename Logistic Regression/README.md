![MEME](https://github.com/maciejbalawejder/MLalgorithms-collection/blob/main/Logistic%20Regression/i-know-logistic-regression-show-me.jpg)
# Logistic Regression

Model of Logistic Regression bulit from scratch in educational purpose which tries to be small and clean. It is not complicated model, containing only 115 lines of code, and explaining in depth operation which are going "behind the scenes" when you use libraries like sklearn. Two init arguments are taken, iterations and learning rate. The rest of the code is explained in depth below. 

-----------------------------------------------------------------------------------------------------------------------------------------------------
### Main functions : 

   #### 1) Sigmoid - 1/(1+e^(-input•weights)) - the function is squashing values and keep them between(0,1), it never reaches 0 or 1, which solves issue in Cross-Entropy ln(0) 
![Sigmoid](https://github.com/maciejbalawejder/Logistic_Regression/blob/main/sigmoid.png)


#### 2) Split - takes two arguments dataset and seprator


#### 3) Cost - Cross-Entropy function  C=−1n∑x[yln(a)+(1−y)ln(1−a)] | a = input•weights, for traning set and validation set
Yeah, but why cross-entropy?\
The bigger error between predicted output and ground truth the more it affects the loss function(more than MSE cause it is logarithmic) and forces to update weights to achive opposite result.


#### 4) Update - updates the weights W[i] = W[i-1] - learning_rate * dJ/dW


#### 5) Predict - gives a predicted values based on trained weights and given inputs(X) - OUTPUT = X • Weights. Default thershold value is 0.5.
        
##### Example 
        '''
        
        model = LogisticRegression()
        predictions = model.predict(X_test)
        
        '''
#### 6) Fit - takes arguments X - inputs, and y - ground truth. For number of iterations, the loss function and updating weights is executed on given X,y. The regularization parameter can be change to True, to use regularization, and validation procent can be change by changing validation float value.  

##### Example
        '''
        
        model = LogisticRegression()
        model.fit(X_train,Y_train,regularization=False,validation=0.1) # regularization=False,validation=0.1 these ones are set by default
        
        '''
### Analyzing tools

#### Confusion matrix - table allowing to __see__ the performance of the model and also calculate parameters such as: 
![Confusion matrix](https://github.com/maciejbalawejder/Logistic_Regression/blob/main/confusion-matrix.png)
##### Precision - out of all the positive classes we have predicted correctly, how many are actually positive
##### Recall - out of all the positive classes, how much we predicted correctly
##### F1-score - in the model when there is no sepecific goal about precision or recall, it is easier to use combination of both
##### Accuracy - pretty straightforward, how accurate is model on testing data
##### Learning plot - function Loss(iterations) for validation and testing set to performance of the model and detect possible overfitting or underfitting problem  


### Fixed :
* confusion matrix
* cost
* updating weights 

### To do :
* scalling data
* optimizing model
  * [x] regularization - improves generalization 
  * [x] validation dataset
  * [x] split function 
  * [x] plot learning curve
  * [x] F1 - score 
  * k-fold validation set sounds cool
 

