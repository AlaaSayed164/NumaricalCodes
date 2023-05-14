# Numerical Codes
**optimisation Meaning:** <br/>
Is the process where we train the model iteratively that results in a maximum and minimum function evaluation.  
**Why do we optimize our machine learning models?** <br/> 
We compare the results in every iteration by changing the hyperparameters in each step until we reach the optimum results. We create an accurate model with less error rate. <br/> 
**MAXIMA AND MINIMA** <br/> <br/>
![image](https://github.com/AlaaSayed164/NumaricalCodes/assets/101005712/189eee9a-c3bd-4c65-b1c9-dca68c30993c) <br/> <br/>
There can be only one global minima and maxima but there can be more than one local minima and maxima.<br/><br/>
**GRADIENT DESCENT** <br/>
- Gradient Descent is an optimization algorithm and it finds out the local minima of a differentiable function. It is a minimization algorithm that minimizes a given function.<br/>
- first-order optimization algorithm as it explicitly makes use of the first-order derivative of the target objective function.<br/><br/>
![image](https://github.com/AlaaSayed164/NumaricalCodes/assets/101005712/e1bbbf80-0916-4f60-97ba-8ef68de459be) <br/> <br/>
Let’s take an example graph of a parabola, Y=X² <br/>
Here, the minima is the origin(0, 0). The slope here is Tanθ. So the slope on the right side is positive as 0<θ<90 and its Tanθ is a positive value. The slope on the left side is negative as 90<θ<180 and its Tanθ is a negative value.<br/><br/>
![image](https://github.com/AlaaSayed164/NumaricalCodes/assets/101005712/b85905a6-68b5-4dd4-a31a-f205f29f69dc) <br/><br/>
Notes:<br/>
- One important observation in the graph is that the slope changes its sign from positive to negative at minima. As we move closer to the minima, the slope reduces.<br/>
- Learning Rate is a hyperparameter or tuning parameter that determines the step size at each iteration while moving towards minima in the function. 


////////////////////////////////////////////////////////
#### Is the process of iteratively improving the accuracy of a machine learning model, lowering the degree of error. <\b> . A major goal of training a machine learning algorithm is to minimise the degree of error between the predicted output and the true output. 
Machine learning models learn to generalise and make predictions about new live data based on insight learned from training data. This works by approximating the underlying function or relationship between input and output data. 
  
Optimisation is measured through a loss or cost function, which is typically a way of defining the difference between the predicted and actual value of data. Machine learning models aim to minimise this loss function, or lower the gap between prediction and reality of output data. Iterative optimisation will mean that the machine learning model becomes more accurate at predicting an outcome or classifying data.  
///////////////////////////////////////////////////////
Momentum is widely used in the machine learning community for optimizing non-convex functions such as deep neural networks.[6] Empirically, momentum methods outperform traditional stochastic gradient descent approaches.
n deep learning, SGD is widely prevalent and is the underlying basis for many optimizers such as Adam, Adadelta, RMSProp, etc. which already utilize momentum to reduce computation speed.[9] The momentum extension for optimization algorithms is available in many popular machine learning frameworks such as PyTorch, tensor flow, and scikit-learn. 
  
 
  Momentum is an extension to the gradient descent optimization algorithm that builds inertia in a search direction to overcome local minima and oscillation of noisy gradients.[1] It is based on the same concept of momentum in physics.
  ////////////////////////////////////////////////////

Numerical  optimization 

[1.implement the gradient descent for linear regressionwith one variable 
](https://github.com/AlaaSayed164/NumaricalCodes/blob/main/Practical%20Session%201%20GD%20Implementation%20for%20LR%20.ipynb)

[2.implement the gradient descent variants (Batch/Mini-Batch/Stochastic) for linear regression with one variable 
](https://github.com/AlaaSayed164/NumaricalCodes/blob/main/Practical%20Session%202%20GD%20Variants%20Batch%20-%20Mini-Batch%20-%20Stochastic.ipynb)

[3.implement the accelerated gradient descent methods (Momentum and NAG) for linear regression with one variable 
](https://github.com/AlaaSayed164/NumaricalCodes/blob/main/Practical%20Session%203%20Momentum%20-%20NAG%20(1).ipynb)

[4.implement the accelerated gradient descent methods with adaptive learning rate (Adagrad, RMSProp, and Adam) for linear regression with one variable 
](https://github.com/AlaaSayed164/NumaricalCodes/blob/main/Practical%20Session%204%20Adagrad-RMSProp-Adam%20(1).ipynb)

