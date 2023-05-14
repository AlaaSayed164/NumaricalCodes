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
- Steps: <br/>
          1.  Compute the slope (gradient) that is the first-order derivative of the function at the current point <br/>
          2.Move-in the opposite direction of the slope increase from the current point by the computed amount <br/><br/>
**Batch /Stochastic / Gradient Descent**
![image](https://github.com/AlaaSayed164/NumaricalCodes/assets/101005712/fe523030-c478-4365-b1ba-542cf587f339)
![image](https://github.com/AlaaSayed164/NumaricalCodes/assets/101005712/8a320019-d7aa-44a3-b4c3-409c3948176d)
![image](https://github.com/AlaaSayed164/NumaricalCodes/assets/101005712/f9666349-7a7d-4c7d-a1a7-ccda4b4b927c) <br/><br/>
**Advantages of Batch Gradient Descent**
- Computationally Efficient: As you may have guessed, this technique is less computationally demanding, as no updates are required after each sample.
- Stable Convergence: Another advantage is the fact that the convergence of the weights to the optimal weights is very stable. By calculating and averaging all individual gradients over each sample in the dataset, we get a very good estimate of the true gradient, indicating the highest increase of the loss function.<br/>
**Disadvantages of Barch Gradient Descent**
-  Slower Learning: The downside of batch gradient descent is a much slower learning process because we perform only one update after N samples have been processed.
-  Local Minima and Saddle Points: Another disadvantage is the fact that during the learning process we can get stuck in a local minimum of the loss function and never reach the global optimum at which the neural network achieves the best results. This is because the gradients we calculate are more or less the same. What we actually need are in fact some noisy gradients. Such small deviations in the directional values would allow the gradient to jump out of a local minimum of the loss function and continue the updates towards the global minimum. Clean gradients, on the other hand, are much more prone to getting stuck in a local minimum <br/>
**Advantages of Stochastic Gradient Descent**
- Immediate Performance Insights: The stochastic gradient descent immediately gives us an insight into the performance of the neural network, since in this case, we do not have to wait until the end of the data set.
- Faster Learning: Accordingly, stochastic gradient descent may result in much faster learning because an update is performed after each data instance is processed. <br/>
**Disadvantages of Stochastic Gradient Descent**
- Noisy Gradients: In contrary to the batch gradient descent where we average the gradient to get one final gradient, in stochastic gradient descent we use every single gradient to update the weights. These gradients can be very noisy and have a lot of variance with respect to their directions and values. Meaning the gradients that we compute on each sample are only rough estimates of the true gradient that points towards the increase of the loss function. In other words, in this case, we have a lot of noise. However, this fact can avoid the local minima during training because the high variance can cause the gradient to jump out of a local minimum.
- Computationally Intensive: The stochastic gradient descent is much more computationally intensive than the batch gradient descent since in this case, we perform the weight updates more often.
- Inability to settle on a global Minimum: Another disadvantage may be the inability of the gradient descent to settle on a global minimum of the loss function. Due to the noisiness, it would be more difficult to find and stay at a global minimum.<br/>
**Advantages of Mini-Batch Gradient Descent**
- Computational Efficiency: In terms of computational efficiency, this technique lies between the two previously introduced techniques.
Stable Convergence: Another advantage is the more stable converge towards the global minimum since we calculate an average gradient over n samples that results in less noise.
- Faster Learning: As we perform weight updates more often than with stochastic gradient descent, in this case, we achieve a much faster learning process.<br/>
**Disadvantages of Mini-Batch Gradient Descent**
- New Hyperparameter: A disadvantage of this technique is the fact that in mini-batch gradient descent, a new hyperparameter n, called as the mini-batch size, is introduced. It has been shown that the mini-batch size after the learning rate is the second most important hyperparameter for the overall performance of the neural network. For this reason, it is necessary to take time and try many different batch sizes until a final batch size is found that works best with other parameters such as the learning rate.


////////////////////////////////////////////////////////
#### Is the process of iteratively improving the accuracy of a machine learning model, lowering the degree of error. <\b> . A major goal of training a machine learning algorithm is to minimise the degree of error between the predicted output and the true output. 
Machine learning models learn to generalise and make predictions about new live data based on insight learned from training data. This works by approximating the underlying function or relationship between input and output data. 
  
Optimisation is measured through a loss or cost function, which is typically a way of defining the difference between the predicted and actual value of data. Machine learning models aim to minimise this loss function, or lower the gap between prediction and reality of output data. Iterative optimisation will mean that the machine learning model becomes more accurate at predicting an outcome or classifying data.  
///////////////////////////////////////////////////////
Momentum is widely used in the machine learning community for optimizing non-convex functions such as deep neural networks.[6] Empirically, momentum methods outperform traditional stochastic gradient descent approaches.
n deep learning, SGD is widely prevalent and is the underlying basis for many optimizers such as Adam, Adadelta, RMSProp, etc. which already utilize momentum to reduce computation speed.[9] The momentum extension for optimization algorithms is available in many popular machine learning frameworks such as PyTorch, tensor flow, and scikit-learn. 
  
 
  Momentum is an extension to the gradient descent optimization algorithm that builds inertia in a search direction to overcome local minima and oscillation of noisy gradients.[1] It is based on the same concept of momentum in physics.
  ////////////////////////////////////////////////////
resorces :
https://towardsdatascience.com/understanding-optimization-algorithms-in-machine-learning-edfdb4df766b#:~:text=Optimization%20is%20the%20process%20where,Learning%20to%20get%20better%20results.
https://www.analyticsvidhya.com/blog/2021/03/variants-of-gradient-descent-algorithm/


Numerical  optimization 

[1.implement the gradient descent for linear regressionwith one variable 
](https://github.com/AlaaSayed164/NumaricalCodes/blob/main/Practical%20Session%201%20GD%20Implementation%20for%20LR%20.ipynb)

[2.implement the gradient descent variants (Batch/Mini-Batch/Stochastic) for linear regression with one variable 
](https://github.com/AlaaSayed164/NumaricalCodes/blob/main/Practical%20Session%202%20GD%20Variants%20Batch%20-%20Mini-Batch%20-%20Stochastic.ipynb)

[3.implement the accelerated gradient descent methods (Momentum and NAG) for linear regression with one variable 
](https://github.com/AlaaSayed164/NumaricalCodes/blob/main/Practical%20Session%203%20Momentum%20-%20NAG%20(1).ipynb)

[4.implement the accelerated gradient descent methods with adaptive learning rate (Adagrad, RMSProp, and Adam) for linear regression with one variable 
](https://github.com/AlaaSayed164/NumaricalCodes/blob/main/Practical%20Session%204%20Adagrad-RMSProp-Adam%20(1).ipynb)



