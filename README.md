# Gradient Descent Data Visualization
The purpose of this project was to provide an animation for the gradient descent algorithm in the context of a simple linear regression model. The **"Gradient Descent Data Visualization.ipynb"** script provides user-control of the:

1. sample size (*n*)
2. intercept (b0) and slope (b1)
3. predictor feature mean and variance 
4. error variance
5. learning rate
6. tolerance

After defining these parameters, the animation function will animate 100 frames (20 frames per second) evenly spread across the iterations of the gradient descent algorithm. Each frame plots the evaluated b0 and b1 values across the iterations up to the current frame. The background color scheme represents a contour plot of the cost function, with red representing lower cost and blue higher cost.

*Note*: Larger values for the predictor feature mean and variance can increase computational time for the gradient descent algorithm.
