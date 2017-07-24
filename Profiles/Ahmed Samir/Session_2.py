from numpy import *
import matplotlib.pyplot as plt

# h(x) = y = mx + b
# m is slope, b is y-intercept or m is theta 1, b is theta 0
#points[i][0]=hyp(x(i)),points[i][1]=y(i)
#========================================================================================================================================================
#Squared error function

def compute_error_for_line_given_points(b, m, points):
    error=0.0
    for i in range(len(points)):
        temp=(((m*points[i][0])+b)-points[i][1])**2
        error+=temp
    return error/(2*len(points))
# ========================================================================================================================================================
#theta0 = theta0 - eta * (1/n)*sum(h(xi)-y(xi))
#theta1 = theta1 - eta * (1/n)*sum(h(xi)-y(xi))*xi

def step_gradient(b_current, m_current, points, learningRate):
    temp0=0.0;temp1=0.0
    for i in range(len(points)):
        temp0+=(((m_current*points[i][0])+b_current)-points[i][1])
        temp1+=((((m_current*points[i][0])+b_current)-points[i][1])*points[i][0])
    return [b_current-((learningRate*temp0)/len(points)),m_current-((learningRate*temp1)/len(points))]

# ========================================================================================================================================================

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations = 3):
    counter = 0 # counter used for the drawing
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
# ========================================================================================================================================================
        #   The drawing staff, we will update it once after each 10 iterations
        print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(counter, b, m,compute_error_for_line_given_points(b, m, points)))
        if counter%100 is 0:
            plt.plot(points[:, 0], points[:, 1], 'bo') # Draw the dataset
            plt.plot([0, 80], [b, 80*m+b], 'b')
            plt.show()
# ========================================================================================================================================================
        b, m = step_gradient(b, m, array(points), learning_rate)
        counter+=1
    return [b, m]

#========================================================================================================================================================

def Train():
    points = genfromtxt("data.csv", delimiter=",") # Function in Numby that reads the data from a file and organize it.
    learning_rate = 0.000001 # Eta
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 2000

    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

#========================================================================================================================================================
#The Main:
Train();

# ========================================================================================================================================================
'''
Some useful resources:
 * Linear regression:
   - http://www.ozzieliu.com/tutorials/Linear-Regression-Gradient-Descent.html
 * Gradient Descent:
   - http://blog.hackerearth.com/gradient-descent-algorithm-linear-regression
 * numpy:
   - https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
 * Existing projects:
   - https://github.com/mattnedrich/GradientDescentExample
   - https://github.com/sq0032/ML/blob/master/LinearRegression/main.py
 * Other blogs:
   - http://www.ozzieliu.com/tutorials/Linear-Regression-Gradient-Descent.html
 * Simple rich tutorial:
   - http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
   - http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-2/
   - http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-3/
'''
