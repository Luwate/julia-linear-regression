# Linear Regression from Scratch

This project follows a tutorial from the extremely talented Doggo dot jl on youtube. The aim of this readme file is to understand the concepts of what is being executed in the code.

First the data is loaded in using the CSV package. The data contains size of houses in one column and the price in another column. The independent variable (X) is size and the dependent variable (Y) is price.

We will use two techniques for linear regression; a non-machine learning approach and a machine learning approach.

Let's begin with the non-machine learning approach.

For this, we use Ordinary Least Squares method EXPLAIN OLS

The predictions are then plotted onto a plot using the predict() function.

For the Machine Learning approach: 

We begin by declaring a new variable, epochs, to zero. This keeps track of the number of iterations that the model has gone through. theta_0 (y - intercept) and theta_1 (m - gradient) are also declared.

As this is a linear regression model, we then define the hypothesis using: h(x) = theta_0 .+ theta_1 * x 

NB: .+ adds the value of theta_0 to each value of theta_1 * x as x takes multiple values. It's a much quicker way than using a for loop

Now we need to define the cost function. The cost function calculates the difference between the predicted and actual values. Hence measures the performance of an algorithm.

In this instance, we use the following cost function:

(1/(2 * m)) * sum((y_hat - Y).^2)

the cost function is then assigned to a variable J and is pushed to a vector J_history in order to keep record of the cost as the model iterates.

Now, now, now... We have a way to calculate the cost but we don't quite yet have a way to adjust the theta_0 and theta_1 which are the main parameters used to plot the line. 

We use the following formula to manipulate theta_0:

(1 / m) * sum(y_hat - Y)

And the following formula to manipulate theta_1

(1 / m) * sum((y_hat - Y) .* X)

We also need to specify learning rates. Learning rates are used to slow down the amount of change per iteration. Initially we want big changes to the model as it is likely way off being a suitable approximator. However, as the model iterates and becomes more accurate, it does not need to make huge changes anymore as it is already quite close. Learning rates help us execute this.

In this instance, we need two learning rates: one for theta_0 and one for theta_1

alpha_0 = 0.09
alpha_1 = 0.00000008

As these are hyperparameters, they're primarily set according to user discretion and in this instance were funged in order to get the model to work. Ordinarily, you would start with the values at 0.1

Then comes the section of code that is iterated:

theta_0_temp and theta_1_temp are defined, which use the functions defined above to manipulate theta_0 and theta_1.

theta_0 and theta_1 then have their respective temp values multipled by the learning rates subtracted from their values. y_hat is then declared as h(x) with h(x) now containing the updated values for theta_0 and theta_1

The new cost value is calculated and pushed into the vector and the epocj counter is iterated by 1. The current approximation is then plotted and the iterated process is repeated with the new plots being placed on the existing canvas to show change in the model.




