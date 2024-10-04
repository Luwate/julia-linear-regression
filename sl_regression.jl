# Non-machine learning approach

#Packages
using CSV, GLM, Plots, TypedTables

data = CSV.File("data/housingdata.csv")

X = data.size

Y = round.(Int, data.price/1000 )

t = Table(X = X, Y = Y)

gr(size = (600,600))

p_scatter = scatter(X, Y; 
    xlims = (0, 5000),
    ylims = (0, 900),
    xlabel = "Size (sqft)",
    ylabel = "Price (in Thousands of Dollars)",
    legend = false,
    color = :red
)

#Use GLM package for linear regression model

ols = lm(@formula(Y ~ X), t)

plot!(X, predict(ols), color = :green, linewidth = 3)

newX = Table(X =[1250])

predict(ols, newX)


#ML approach
epochs = 0

p_scatter = scatter(X, Y; 
    xlims = (0, 5000),
    ylims = (0, 900),
    xlabel = "Size (sqft)",
    ylabel = "Price (in Thousands of Dollars)",
    title = "Housing Prices in portland (epochs = $epochs)",
    legend = false,
    color = :red
)

theta_0 = 0.0 #Y-intercept
theta_1 = 0.0 #slope

# Define model

h(x) = theta_0 .+ theta_1 * x

plot!(X, h(X), color = :blue, linewidth = 3)

#Cost function

m = length(X)

y_hat = h(X)

function cost(X,Y)
    (1/(2 * m)) * sum((y_hat - Y).^2)
    
end

J = cost(X, Y)

#Push cost value into Vector

J_history = []

push!(J_history, J)

#Define gradient descent algo (Used to manipulate theta_0 and theta_1 to minimize J)

function pd_theta_0(X,Y)
    (1 / m) * sum(y_hat - Y)
end

function pd_theta_1(X, Y)
    (1 / m) * sum((y_hat - Y) .* X)
end

alpha_0= 0.09

alpha_1 = 0.00000008


# Calculate partial derivatives

theta_0_temp = pd_theta_0(X, Y)

theta_1_temp = pd_theta_1(X, Y) 

#Adjust parameters by learning rate

theta_0 -= alpha_0 * theta_0_temp

theta_1 -= alpha_1 *theta_1_temp

y_hat = h(X)

J = cost(X, Y)

push!(J_history, J)

epochs += 1

plot!(X, y_hat, 
    color = :blue, 
    linewidth = 3, 
    alpha = 0.5,
    title = "Housing Prices in Portland (epochs = $epochs)"
)

#Measure Performance

plot!(X, predict(ols), color = :green, linewidth = 3)

gr(size = (600, 600))

p_line = plot(0:epochs, J_history,
    xlabel= "Epoch",
    ylabel = "Cost", 
    title = "Learning Curve",
    legend = false,
    color = :blue,
    linewidth = 2)

newX_ml = [1250]

h(newX_ml)

predict(ols, newX)
