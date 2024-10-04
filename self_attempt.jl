using CSV, GLM, TypedTables, Plots

data = CSV.File("data/housingdata.csv")

X = data.size
Y = data.price

Y = round.(Int, Y/1000)


test = scatter(X, Y,
    xlabel = "Size",
    ylabel = "Price",
    legend = false,
    xlims = (0, 5000),
    ylims = (0, 900)
)

epochs = 0
theta_0 = 0.0
theta_1 = 0.0

h(x) = theta_0 .+ theta_1 * x

ŷ = h(X)

m = length(X)

function cost(X, Y)
    (1/(2 * m)) * sum((ŷ - Y).^2)
end

J = cost(X, Y)
J_history = []
push!(J_history, J)

function pd_theta_0(X, Y)
    (1 / m) * sum(ŷ - Y)
end

function pd_theta_1(X, Y)
    (1 / m) * sum((ŷ - Y) .* X)
end

alpha_0 = 0.09
alpha_1 = 0.00000008

theta_0_temp = pd_theta_0(X, Y)
theta_1_temp = pd_theta_1(X, Y)

theta_0 -= theta_0_temp * alpha_0
theta_1 -= theta_1_temp * alpha_1

ŷ = h(X)

J = cost(X, Y)
push!(J_history, J)
epochs += 1

plot!(X, ŷ,
    color = :red,
    title = "Chart of Size against Price on epoch: $epochs",
    linewidth = 3
)

newX = [1500]

h(newX)
