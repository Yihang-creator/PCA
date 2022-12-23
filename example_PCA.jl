using DelimitedFiles

# include("PCA.jl")
# # Load data
# dataTable = readdlm("animals.csv",',')
# X = float(real(dataTable[2:end,2:end]))
# (n,d) = size(X)
# model = PCA(X,2)
# Z = model.compress(X)
# @show dataTable[2:end,1]



# # Plot matrix as image
# using Plots
# scatter(Z[:,1],Z[:,2])
# annotate!(Z[:,1],Z[:,2],dataTable[2:end,1], annotationfontsize=8)
# savefig("animals.png")
include("PCA.jl")
# Load data
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
mu = mean(X,dims=1)
X = X - repeat(mu,n,1)
(n,d) = size(X)
model = PCA(X,12)
W = model.W
Z = model.compress(X)
res = (Z*W-X)
k = sum(res.^2)/sum(X.^2)
@show (1-k)


