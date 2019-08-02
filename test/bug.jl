exit()
cd("/home/gkraemer/progs/julia/MLKernelsMissing")
using Pkg
Pkg.activate(".")

using MLKernelsMissing

kpca = fit(kPCA, GaussianKernel(1.0), convert(Array{Union{Float64, Missing}}, rand(10, 100)))

transform(kpca, convert(Array{Union{Float64, Missing}}, rand(10, 1)))
