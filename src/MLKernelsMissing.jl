module MLKernelsMissing

export
    kPCA, fit, transform,
    jacobian,
    Kernel,
        MercerKernel,
            AbstractExponentialKernel,
                ExponentialKernel,
                LaplacianKernel,
                SquaredExponentialKernel,
                GaussianKernel,
                RadialBasisKernel,
                GammaExponentialKernel,
            AbstractRationalQuadraticKernel,
                RationalQuadraticKernel,
                GammaRationalQuadraticKernel,
            MaternKernel,
            LinearKernel,
            PolynomialKernel,
            ExponentiatedKernel,
            PeriodicKernel,
        NegativeDefiniteKernel,
            PowerKernel,
            LogKernel,
        SigmoidKernel

import MLKernels
import MLKernels:
    BaseFunction,
    Orientation,
    kappa,
    kernelmatrix, kernelmatrix!,
    basematrix!,
    basefunction,
    checkdimensions, subvector,
    base_initiate,
    base_aggregate,
    base_return,
    unsafe_base_evaluate,

# reexported
    Kernel,

# reexported, so we don't have to say `using MLKernels`
        MercerKernel,
            AbstractExponentialKernel,
                ExponentialKernel,
                LaplacianKernel,
                SquaredExponentialKernel,
                GaussianKernel,
                RadialBasisKernel,
                GammaExponentialKernel,
            AbstractRationalQuadraticKernel,
                RationalQuadraticKernel,
                GammaRationalQuadraticKernel,
            MaternKernel,
            LinearKernel,
            PolynomialKernel,
            ExponentiatedKernel,
            PeriodicKernel,
        NegativeDefiniteKernel,
            PowerKernel,
            LogKernel,
        SigmoidKernel

import LinearAlgebra: Hermitian, eigen, copytri!
import Statistics: mean

# reexported
import StatsBase: fit
import MultivariateStats: transform

include("util.jl")
include("basematrix.jl")
include("kernelmatrix.jl")
include("kpca.jl")
include("jacobian.jl")
# include("datasets.jl")

end # module MLKernels
