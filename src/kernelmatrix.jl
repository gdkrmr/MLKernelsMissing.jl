kappa(κ::K, ::Missing) where K <: Kernel{T} where T = missing

function kappamatrix!(
    κ::Kernel{T},
    P#::AbstractMatrix{mT}
) where mT <: Union{Missing, T} where T <: AbstractFloat
    for i in eachindex(P)
        @inbounds P[i] = kappa(κ, P[i])
    end
    return P
end

function symmetric_kappamatrix!(
    κ::Kernel{T},
    P::AbstractMatrix{mT},
    symmetrize::Bool
) where {mT <: Union{Missing, T}} where {T<:AbstractFloat}
    if !((n = size(P,1)) == size(P,2))
        throw(DimensionMismatch("Pairwise matrix must be square."))
    end
    for j = 1:n, i = (1:j)
        @inbounds P[i,j] = kappa(κ, P[i,j])
    end

    symmetrize && copytri!(P, 'U', false)

    return P
end

"""
    kernelmatrix!(σ::Orientation, K::Matrix, κ::Kernel, X::Matrix, symmetrize::Bool)

In-place version of `kernelmatrix` where pre-allocated matrix `K` will be overwritten
with the kernel matrix.
"""
function kernelmatrix!(
    σ::Orientation,
    K::Matrix{Union{Missing, T}},
    κ::Kernel{T},
    X::AbstractMatrix{mT1},
    symmetrize::Bool
) where
    {mT1 <: Union{Missing, T}} where
    {T <: AbstractFloat}

    f = basefunction(κ)

    basematrix!(σ, K, f, X, false)
    symmetric_kappamatrix!(κ, K, symmetrize)

    return K
end

"""
    kernelmatrix!(σ::Orientation, K::Matrix, κ::Kernel, X::Matrix, Y::Matrix)

In-place version of `kernelmatrix` where pre-allocated matrix `K` will be overwritten with
the kernel matrix.
"""
function kernelmatrix!(
    σ::Orientation,
    K::Matrix{Union{Missing, T}},
    κ::Kernel{T},
    X::AbstractMatrix{mT1},
    Y::AbstractMatrix{mT2}
) where
    {mT1 <: Union{Missing, T}} where
    {mT2 <: Union{Missing, T}} where
    {T <: AbstractFloat}

    # NOTE: for whatever reason, this gets inferred to ::Any which causes the
    # compiler to think that there is no method for basematrix! because there is
    # no basematrix! method with f::Any
    f = basefunction(κ)

    basematrix!(σ, K, f, X, Y)
    kappamatrix!(κ, K)

    return K
end

function kernelmatrix(
    σ::Orientation,
    κ::Kernel{T},
    X::AbstractMatrix{Union{T, Missing}},
    symmetrize::Bool = true
) where {T<:AbstractFloat}

    K = allocate_basematrix(σ, X)
    kernelmatrix!(σ, K, κ, X, symmetrize)

    return K
end

function kernelmatrix(
    σ::Orientation,
    κ::Kernel,
    X::AbstractMatrix{mT1},
    Y::AbstractMatrix{mT2}
) where {mT1 <: Union{Missing, T}} where {mT2 <: Union{Missing, T}} where {T<:AbstractFloat}

    K = allocate_basematrix(σ, X, Y)
    kernelmatrix!(σ, K, κ, X, Y)

    return K
end
