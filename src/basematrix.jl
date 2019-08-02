# Base.:(*)(x, ::Missing) = missing
# Base.:(*)(::Missing, y) = missing
# Base.:(+)(x, ::Missing) = missing
# Base.:(+)(::Missing, y) = missing

# @inline base_aggregate(::MLKernels.ScalarProduct, s, x, y) = s + x * y
# @inline base_aggregate(::MLKernels.SquaredEuclidean, s, x, y) = s + (x - y) ^ 2

@inline base_aggregate(f::BaseFunction, s::T, n, x::T,       y::T)       where T =
    base_aggregate(f, s, x, y), n + 1
@inline base_aggregate( ::BaseFunction, s::T, n, x::T,        ::Missing) where T = s, n
@inline base_aggregate( ::BaseFunction, s::T, n,  ::Missing, y::T)       where T = s, n
@inline base_aggregate( ::BaseFunction, s::T, n,  ::Missing,  ::Missing) where T = s, n

@inline function unsafe_base_evaluate(
    f::BaseFunction,
    x::AbstractArray{mT1},
    y::AbstractArray{mT2}
) where mT1 <: Union{T, Missing} where mT2 <: Union{T, Missing} where T <: AbstractFloat

    s, n = base_initiate(f, T), 0
    # s, n = base_initiate(f, promote_type(eltype(x), eltype(y))), 0
    @inbounds for i in eachindex(x, y)
        s, n = base_aggregate(f, s, n, x[i], y[i])
    end

    n > 0 ? base_return(f, s / n) : missing
end

# TODO: ask MLkernels.jl to  incorporate these changes
for orientation in (:row, :col)

    row_oriented = orientation == :row
    dim_obs      = row_oriented ? 1 : 2

    @eval begin

        @inline function allocate_basematrix(
             ::Val{$(Meta.quot(orientation))},
            X::AbstractMatrix{Tm}
        ) where Tm
            Array{Tm}(undef, size(X,$dim_obs), size(X,$dim_obs))
        end

        @inline function allocate_basematrix(
             ::Val{$(Meta.quot(orientation))},
            X::AbstractMatrix{mT1},
            Y::AbstractMatrix{mT2}
        ) where mT1 <: Union{T, Missing} where mT2 <: Union{T, Missing} where T <: AbstractFloat
            Array{Union{T, Missing}}(undef, size(X,$dim_obs), size(Y,$dim_obs))
        end

    end
end

function basematrix!(
    σ::Orientation,
    P::Matrix{Union{T, Missing}},
    f::BaseFunction,
    X::Matrix{mT},
    symmetrize::Bool
) where
    {mT <: Union{T, Missing}} where
    {T <: AbstractFloat}

    n = checkdimensions(σ, P, X)

    @inbounds for j in 1:n
        xj = subvector(σ, X, j)
        for i in 1:j
            xi = subvector(σ, X, i)
            P[i, j] = unsafe_base_evaluate(f, xi, xj)
        end
    end

    symmetrize && copytri!(P, 'U', false)

    return P
end

function basematrix!(
    σ::Orientation,
    P::Matrix{Union{T, Missing}},
    f::BaseFunction,
    X::AbstractMatrix{mT1},
    Y::AbstractMatrix{mT2},
) where
    mT1 <: Union{T, Missing} where
    mT2 <: Union{T, Missing} where
    T <: AbstractFloat

    n, m = checkdimensions(σ, P, X, Y)

    @inbounds for j = 1:m
        yj = subvector(σ, Y, j)
        for i = 1:n
            xi = subvector(σ, X, i)
            P[i,j] = unsafe_base_evaluate(f, xi, yj)
        end
    end

    return P
end
