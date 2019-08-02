"""
    Kernel PCA object
"""
struct kPCA{T <: Real}
    X::Matrix{Union{T, Missing}} # original data
    κ::Kernel{T} # kernel function (from MLKernels)
    μ::Vector{T} # row/col means of the kernel matrix
    μ2::T        # mean of the kernel matrix
    λ::Vector{T} # eigenvalues in feature space
    α::Matrix{T} # eigenvectors in feature space

    function kPCA(
        X  :: Matrix{Union{T, Missing}},
        κ  :: Kernel{T},
        μ  :: Vector{T},
        μ2 :: T,
        λ  :: Vector{T},
        α  :: Matrix{T}
    ) where T <: Real
        @assert length(λ) == size(α, 2)
        @assert length(μ) == size(X, 2)

        new{T}(X, κ, μ, μ2, λ, α)
    end
end

indim(M::kPCA) = size(M.X, 1)
outdim(M::kPCA) = length(M.λ)

projection(M::kPCA) = M.V ./ sqrt.(M.λ)
principalvars(M::kPCA) = M.λ

"""
fit a kernel PCA.

    κ :: Kernel{T} the kernel function, see MLKernels.jl for documentation.
    X :: Array{Union{T, Missing}} the data to embed, with observations ins columns.
"""
function fit(
      :: Type{kPCA},
    κ :: Kernel{T},
    X :: Matrix{Union{T, Missing}},
    ncomp = 2
) where T

    K = kernelmatrix(Val(:col), κ, X, true)
    K2, μ, μ2 = missing_to_mean(K)
    K2 .= K2 .- μ .- μ' .+ μ2

    e = K2 |> Hermitian |> eigen
    e_ev_perm = sortperm(e.values, rev = true)[1:ncomp]

    λ = e.values[e_ev_perm]::Vector{T}
    α = e.vectors[:, e_ev_perm]::Matrix{T}
    return kPCA(X, κ, μ, μ2, λ, α)
end

"""Calculate transformation to kernel space"""
# function transform(M::kPCA{T}, x::AbstractMatrix{mT}) where mT <: Union{T, Missing} where T <: Real
function transform(M::kPCA{T}, x::AbstractMatrix{mT}) where mT where T

    ##### There is some typeinference bug that leads to illegal instructions and
    ##### reaches the unreachable...

    K = kernelmatrix(Val(:col), M.κ, M.X, x)
#     ##### instead of `kernelmatrix`:
#     # K = allocate_basematrix(Val(:col), M.X, x)
#     K = Array{T}(undef, size(M.X, 2), size(x, 2))
#     f = basefunction(M.κ)

#     # basematrix!(Val(:col), K, f, M.X, x)
#     ##### instead of `basematrix!`:
#     n, m = checkdimensions(Val(:col), K, M.X, x)
#     @inbounds for j = 1:m
#         yj = subvector(Val(:col), x, j)
#         for i = 1:n
#             xi = subvector(Val(:col), M.X, i)

# ##### this causes slightly less allocations but far worse performance at least it doesn't crash the julia session
# # # unsafe_base_evaluate
# #     s, nn = base_initiate(f, T), 0
# #     # s, nn = base_initiate(f, promote_type(eltype(x), eltype(y))), 0
# #     for i in eachindex(xi, yj)
# #         s, nn = base_aggregate(f, s, nn, xi[i], yj[i])
# #     end
# #     K[i, j] = nn > 0 ? base_return(f, s / nn) : missing
# # # end unsafe_base_evaluate

#             K[i,j] = unsafe_base_evaluate(f, xi, yj)
#         end
#     end

#     ##### end instead of `basematrix!`

#     kappamatrix!(M.κ, K)
#     ##### end instead of `kernelmatrix`

    K .= K .- M.μ .- mean(K, dims = 1) .+ M.μ2

    return (M.α' ./ sqrt.(M.λ)) * K
end

transform(M::kPCA) = sqrt.(M.λ) .* M.α'

function Base.show(io::IO, ::MIME"text/plain", M::kPCA{T}) where T
    println(io, "kPCA{", T, "}(κ: ", M.κ, ", dims: ",  indim(M), "─→", outdim(M), ")")
end

# """Kernel PCA type"""
# struct KernelPCA{T<:Real}
#     X::AbstractMatrix{T}           # fitted data or precomputed kernel
#     ker::Union{Nothing, Function}  # kernel function
#     center::KernelCenter           # kernel center
#     λ::AbstractVector{T}           # eigenvalues  in feature space
#     α::AbstractMatrix{T}           # eigenvectors in feature space
#     inv::AbstractMatrix{T}         # inverse transform coefficients
# end
# struct KernelCenter{T<:Real}
#     means::AbstractVector{T}
#     total::T
# end

# import MultivariateStats: KernelPCA, KernelCenter, transform!
# import Base.convert

# """Center kernel matrix."""
# function transform!(C::KernelCenter{T}, K::AbstractMatrix{T}) where {T<:Real}
#     r, c = size(K)
#     tot = C.total
#     means = mean(K, dims=1)
#     K .= K .- C.means .- means .+ tot
#     return K
# end

# TODO: does not work
# function convert(::Type{KernelPCA{T}}, M::kPCA) where T
#     KernelPCA(missing_to_mean_slice(M.X, dims = 2),
#               (x, y) -> kappa(M.κ, unsafe_base_evaluate(basefunction(M.κ), x, y)),
#               KernelCenter(M.μ, M.μ2),
#               M.λ,
#               M.α,
#               zeros(T, 0, 0))
# end
