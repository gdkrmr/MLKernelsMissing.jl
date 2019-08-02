function missing_to_zeros!(X)
    for i in eachindex(X)
        if ismissing(X[i])
            X[i] = 0
        end
    end
    X
end

my_mean(μ, n, x)          = μ + (x - μ) / (n + 1), n + 1
my_mean(μ, n, x::Missing) = μ, n
function agg_mean(x)
    μ, n = zero(eltype(x)), 0
    @inbounds for xi in x
        μ, n = my_mean(μ, n, xi)
    end
    return μ
end

function missing_to_mean(X :: Matrix{Union{T, Missing}}) where T

    # If there are missing values here, then there are too many!
    # this causes quite a lot of allocations, but is still pretty fast
    m = [mean(skipmissing(view(X, :, i))) for i in 1:size(X, 2)]
    # much less allocations, but slow:
    # m = [agg_mean(view(X, :, i)) for i in 1:size(X, 2)]
    # this one is way to slow:
    # m = mapslices(x -> mean(skipmissing(x)), X, dims = 1)
    m2 = mean(m)

    X2 = Array{T}(undef, size(X))
    @inbounds for i in eachindex(X, X2)
        X2[i] = ismissing(X[i]) ? m2 : X[i]
    end

    return X2, m, m2
end

# eachslice requires Julia 1.1!
function missing_to_mean_slice(X :: Matrix{Union{T, Missing}}, dims = 1) where T

    X2 = Matrix{T}(undef, size(X))
    for (s1, s2) in zip(eachslice(X, dims = dims), eachslice(X, dims = dims))
        m = s1 |> skipmissing |> mean
        for i in eachindex(s1, s2)
            @inbounds s1[i] = ismissing(s2[i]) ? m : s2[i]
        end
    end

    X2
end

function remove_random!(
    x::Array{Union{T, Missing}},
    p = 0.05
) where {T <: AbstractFloat}
    for i in eachindex(x)
        if rand(T) < T(p)
            x[i] = missing
        end
    end
    x
end
