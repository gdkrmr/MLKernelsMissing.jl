
function jacobian(f::Function, x::Vector; δ = 0.001)

    y = f(x)

    J = Matrix{Union{eltype(y), Missing}}(undef, length(y), length(x))

    for i in 1:length(x)
        if ismissing(x[i])
            J[:, i] .= missing
        else
            dx = copy(x)
            dx[i] += δ
            J[:, i] .= (y .- f(dx)) ./ δ
        end
    end

    return J
end
