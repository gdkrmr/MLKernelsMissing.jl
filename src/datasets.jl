
function swiss_roll(n, σ = 0.1)
    # x = rand(n) .* (3.0 * π) .+ (1.5 * π)
    # y = rand(n) .* (30.0 * π)

    ### 1
    # f = permutedims([(x, y) -> x .* cos.(x),
    #                  (x, y) -> y,
    #                  (x, y) -> x .* sin.(x) ])
    # broadcast((f, x, y) -> f(x, y), f, x, y)

    ### 2
    # f = ((x, y) -> x .* cos.(x),
    #      (x, y) -> y,
    #      (x, y) -> x .* sin.(x) )
    # for i in axis(res, 1), j in axis(res, 2)
    #     res[i, j] = ff[j](x[i], y[i])
    # end
    # return res

    ### 3
    # res = Array{Float64, 2}(undef, n, 3)
    # f1(x, y) = x .* cos.(x)
    # f2(x, y) = y
    # f3(x, y) = x .* sin.(x)
    # res[:, 1] .= f1.(x, y) .+ randn(n) .* σ
    # res[:, 2] .= f2.(x, y) .+ randn(n) .* σ
    # res[:, 3] .= f3.(x, y) .+ randn(n) .* σ
    # return res

    ### 4
    # hcat(x .* cos.(x), y, x .* sin.(x));

    ### 5
    # f(x, y) = (x * cos(x), y, x * sin(x))
    # permutedims(reshape(reinterpret(Float64, f.(x, y)), (3, n)))

    xx() = rand() .* (3.0 * π) .+ (1.5 * π)
    yy() = rand() .* (30.0 * π)

    permutedims(reshape(reinterpret(Float64, [
        begin
            x = xx()
            y = yy()
            (x * cos(x), y, x * sin(x))
        end for i in 1:n
    ]), (3, n)))
end

function circles(n, σ = 0.1, factor = 0.8)
    res = Array{Float64, 2}(undef, n, 2)

    classes = rand(Bool, n)
    ff = classes .* factor
    res[:, 1] .= cos.(range(0, stop = 2 * π, length = n)) .* ff
    res[:, 2] .= sin.(range(0, stop = 2 * π, length = n)) .* ff

    res .+= randn(size(res)...) .* σ
    res, classes
end
