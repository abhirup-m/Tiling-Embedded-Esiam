include("./constants.jl")

# helper functions for switching back and forth between the 1D flattened representation (1 → N^2) 
# and the 2D representation ((1 → N)×(1 → N))
function map1DTo2D(point::Int64, num_kspace::Int64)
    # Convert overall point to row and column values.
    # These serve as indices of kx and ky
    kx_index = (point - 1) % num_kspace + 1
    ky_index = div((point - 1), num_kspace) + 1

    # construct a 1D array of possible k values, and convert
    # kx_index and ky_index into values
    k_values = range(K_MIN, stop = K_MAX, length = num_kspace)
    return [k_values[kx_index], k_values[ky_index]]
end
function map1DTo2D(point::Vector{Int64}, num_kspace::Int64)
    # same as above, but for multiple points. In this case,
    # two tuples are returned, for kx values and ky values.
    kx_index = (point .- 1) .% num_kspace .+ 1
    ky_index = div.((point .- 1), num_kspace) .+ 1
    k_values = range(K_MIN, stop = K_MAX, length = num_kspace)
    return [k_values[kx_index], k_values[ky_index]]
end

function map2DTo1D(kx_val::Float64, ky_val::Float64, num_kspace::Int64)
    # obtain the indices, given the values of kx and ky
    k_values = range(K_MIN, stop = K_MAX, length = num_kspace)
    kx_index, ky_index = argmin(abs.(k_values .- kx_val)), argmin(abs.(k_values .- ky_val))

    # covert the row and column indices into an overall flattened index
    return kx_index + (ky_index - 1) * num_kspace
end
function map2DTo1D(kx_val::Vector{Float64}, ky_val::Vector{Float64}, num_kspace::Int64)
    # same but for multiple (kx,ky) pairs.
    k_values = range(K_MIN, stop = K_MAX, length = num_kspace)
    points = Int64[]
    for (kx, ky) in zip(kx_val, ky_val)
        kx_index, ky_index = argmin(abs.(k_values .- kx)), argmin(abs.(k_values .- ky))
        push!(points, kx_index + (ky_index - 1) * num_kspace)
    end
    return points
end


# create flattened tight-binding dispersion
function tightBindDisp(kx_vals::Vector{Float64}, ky_vals::Vector{Float64})
    return -2 * HOP_T .* (cos.(kx_vals) + cos.(ky_vals))
end


function teleportToAllQuads(point::Int64, num_kspace::Int64)
    # given a point in the lower left quadrant, find other points symmetric to this,
    # through the tranformation kx -> 2\pi - kx, ky -> 2pi - ky.
    # -------------------
    # | x             x |
    # |                 |
    # |                 |
    # | o             x |
    # -------------------
    kx_index_L, ky_index_D = map1DTo2D(point, num_kspace)
    kx_index_R = (num_kspace + 1) - kx_index_L # gives kx' = 2pi - kx
    ky_index_U = (num_kspace + 1) - ky_index_D # gives ky' = 2pi - ky

    # get all four points by combining the transformed coordinates:
    # (for eg., (kx, ky), (kx', ky), (kx, ky'), (kx', ky')
    telepPoints = [
        map2DTo1D(kx_index, ky_index, num_kspace) for (kx_index, ky_index) in [
            [kx_index_L, ky_index_D],
            [kx_index_L, ky_index_U],
            [kx_index_R, ky_index_D],
            [kx_index_R, ky_index_U],
        ]
    ]
    return telepPoints
end

function getDensityOfStates(dispersionFunc, num_kspace)
    kx_vals = repeat(range(K_MIN, stop = K_MAX, length = num_kspace), outer = num_kspace)
    ky_vals = repeat(range(K_MIN, stop = K_MAX, length = num_kspace), inner = num_kspace)

    dispersion = dispersionFunc(kx_vals, ky_vals)
    dispersion_xplus1 = dispersionFunc(circshift(kx_vals, 1), ky_vals)
    dispersion_xminus1 = dispersionFunc(circshift(kx_vals, -1), ky_vals)
    dispersion_yplus1 = dispersionFunc(ky_vals, circshift(ky_vals, num_kspace))
    dispersion_yminus1 = dispersionFunc(ky_vals, circshift(ky_vals, -num_kspace))
    dOfStates =
        4 ./
        sqrt.(
            (dispersion_xplus1 - dispersion_xminus1) .^ 2 +
            (dispersion_yplus1 - dispersion_yminus1) .^ 2
        )

    # van-Hoves might result in Nan, replace them with the largest finite value
    replace!(dOfStates, Inf => maximum(dOfStates[dOfStates.≠Inf]))

    # normalise the DOS according to the convention \int dE \rho(E) = N (where N is the total number of k-states)
    dOfStates /=
        sum([
            dos * abs(dispersion[i%num_kspace+1] - dispersion[i]) for
            (i, dos) in enumerate(dOfStates)
        ]) / num_kspace^2
    return dOfStates, dispersion
end


function getIsoEngCont(dispersion::Vector{Float64}, energy::Float64)
    # obtain the k-space points that have the specified energy. We count one point per row.

    difference = sign.(dispersion .- energy)
    contourPoints = Int64[]
    for point in eachindex(difference[1:end-1])
        if difference[point] != difference[point+1]
            push!(contourPoints, point)
        end
    end
    return contourPoints
end


function particleHoleTransf(point::Vector{Int64}, num_kspace::Int64)
    kx_val, ky_val = map1DTo2D(point, num_kspace)
    kx_new = [
        kx .< (K_MAX + K_MIN) / 2 ? kx .+ (K_MAX - K_MIN) / 2 : kx .- (K_MAX - K_MIN) / 2 for kx in kx_val
    ]
    ky_new = [
        ky .< (K_MAX + K_MIN) / 2 ? ky .+ (K_MAX - K_MIN) / 2 : ky .- (K_MAX - K_MIN) / 2 for ky in ky_val
    ]
    return map2DTo1D(kx_new, ky_new, num_kspace)
end
