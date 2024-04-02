using LinearAlgebra
using DelimitedFiles

function totalScatterProb(kondoJArray::Array{Float64,3}, num_kspace::Int64, stepIndex::Int64, savePath::String)
    bare_J_squared = reshape(
        diag(kondoJArray[:, :, 1] * kondoJArray[:, :, 1]'),
        (num_kspace, num_kspace),
    )
    results_unnorm =
        reshape(
            diag(kondoJArray[:, :, stepIndex] * kondoJArray[:, :, stepIndex]'),
            (num_kspace, num_kspace),
        )
    results_norm = results_unnorm ./ bare_J_squared
    results_bool = (sign.(
        reshape(
            diag(kondoJArray[:, :, stepIndex] * kondoJArray[:, :, stepIndex]'),
            (num_kspace, num_kspace),
        ) .- bare_J_squared
    ))
    writedlm(savePath, results_norm)
    writedlm(savePath * "-bool", results_norm)
    return results_norm, results_unnorm, results_bool
end
