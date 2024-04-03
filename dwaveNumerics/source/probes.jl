using LinearAlgebra
using DelimitedFiles

function totalScatterProb(kondoJArray::Array{Float64,3}, stepIndex::Int64, savePath::String)
    bare_J_squared = diag(kondoJArray[:, :, 1] * kondoJArray[:, :, 1]')
    results_unnorm = diag(kondoJArray[:, :, stepIndex] * kondoJArray[:, :, stepIndex]')
    results_norm = results_unnorm ./ bare_J_squared
    results_bool = [tolerantSign(results_norm_i, 1) for results_norm_i in results_norm]

    writedlm(savePath, results_norm)
    writedlm(savePath * "-bool", results_norm)
    return results_norm, results_unnorm, results_bool
end
