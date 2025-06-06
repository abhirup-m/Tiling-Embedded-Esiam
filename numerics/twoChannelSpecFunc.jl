using Fermions, Plots, Serialization, LsqFit, LaTeXStrings


hop_t = 1.0
kondoJ = 0.5
maxSize = 2000
p = scatter()
#=p = scatter(xscale=:log10, yscale=:log10)=#
lstyles = [:solid, :dot]
totalSpecFunc = nothing
specFuncDefDict = Dict("f0" => [("+", [3], 1.0)], "imp" => [("+-+", [1, 2, 4], 1.0), ("+-+", [2, 1, 3], 1.0)])
#=freqValues = 0.0001:0.0001:0.1=#
freqValues = collect(10 .^ (-3.5:0.01:1.5))
freqValues = vcat(reverse(-1 .* freqValues), freqValues)
for numChannels in [2]
    for numBathSites in [30]
        effkondoJ = kondoJ / (numChannels ^ 2)
        path = "saveData/K=$(numChannels),J_by_t=$(effkondoJ/hop_t),N=$(numBathSites),M=$(maxSize)"
        if isfile(path*"SFO") && isfile(path*"SP") && false
            specFuncOperators = deserialize(path*"SFO")
            savePaths = deserialize(path*"SP")
            totalSpecFunc = IterSpecFunc(savePaths, specFuncOperators["imp"], freqValues, 0.01)
        else
            if numChannels == 1
                hamiltonian = KondoModel(numBathSites, hop_t, kondoJ)
            else
                hamiltonian = KondoModel(numBathSites, hop_t, [effkondoJ, effkondoJ])
            end
            partitions = 2*(1 + numChannels):2*numChannels:2*(1 + numChannels * numBathSites) |> collect
            if 2*(1 + numChannels * numBathSites) ∉ partitions
                push!(partitions, 2*(1 + numChannels * numBathSites))
            end
            hamiltonianFlow = MinceHamiltonian(hamiltonian, partitions)

            savePaths, resultsDict, specFuncOperators = IterDiag(hamiltonianFlow, maxSize;
                                                                 symmetries=Char['N', 'S'],
                                                                 specFuncDefDict=specFuncDefDict,
                                                                 occReq=(x,N)->abs(x-N/2) < 10,
                                                                 magzReq=(m,N)->-5 < abs(m) < 6,
                                                                 )
            serialize(path*"SFO", specFuncOperators)
            serialize(path*"SP", savePaths)
            totalSpecFunc = IterSpecFunc(savePaths, specFuncOperators["imp"], freqValues, 0.01)
        end
        totalSpecFunc ./= sum([A * ((i+1) ∈ eachindex(freqValues) ? freqValues[i+1] - freqValues[i] : freqValues[i] - freqValues[i-1]) 
                               for (i, A) in enumerate(totalSpecFunc)])
        ydata = totalSpecFunc[0.0 .≤ freqValues .≤ 0.003]
        xdata = freqValues[0.0 .≤ freqValues .≤ 0.003]
        model(x, p) = p[1] .* (x .^ p[2]) .+ p[3]
        fit = curve_fit(model, xdata, ydata, [1., 2., 0.])
        params = coef(fit)
        println(params)
        println(stderror(fit))
        scatter!(p, xdata, ydata, label="K=$(numChannels), L=$(numBathSites)", ls=lstyles[numChannels])
        plot!(p, xdata, model(xdata, params), label=L"Fit: $ %$(trunc(params[3], digits=2)) + %$(trunc(params[1], digits=2)) \omega^{%$(trunc(params[2], digits=2))}$", ls=lstyles[numChannels])
    end
end
#=plot!(xscale=:log10, yscale=:log10)=#
savefig("specFuncMCK.pdf")
