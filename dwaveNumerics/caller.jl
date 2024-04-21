include("./source/main.jl")

J_val = 0.1
size_BZ = 53
node = map2DTo1D(-pi/2, -pi/2, size_BZ)
antinode = map2DTo1D(-pi, 0., size_BZ)
labels = ["p", "poff", "d", "doff"]
for J_sym in labels
    row_results = [["N", "N"] for _ in labels]
    for (j, W_sym) in enumerate(labels)
        noderesults = [0, 0, 0]
        antinoderesults = [0, 0, 0]
        for (i, W_by_J) in enumerate([0, 1., -1.])
            kondoJArrayFull, dispersion, fixedpointEnergy = momentumSpaceRG(size_BZ, -2., J_val, J_val * W_by_J, [J_sym, W_sym]; progressbarEnabled=false)
            _, _, results_bool = scattProb(kondoJArrayFull, size(kondoJArrayFull)[3], size_BZ, dispersion, fixedpointEnergy)
            noderesults[i] = results_bool[node]
            antinoderesults[i] = results_bool[antinode]
        end
        if noderesults[2] < 0 && antinoderesults[2] < 0 && (noderesults[1] >= 0 || antinoderesults[1] >= 0)
            row_results[j][1] = "Y"
        end
        if noderesults[3] < 0 && antinoderesults[3] < 0 && (noderesults[1] >= 0 || antinoderesults[1] >= 0)
            row_results[j][2] = "Y"
        end
    end
    println(join(join.(row_results, ""), "\t"))
end

# manager(33, -2.0, 0.1, collect(0:-0.5:-1), ["p", "doff"], ["scattProb"], figScale=1.0)
#
# Columns labels are for W-symmetry.
# Row labels are for J-symmetry.
# First(second) character in each entry is for W>(<)0.
# N means no transition. Y means there is a transition.
#
# W      p      poff    d     doff 
# J    |-------------------------
# p    | NY     NN     NN     NN
# poff | NN     NY     NN     NN
# d    | NN     NN     NN     NN
# doff | NN     NN     NN     NY
#
