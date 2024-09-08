f = Figure(size=(400, 300), figure_padding=8)
ax = Axis(f[1, 1], xlabel=L"Kondo Int. $(J)$", ylabel=L"Bath Int. $(-W)$",
          xlabelsize=14, ylabelsize=14,)
heatmap!(ax, J_val_arr, -1 .* W_val_arr, phaseDiagram;
         colormap=cmap, thickness_scaling=1.5, legend=true)

sc1 = scatterlines!(ax, J_val_arr[nodeGap .!= nothing][trunc.(Int, range(1, stop=end, length=40))], -1. * nodeGap[nodeGap .!= nothing][trunc.(Int, range(1, stop=end, length=40))],
       color=lineColorSet[1], marker=:rect, markersize=7, strokecolor=:gray, strokewidth=1)
sc2 = scatterlines!(ax, J_val_arr[antiNodeGap .!= nothing][trunc.(Int, range(1, stop=end, length=40))], -1. * antiNodeGap[antiNodeGap .!= nothing][trunc.(Int, range(1, stop=end, length=40))],
             color=lineColorSet[2], marker=:circle, markersize=7, strokecolor=:gray, strokewidth=1)
sc3 = scatterlines!(ax, J_val_arr[midPointGap .!= nothing][trunc.(Int, range(1, stop=end, length=40))], -1. * midPointGap[midPointGap .!= nothing][trunc.(Int, range(1, stop=end, length=40))],
             color=lineColorSet[3], marker=:star6, markersize=7, strokecolor=:gray, strokewidth=1)
legnd = [[PolyElement(color=col) for col in colorSet]; [sc1, sc2, sc3]]
axislegend(ax, legnd, labelSet, labelsize=12, markersize=2)
save("phaseDiagram.pdf", f, pt_per_unit=2)
