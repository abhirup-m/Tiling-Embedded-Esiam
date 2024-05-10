using Test
using Random
using ProgressMeter
include("../source/probes.jl")
include("../source/helpers.jl")
include("../source/rgFlow.jl")


function scattProbtest()
    @testset "Flow of totalScatterProb" begin
        kondoJArray = rand(Float64, (SIZE_BZ[1]^2, SIZE_BZ[1]^2, 3))
        dOfStates, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
        results_norm, results_bare, results_bool = scattProb(kondoJArray, 3, SIZE_BZ[1], dispersion, 0.0)
        @testset for p1 in 1:SIZE_BZ[1]^2
            results_compare = sum(kondoJArray[p1, FS_POINTS, 3] .^ 2)
            results_compare_bare = sum(kondoJArray[p1, FS_POINTS, 1] .^ 2)
            results_compare_bool = tolerantSign(results_compare ./ results_compare_bare, RG_RELEVANCE_TOL)
            if !(p1 in CORNER_POINTS || p1 in CENTER_POINTS)
                if results_compare_bool < 0
                    results_compare = NaN
                end
                @test results_norm[p1] ≈ results_compare ./ results_compare_bare rtol = TOLERANCE
                @test results_bool[p1] == results_compare_bool
            end
        end

    end
end


function correlationFuncTest()
    size_BZ_range = [5]
    W_val_range = -0.05
    @testset "size=$size_BZ" for size_BZ in size_BZ_range
        halfSize = trunc(Int, (size_BZ + 1) / 2)
        kx_vals_FS_SE = range(0, stop=pi, length=halfSize)
        ky_vals_FS_SE = range(-pi, stop=0, length=halfSize)
        @testset "W=$W_val" for W_val in W_val_range
            kondoJArray, dispersion, fixedpointEnergy = momentumSpaceRG(size_BZ, -2.0, 0.1, W_val, ("p", "p"))
            results, results_bare, results_bool = spinFlipCorrMap(size_BZ, dispersion, kondoJArray, W_val, ("p", "p"))
            @testset "FS point symmetry, point ($kx, $ky)" for (kx, ky) in zip(kx_vals_FS_SE, ky_vals_FS_SE)
                @test results[map2DTo1D(kx, ky, size_BZ)] ≈ results[map2DTo1D(pi - kx, -pi - ky, size_BZ)] atol = 1e-8 rtol = 1e-5
                println(results[map2DTo1D(kx, ky, size_BZ)])
            end
        end
    end
end
