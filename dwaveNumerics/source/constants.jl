# define the edges of the Brillouin zone
(@isdefined K_MIN) || const K_MIN = -pi
(@isdefined K_MAX) || const K_MAX = pi

# tolerance for identifying energy contours
(@isdefined TOLERANCE) || const TOLERANCE = 1e-10

# overlap integral 't' set to 1
(@isdefined HOP_T) || const HOP_T = 1.0

(@isdefined DISCRETE_CGRAD) || const DISCRETE_CGRAD = cgrad(:BuPu_3, 3, categorical=true)
(@isdefined FIG_SIZE) || const FIG_SIZE = (750, 300)

# for testing. 
(@isdefined SIZE_BZ) || const SIZE_BZ = [5, 101, 1001]
(@isdefined KX_VALUES) || const KX_VALUES = [K_MIN, K_MAX, K_MIN, K_MAX, 0]
(@isdefined KY_VALUES) || const KY_VALUES = [K_MIN, K_MIN, K_MAX, K_MAX, 0]
POINTS(num_kspace) = [
    1,
    num_kspace,
    num_kspace^2 - num_kspace + 1,
    num_kspace^2,
    Int(0.5 * num_kspace^2 + 0.5),
]

(@isdefined FS_POINTS) || const FS_POINTS = [3, 7, 9, 11, 15, 17, 19, 23]
(@isdefined FS_POINTS_LEFT) || const FS_POINTS_LEFT = [3, 7, 11, 17, 23]
(@isdefined CENTER_POINTS) || const CENTER_POINTS = [13]
(@isdefined CORNER_POINTS) || const CORNER_POINTS = [1, 5, 21, 25]
(@isdefined OUTSIDE_POINTS) || const OUTSIDE_POINTS = [2, 4, 6, 10, 16, 20, 22, 24]
(@isdefined INSIDE_POINTS) || const INSIDE_POINTS = [8, 12, 14, 18]
(@isdefined PROBE_ENERGIES) || const PROBE_ENERGIES = [0, -2 * HOP_T, 2 * HOP_T, -4 * HOP_T, 4 * HOP_T]
(@isdefined DELTA_ENERGY) || const DELTA_ENERGY = PROBE_ENERGIES[3] - PROBE_ENERGIES[1]
(@isdefined PROCEED_FLAGS) || const PROCEED_FLAGS = fill(1, SIZE_BZ[1]^2, SIZE_BZ[1]^2)
(@isdefined INNER_INDICES_ONE) || const INNER_INDICES_ONE = [2, 3, 6, 7, 8, 11, 12, 16, 17, 18, 22, 23]
(@isdefined EXCLUDED_INDICES_ONE) || const EXCLUDED_INDICES_ONE = [24, 19, 20, 14, 15, 9, 10, 4]

