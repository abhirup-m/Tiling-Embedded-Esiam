# define the edges of the Brillouin zone
const K_MIN = -pi
const K_MAX = pi

# tolerance for identifying energy contours
const TOLERANCE = 1e-10

# overlap integral 't' set to 1
const HOP_T = 1.0

# for testing. 
const SIZE_BZ = [5, 101, 1001]
const KX_VALUES = [K_MIN, K_MAX, K_MIN, K_MAX, 0]
const KY_VALUES = [K_MIN, K_MIN, K_MAX, K_MAX, 0]
const POINTS(num_kspace) = [
    1,
    num_kspace,
    num_kspace^2 - num_kspace + 1,
    num_kspace^2,
    Int(0.5 * num_kspace^2 + 0.5),
]

const FS_POINTS = [3, 7, 9, 11, 15, 17, 19, 23]
const FS_POINTS_LEFT = [3, 7, 11, 17, 23]
const CENTER_POINTS = [13]
const CORNER_POINTS = [1, 5, 21, 25]
const OUTSIDE_POINTS = [2, 4, 6, 10, 16, 20, 22, 24]
const INSIDE_POINTS = [8, 12, 14, 18]
const PROBE_ENERGIES = [0, -2 * HOP_T, 2 * HOP_T, -4 * HOP_T, 4 * HOP_T]
const DELTA_ENERGY = PROBE_ENERGIES[3] - PROBE_ENERGIES[1]
const PROCEED_FLAGS = fill(1, SIZE_BZ[1]^2, SIZE_BZ[1]^2)
