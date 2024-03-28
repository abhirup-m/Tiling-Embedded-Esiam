import numpy as np
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool

K_MIN = -np.pi
K_MAX = np.pi


def getHolePoint(point, size):
    kx_val, ky_val = map1Dto2D(point, size)
    kx_new = (
        kx_val + (K_MAX - K_MIN) / 2
        if kx_val < (K_MAX + K_MIN) / 2
        else kx_val - (K_MAX - K_MIN) / 2
    )
    ky_new = (
        ky_val + (K_MAX - K_MIN) / 2
        if ky_val < (K_MAX + K_MIN) / 2
        else ky_val - (K_MAX - K_MIN) / 2
    )
    new_point = map2Dto1D(kx_new, ky_new, size)
    return new_point


def bathIntFunc(k_points, bathIntType, size):
    assert bathIntType in ("d", "p")
    k_points = np.asarray(k_points)
    kx_arr, ky_arr = map1Dto2D(k_points, size)
    if bathIntType == "d":
        return np.prod(np.cos(kx_arr) - np.cos(ky_arr))
    else:
        return 0.5 * (
            np.cos(kx_arr[0] + kx_arr[2] - kx_arr[1] - kx_arr[3])
            + np.cos(ky_arr[0] + ky_arr[2] - ky_arr[1] - ky_arr[3])
        )


def map2Dto1D(kx_val, ky_val, size):
    k_1Darr = np.linspace(K_MIN, K_MAX, size)
    kx_index = np.argmin(np.abs(k_1Darr - kx_val))
    ky_index = np.argmin(np.abs(k_1Darr - ky_val))
    return ky_index * size + kx_index


def map1Dto2D(points, size):
    kx = points % size
    ky = points // size
    k_1Darr = np.linspace(K_MIN, K_MAX, size)
    return k_1Darr[kx], k_1Darr[ky]


def tightBindDispersion(size, hop_t):
    k_1Darr = np.linspace(K_MIN, K_MAX, size)
    kx_arr, ky_arr = np.meshgrid(k_1Darr, k_1Darr)
    return -2 * hop_t * (np.cos(kx_arr) + np.cos(ky_arr))


def getDensityofStates(size, dispersionFunc, hop_t):
    k_1Darr = np.linspace(K_MIN, K_MAX, size)
    dispersion = dispersionFunc(size, hop_t)
    dispersion_x_plus1 = np.roll(dispersion, 1, axis=0)
    dispersion_x_minus1 = np.roll(dispersion, -1, axis=0)
    dispersion_y_plus1 = np.roll(dispersion, 1, axis=1)
    dispersion_y_minus1 = np.roll(dispersion, -1, axis=1)
    dOfStates = 4 / np.sqrt(
        (dispersion_x_plus1 - dispersion_x_minus1) ** 2
        + (dispersion_y_plus1 - dispersion_y_minus1) ** 2
    )
    dOfStates[dOfStates == np.inf] = np.amax(dOfStates[dOfStates != np.inf])

    return dOfStates


def getIsoEnergeticContour(dispersionArray, energy):
    difference = np.sign(dispersionArray.flatten() - energy)
    contourPoints = [
        point
        for point, diff in enumerate(difference[:-1])
        if difference[point] != difference[point + 1]
    ]
    return contourPoints


def deltaJ_k1k2(argsList):
    (index1, index2), args = argsList
    UVPoints = args["UVPoints"]
    kondoCoupPrev = args["kondoCoupPrev"]

    if args["proceedFlags"][index1, index2] == 0:
        renormalisation = 0
    bathIntfactor = args["bathIntCoup"] * np.array(
        [
            bathIntFunc(
                (holePoint, index1, index2, point), args["bathIntType"], args["size"]
            )
            for point, holePoint in zip(UVPoints, args["holeUVPoints"])
        ]
    )

    renormalisation = -args["deltaEnergy"] * sum(
        args["dOfStates"][UVPoints]
        * (
            kondoCoupPrev[index2][UVPoints].flatten()
            * kondoCoupPrev[index1][UVPoints].flatten()
            + 4 * args["kondoCoup_q_qbar"] * bathIntfactor
        )
        / args["denominators"]
    )
    # print(kondoCoupPrev[index1, index2], renormalisation)
    # print(kondoCoupPrev[index1, index2])
    if (kondoCoupPrev[index1, index2] + renormalisation) * kondoCoupPrev[
        index1, index2
    ] < 0:
        renormalisation = 0
    return renormalisation, index1, index2


def getDeltaJ(args):
    innerIndices = args["innerIndices"]
    UVenergy = args["UVenergy"]
    UVPoints = args["UVPoints"]
    proceedFlags = args["proceedFlags"]
    kondoCoupPrev = args["kondoCoupPrev"]
    kondoCoupNext = args["KondoCoupNext"]
    bathIntCoup = args["bathIntCoup"]
    size = args["size"]
    UVenergy = args["UVenergy"]
    omega = args["omega"]
    bathIntType = args["bathIntType"]
    bathIntUV = bathIntCoup * np.array(
        [
            bathIntFunc((point, point, point, point), bathIntType, size)
            for point in UVPoints
        ]
    )
    denominators = (
        omega
        - abs(UVenergy) / 2
        + kondoCoupPrev[UVPoints, UVPoints] / 4
        + bathIntUV / 2
    )

    holeUVPoints = [getHolePoint(point, size) for point in UVPoints]
    kondoCoup_q_qbar = np.array(
        [
            kondoCoupPrev[point][holePoint]
            for point, holePoint in zip(UVPoints, holeUVPoints)
        ]
    )
    args["holeUVPoints"] = holeUVPoints
    args["denominators"] = denominators
    args["kondoCoup_q_qbar"] = kondoCoup_q_qbar
    argsList = [
        ((index1, index2), args) for index1, index2 in product(innerIndices, repeat=2)
    ]
    for renormalisation, index1, index2 in Pool().map(deltaJ_k1k2, argsList):
        kondoCoupNext[index1, index2] += renormalisation
        if kondoCoupNext[index1, index2] * kondoCoupPrev[index1, index2] < 0:
            proceedFlags[index1, index2] = 0
    return kondoCoupNext, proceedFlags


def main(args):
    size_half = args["size_half"]
    hop_t = args["hop_t"]
    kondoCoup = args["kondoCoup"]
    bathIntCoup = args["bathIntCoup"]
    orbital = args["orbital"]

    assert orbital in ("d", "p")
    size = 2 * size_half + 1
    dispersionArray = tightBindDispersion(size, hop_t)
    dispersion1D = np.flip(np.sort(dispersionArray[0]))
    rgScales = dispersion1D[dispersion1D > 0]
    dOfStates = getDensityofStates(size, tightBindDispersion, hop_t).flatten()
    kondoCoupArray = np.zeros((len(dispersion1D), size**2, size**2))
    for index1 in tqdm(range(size**2), desc="Creating initial Kondo coupling matrix."):
        kx1, ky1 = map1Dto2D(index1, size)
        kx2_arr, ky2_arr = map1Dto2D(np.arange(size**2), size)
        if orbital == "d":
            kondoCoupArray[0][index1, np.arange(size**2)] = (
                kondoCoup
                * (np.cos(kx1) - np.cos(ky1))
                * (np.cos(kx2_arr) - np.cos(ky2_arr))
            )
        else:
            kondoCoupArray[0][index1, np.arange(size**2)] = kondoCoup * (
                np.cos(kx1 - kx2_arr) + np.cos(ky1 - ky2_arr)
            )
    proceedFlags = np.ones((size**2, size**2))
    for step, UVenergy in tqdm(enumerate(rgScales[:-1]), total=len(rgScales)):
        deltaEnergy = abs(rgScales[step] - rgScales[step + 1])
        kondoCoupArray[step + 1] = kondoCoupArray[step]
        UVPoints = getIsoEnergeticContour(dispersionArray, UVenergy)
        proceedFlags[UVPoints, :] = 0
        proceedFlags[:, UVPoints] = 0
        if np.all(proceedFlags == 0):
            break

        innerIndices = [
            point
            for point, energy in enumerate(dispersionArray.flatten())
            if energy < UVenergy
        ]
        args["innerIndices"] = innerIndices
        args["dOfStates"] = dOfStates
        args["UVenergy"] = UVenergy
        args["deltaEnergy"] = deltaEnergy
        args["UVPoints"] = UVPoints
        args["proceedFlags"] = proceedFlags
        args["kondoCoupPrev"] = kondoCoupArray[step]
        args["KondoCoupNext"] = kondoCoupArray[step + 1]
        args["bathIntCoup"] = bathIntCoup
        args["size"] = size
        args["UVenergy"] = UVenergy
        args["omega"] = -np.amax(dispersionArray) / 2
        args["bathIntType"] = orbital
        kondoCoupNext, proceedFlagsNew = getDeltaJ(args)
        proceedFlags = proceedFlagsNew
        kondoCoupArray[step + 1] = kondoCoupNext
    return kondoCoupArray
