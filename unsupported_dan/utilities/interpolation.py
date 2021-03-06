from scipy.spatial import cKDTree as kdTree
import numpy as np


def nn_evaluation(_fromCoords, _toCoords, n=1, weighted=False):

    """
    This function provides nearest neighbour information for uw swarms,
    given the "_toCoords", which could be the .data handle (coordinates) of a mesh or a differnet swarm,
    this function returns the indices of the n nearest neighbours in "_fromCoords" (will usually be swarm.particleCoordinates.data )
    it also returns the inverse-distance if weighted=True.

    The function works in parallel.

    The arrays come out a bit differently when used in nearest neighbour form
    (n = 1), or IDW: (n > 1). This is a result of how underworld's evaluate functions work
    The examples below show how to use nn_evaluation in each case.


    Usage n = 1:
    ------------
    ix, weights = nn_evaluation(swarm.particleCoordinates.data, mesh.data, n=n, weighted=False)
    toSwarmVar.data[:,0] =  fromSwarmVar.evaluate(fromSwarm)[_ix][:,0]

    Usage n > 1:
    ------------
    ix, weights = nn_evaluation(swarm.particleCoordinates.data, toSwarm.particleCoordinates.data, n=n, weighted=False)
    toSwarmVar.data[:,0] =  np.average(fromSwarmVar.evaluate(fromSwarm)[ix][:,:,0], weights=weights, axis=1)

    """

    if len(_toCoords) > 0: #this is required for safety in parallel

        #we rebuild the tree as we assume the fromSwarm is being advected
        tree = kdTree(_fromCoords)
        d, ix = tree.query(_toCoords, n)
        if n == 1:
            weights = np.ones(_toCoords.shape[0])
        elif not weighted:
            weights = np.ones((_toCoords.shape[0], n))*(1./n)
        else:
            weights = (1./d[:])/(1./d[:]).sum(axis=1)[:,None]
        return ix,  weights, d
    else:
        return  np.empty(0., dtype="int"),  np.empty(0., dtype="int"), np.empty(0., dtype="int")
