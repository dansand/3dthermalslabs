
# coding: utf-8

# In[ ]:

from unsupported_dan.interfaces.marker3D import markerSurface3D
import underworld as uw


res = 8
elementType="Q1/dQ0"

# lets do some 3d tests
minCoord = [-1.,-1.,-1.]
maxCoord = [ 1., 1., 1.]
mesh = uw.mesh.FeMesh_Cartesian(elementType='Q1', elementRes=(res,res,res), minCoord=minCoord, maxCoord=maxCoord)

velocityField   = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )


# In[4]:

import numpy as np
testPoints = np.linspace(-0.2, 0.2, 1000)
#print(testPoints)


# In[ ]:

#surface = markerSurface3D(mesh, velocityField, [], [], [], 0.2, 1.)
surface = markerSurface3D(mesh, velocityField, testPoints, testPoints, testPoints, 0.2, 1.)

