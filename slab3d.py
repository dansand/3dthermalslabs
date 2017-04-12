
# coding: utf-8

# ## 3D subduction
# 
# Some of the features of this model:
# 
# * use embedded manifolds to impose slabs / temperature field - here marker lines representing the top of the slabs
# * have particles only in some parts of the model
#     * i.e. only near surface, or where temp is below critical temp
# * switch between thermal and 'compositional' models:
#     * In all cases we will build up a proxy-temp field on the particles
#     * If we want to solve the energy equation, we need to map this proxy-temp to mesh.     
# * open / closed / periodic / free surface boundary conds. 
# 
# 
# 
# 

# In[79]:

import numpy as np
import underworld as uw
import math
from underworld import function as fn
import glucifer
import os
import sys
from easydict import EasyDict as edict
import operator



#
from unsupported_dan.utilities.interpolation import nn_evaluation
#from unsupported_dan.interfaces.marker2D import markerLine2D
#from unsupported_dan.faults.faults2D import fault2D, fault_collection
from unsupported_dan.interfaces.marker3D import markerSurface3D
from unsupported_dan.alchemy.materialGraph import MatGraph


# In[ ]:




# ## Setup output directories

# In[80]:

############
#Model letter and number
############


#Model letter identifier default
Model = "T"

#Model number identifier default:
ModNum = 0

#Any isolated letter / integer command line args are interpreted as Model/ModelNum

if len(sys.argv) == 1:
    ModNum = ModNum 
elif sys.argv[1] == '-f': #
    ModNum = ModNum 
else:
    for farg in sys.argv[1:]:
        if not '=' in farg: #then Assume it's a not a paramter argument
            try:
                ModNum = int(farg) #try to convert everingthing to a float, else remains string
            except ValueError:
                Model  = farg


# In[81]:

###########
#Standard output directory setup
###########

outputPath = "results" + "/" +  str(Model) + "/" + str(ModNum) + "/" 
imagePath = outputPath + 'images/'
filePath = outputPath + 'files/'
checkpointPath = outputPath + 'checkpoint/'
dbPath = outputPath + 'gldbs/'
xdmfPath = outputPath + 'xdmf/'
outputFile = 'results_model' + Model + '_' + str(ModNum) + '.dat'

if uw.rank()==0:
    # make directories if they don't exist
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    if not os.path.isdir(checkpointPath):
        os.makedirs(checkpointPath)
    if not os.path.isdir(imagePath):
        os.makedirs(imagePath)
    if not os.path.isdir(dbPath):
        os.makedirs(dbPath)
    if not os.path.isdir(filePath):
        os.makedirs(filePath)
    if not os.path.isdir(xdmfPath):
        os.makedirs(xdmfPath)
        
uw.barrier() #Barrier here so no procs run the check in the next cell too early


# ## Model parameters and scaling

# In[82]:

#1./1.87e9, 1./2.36e14


# In[83]:

dp = edict({})
#Main physical paramters
dp.depth=1200e3                         #Depth
dp.refDensity=3300.                        #reference density
dp.refGravity=9.8                          #surface gravity
dp.refViscosity=1.8e19                       #reference upper mantle visc.
dp.refDiffusivity=1e-6                     #thermal diffusivity
dp.refExpansivity=3e-5                     #surface thermal expansivity
dp.gasConstant=8.314                    #gas constant
dp.specificHeat=1250.                   #Specific heat (Jkg-1K-1)
dp.potentialTemp=1573.                  #mantle potential temp (K)
dp.surfaceTemp=273.                     #surface temp (K)
#Rheology - flow law paramters
dp.cohesionMantle=20e6                   #mantle cohesion in Byerlee law
dp.cohesionCrust=2e6                    #crust cohesion in Byerlee law
dp.frictionMantle=0.2                   #mantle friction coefficient in Byerlee law (tan(phi))
dp.frictionCrust=0.02                   #crust friction coefficient 
dp.diffusionPreExp=5.34e-10             #1./1.87e9, pre-exp factor for diffusion creep
dp.diffusionEnergy=3e5 
dp.diffusionVolume=5e-6
dp.lowerMantlePreExp=4.23e-15           #1./2.36e14
dp.lowerMantleEnergy=2.0e5
dp.lowerMantleVolume=1.5e-6
    #dp.dislocationPreExp=5e-16              #pre-exp factor for dislocation creep
    #dp.peierlsPreExp=1e-150                 #pre-exp factor for Peierls creep
    #dp.dislocationEnergy=5.4e5
    #dp.peierlsEnergy=5.4e5
    #dp.dislocationVolume=12e-6
    #dp.peierlsVolume=10e-6

    #dp.dislocationExponent=3.5              #Dislocation creep stress exponent
    #dp.peierlsExponent=20.                  #Peierls creep stress exponent 

#Rheology - cutoff values
dp.viscosityMin=dp.refViscosity*5e-2
dp.viscosityMax=dp.refViscosity*1e3                 #viscosity max in the mantle material
dp.viscosityMinCrust=1e20               #viscosity min in the weak-crust material
dp.viscosityMaxCrust=1e20               #viscosity max in the weak-crust material
dp.yieldStressMax=300*1e6              #
#Intrinsic Lengths
dp.mantleCrustDepth=20.*1e3              #Crust depth
dp.faultThickness = 25.*1e3              #interface material (crust) an top of slabs
dp.crustMantleDepth=250.*1e3 
dp.lowerMantleDepth=660.*1e3  
dp.crustLimitDepth=250.*1e3             #Deeper than this, crust material rheology reverts to mantle rheology
#Slab and plate init. parameters
dp.subZoneLoc=3000e3                    #Y position of subduction zone...km
dp.maxDepth=250e3
dp.theta=40                             #Angle of slab
dp.radiusOfCurv = 250e3                          #radius of curvature
dp.slabMaxAge=100e6                     #age of subduction plate at trench
dp.plateMaxAge=100e6                    #max age of slab (Plate model)
dp.opMaxAge=100e6                       #age of op
#Misc
dp.stickyAirDepth=100e3                 #depth of sticky air layer
dp.viscosityStickyAir=1e19              #stick air viscosity, normal
dp.lowerMantleViscFac=30.
#derived params
dp.deltaTemp = dp.potentialTemp-dp.surfaceTemp
dp.tempGradMantle = (dp.refExpansivity*dp.refGravity*(dp.potentialTemp))/dp.specificHeat
dp.tempGradSlab = (dp.refExpansivity*dp.refGravity*(dp.surfaceTemp + 400.))/dp.specificHeat



#Modelling and Physics switches

md = edict({})
md.aspectX = 1.0
md.aspectY = 4.
md.refineMeshStatic=False
md.stickyAir=False
md.aspectRatio=5.
md.res=16
md.ppc=25                                 #particles per cell
md.elementType="Q1/dQ0"
#md.elementType="Q2/DPC1"
md.secInvFac=math.sqrt(1.)
md.courantFac=0.5                         #extra limitation on timestepping
md.thermal = False                        #thermal system or compositional
md.swarmInitialFac = 0.6                 #initial swarm layout will be int(md.ppc*md.swarmInitialFac), popControl will densify later
md.compBuoyancy = False
md.uniformAge = True



# In[84]:

####TEST BLOCK, smaller activation energy

fracE = 0.6 #We want to multiply the activation Energy by this value
delE= dp.diffusionEnergy - dp.diffusionEnergy*fracE
dp.diffusionEnergy *= fracE
dp.diffusionVolume *=0.8
dp.diffusionPreExp /= np.exp(delE /(dp.gasConstant*dp.potentialTemp))


# In[85]:

##Parse any command-line args

from unsupported_dan.cl_args import easy_args
sysArgs = sys.argv

#We want to run this on both the paramter dict, and the model dict
easy_args(sysArgs, dp)
easy_args(sysArgs, md)


# In[86]:

sf = edict({})

sf.lengthScale=2900e3
sf.refViscosity = dp.refViscosity
sf.stress = (dp.refDiffusivity*dp.refViscosity)/sf.lengthScale**2
sf.lithGrad = dp.refDensity*dp.refGravity*(sf.lengthScale)**3/(dp.refViscosity*dp.refDiffusivity) 
sf.lithGrad = (dp.refViscosity*dp.refDiffusivity) /(dp.refDensity*dp.refGravity*(sf.lengthScale)**3)
sf.velocity = sf.lengthScale/dp.refDiffusivity
sf.strainRate = dp.refDiffusivity/(sf.lengthScale**2)
sf.time = 1./sf.strainRate
sf.actVolume = (dp.gasConstant*dp.deltaTemp)/(dp.refDensity*dp.refGravity*sf.lengthScale)
sf.actEnergy = (dp.gasConstant*dp.deltaTemp)
sf.diffusionPreExp = 1./dp.refViscosity
sf.deltaTemp  = dp.deltaTemp

#sf.dislocationPreExp = ((dp.refViscosity**(-1.*dp.dislocationExponent))*(dp.refDiffusivity**(1. - dp.dislocationExponent))*(sf.lengthScale**(-2.+ (2.*dp.dislocationExponent)))),
#sf.peierlsPreExp = 1./2.6845783276046923e+40 #same form as Ads, but ndp.np =20. (hardcoded because numbers are too big)



#dimesionless params
ndp  = edict({})

ndp.rayleigh = (dp.refExpansivity*dp.refDensity*dp.refGravity*dp.deltaTemp*sf.lengthScale**3)/(sf.refViscosity*dp.refDiffusivity)

ndp.potentialTemp = dp.potentialTemp/sf.deltaTemp
ndp.surfaceTemp = dp.surfaceTemp/sf.deltaTemp 
ndp.tempGradMantle = dp.tempGradMantle/(sf.deltaTemp/sf.lengthScale)
ndp.tempGradSlab = dp.tempGradSlab/(sf.deltaTemp/sf.lengthScale)

#lengths / distances
ndp.depth = dp.depth/sf.lengthScale
ndp.yLim = ndp.depth*md.aspectY
ndp.xLim = ndp.depth*md.aspectX
ndp.faultThickness = dp.faultThickness/sf.lengthScale
ndp.mantleCrustDepth =  dp.mantleCrustDepth/sf.lengthScale
ndp.crustLimitDepth = dp.crustLimitDepth/sf.lengthScale
ndp.lowerMantleDepth = dp.lowerMantleDepth/sf.lengthScale

#times - for convenience the dimensional values are in years, conversion to seconds happens here
ndp.slabMaxAge =  dp.slabMaxAge*(3600*24*365)/sf.time
ndp.plateMaxAge =  dp.plateMaxAge*(3600*24*365)/sf.time
ndp.opMaxAge = dp.opMaxAge*(3600*24*365)/sf.time


#Rheology - flow law paramters
ndp.cohesionMantle=dp.cohesionMantle/sf.stress                  #mantle cohesion in Byerlee law
ndp.cohesionCrust=dp.cohesionCrust/sf.stress                  #crust cohesion in Byerlee law
ndp.frictionMantle=dp.frictionMantle/sf.lithGrad                  #mantle friction coefficient in Byerlee law (tan(phi))
ndp.frictionCrust=dp.frictionCrust/sf.lithGrad                  #crust friction coefficient 
ndp.diffusionPreExp=dp.diffusionPreExp/sf.diffusionPreExp                #pre-exp factor for diffusion creep
ndp.diffusionEnergy=dp.diffusionEnergy/sf.actEnergy
ndp.diffusionVolume=dp.diffusionVolume/sf.actVolume
ndp.lowerMantlePreExp=dp.lowerMantlePreExp/sf.diffusionPreExp 
ndp.lowerMantleEnergy=dp.lowerMantleEnergy/sf.actEnergy
ndp.lowerMantleVolume=dp.lowerMantleVolume/sf.actVolume
ndp.yieldStressMax=dp.yieldStressMax/sf.stress 
#Rheology - cutoff values
ndp.viscosityMin= dp.viscosityMin /sf.refViscosity
ndp.viscosityMax=dp.viscosityMax/sf.refViscosity

ndp.viscosityMinCrust= dp.viscosityMinCrust /sf.refViscosity
ndp.viscosityMaxCrust = dp.viscosityMaxCrust/sf.refViscosity


#Slab and plate init. parameters
ndp.subZoneLoc = dp.subZoneLoc/sf.lengthScale
ndp.maxDepth = dp.maxDepth/sf.lengthScale
ndp.radiusOfCurv = dp.radiusOfCurv/sf.lengthScale




# In[87]:

#Domain and Mesh paramters
zres = int(md.res)
yres = int(md.res*md.aspectY)
xres = int(md.res*md.aspectX) 



mesh = uw.mesh.FeMesh_Cartesian( elementType = (md.elementType),
                                 elementRes  = (xres, yres, zres), 
                                 minCoord    = (0., 0., 1. - ndp.depth), 
                                 maxCoord    = (ndp.xLim,ndp.yLim, 1.)) 

velocityField   = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=3 )
pressureField   = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )

temperatureField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )

if md.thermal:
    temperatureDotField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 ) #create this only if Adv-diff
    diffusivityFn = fn.misc.constant(1.)


# In[ ]:




# ## miscellaneous Python functions 
# 

# In[88]:

def bbox(mesh):
    return ((mesh.minCoord[0], mesh.minCoord[1], mesh.minCoord[2]),(mesh.maxCoord[0], mesh.maxCoord[1], mesh.minCoord[2]))


# In[89]:

## general underworld2 functions 


coordinate = fn.input()
depthFn = mesh.maxCoord[2] - coordinate[2] #a function providing the depth


xFn = coordinate[0]  #a function providing the x-coordinate
yFn = coordinate[1]


#Create a binary circle
def inCircleFnGenerator(centre, radius):
    coord = fn.input()
    offsetFn = coord - centre
    return fn.math.dot( offsetFn, offsetFn ) < radius**2





# In[ ]:




# In[90]:

mesh.minCoord, mesh.maxCoord


# ## 1. Static Mesh refinement

# In[91]:

if md.refineMeshStatic:
    mesh.reset()

    jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
    yFn = coordinate[1]
    yField = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=1 )
    yField.data[:] = 0.
    yBC = uw.conditions.DirichletCondition( variable=yField, indexSetsPerDof=(jWalls,) )

    # set bottom wall temperature bc
    for index in mesh.specialSets["MinJ_VertexSet"]:
        yField.data[index] = mesh.minCoord[1]
    # set top wall temperature bc
    for index in mesh.specialSets["MaxJ_VertexSet"]:
        yField.data[index] = mesh.maxCoord[1]



    s = 2.5
    intensityFac = 1.5
    intensityFn = (((yFn - mesh.minCoord[1])/(mesh.maxCoord[1]-mesh.minCoord[1]))**s)
    intensityFn *= intensityFac
    intensityFn += 1.


    yLaplaceEquation = uw.systems.SteadyStateHeat(temperatureField=yField, fn_diffusivity=intensityFn, conditions=[yBC,])

    # get the default heat equation solver
    yLaplaceSolver = uw.systems.Solver(yLaplaceEquation)
    # solve
    yLaplaceSolver.solve()


    #Get the array of Y positions - copy may be necessary, not sure. 
    newYpos = yField.data.copy() 

    uw.barrier()
    with mesh.deform_mesh():
         mesh.data[:,1] = newYpos[:,0]


# In[92]:

#fig= glucifer.Figure(quality=3)

#fig.append(glucifer.objects.Mesh(mesh, opacity=0.8))
#fig.append(glucifer.objects.Surface(mesh, intensityFn))

#fig.show()
#fig.save_database('test.gldb')


# In[ ]:




# ## Boundary Conditions

# In[93]:

#Stokes BCs

iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
kWalls = mesh.specialSets["MinK_VertexSet"] + mesh.specialSets["MaxK_VertexSet"]

tWalls = mesh.specialSets["MaxK_VertexSet"]
bWalls =mesh.specialSets["MinK_VertexSet"]
      
        
freeslipBC = uw.conditions.DirichletCondition( variable      = velocityField, 
                                               indexSetsPerDof = ( iWalls, jWalls, kWalls) )


# In[94]:

#Energy BCs

if md.thermal:
    dirichTempBC = uw.conditions.DirichletCondition(     variable=temperatureField, 
                                              indexSetsPerDof=(tWalls,) )


# ## Swarm

# In[95]:

#create material swarm
swarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
layout = uw.swarm.layouts.PerCellRandomLayout(swarm=swarm, particlesPerCell=int(md.ppc*md.swarmInitialFac))
swarm.populate_using_layout( layout=layout ) # Now use it to populate.


materialVariable      = swarm.add_variable( dataType="int", count=1 )
signedDistanceVariable = swarm.add_variable( dataType="double", count=1 )
#plateVariable = swarm.add_variable( dataType="double", count=1 )
directorVector   = swarm.add_variable( dataType="double", count=3)
proxyTempVariable = swarm.add_variable( dataType="double", count=1 )

directorVector.data[:,:] = 0.0
#plateVariable.data[:] = 0
signedDistanceVariable.data[:] = 0.0
proxyTempVariable.data[:] = 0.0


# ## Materials

# In[96]:

#Materials
mantleID = 0
crustID = 1
airID = 2      #in case we use sticky air


# Swarm variables
materialVariable.data[:] = mantleID

#list of all material indexes
material_list = [mantleID, crustID, airID]


# In[97]:

#mesh.maxCoord


# ## Initial Conditions

# In[98]:

proxyageFn = fn.branching.conditional([(yFn < ndp.subZoneLoc, ndp.slabMaxAge*fn.math.abs(yFn)), #idea is to make this arbitrarily complex
                                  (True, ndp.opMaxAge)])



if md.uniformAge:
    sig = 150e3/sf.lengthScale
    ridgeFn = 1. -                  fn.math.exp(-1.*(yFn - 0.)**2/(2 * sig**2))-                 fn.math.exp(-1.*(yFn - mesh.maxCoord[1])**2/(2 * sig**2))
    
        
    proxyageFn = fn.branching.conditional([(yFn < ndp.subZoneLoc, ridgeFn*ndp.slabMaxAge), #idea is to make this arbitrarily complex
                                  (True, ridgeFn*ndp.opMaxAge)])
    



thicknessAtTrench = 2.3*math.sqrt(1.*ndp.slabMaxAge)


# ### Marker Surface for slab

# #create coordinates for markrSurface
# SPACING = 5e3/sf.lengthScale #points at .. km intervals
# radians = (dp.theta*math.pi)/180.
# dydx = math.tan(radians)
# ylimslab = ndp.subZoneLoc + ndp.maxDepth/dydx
# 
# slabxs = np.arange(mesh.minCoord[0], mesh.maxCoord[0], SPACING)
# slabys = np.arange(ndp.subZoneLoc, ylimslab,math.cos(radians)*SPACING)
# coords = np.meshgrid(slabxs, slabys)
# xs = coords[0].flatten()
# ys = coords[1].flatten()
# 
# zs = 1.- (ys - ndp.subZoneLoc)*dydx

# In[99]:

def slab_top(trench, normal, gradientFn, ds, maxDepth, mesh):
    """
    Create points representing the top of a slab from trench to maxDepth
    
    Parameter
    ---------
    trench : list or list like 
            Points represnting trench location, 
    normal: list or list like
            vector in the horizontal plane normal to trench
    gradientFn: function
             function that returns the dip or the slab dz/ds 
             where s is the distance along the normal vector
    ds: float
            distance between points, in model coordinates
    
    max depth: float, or list or list like
            Maximum depth of slab
    mesh: uw 2 mesh   
    
    """
    
    #convert everything to numpy arrays
    trench = np.array(trench)
    normal = np.array(normal)/np.linalg.norm(normal)
    maxDepth = np.array(maxDepth)
    
    #test if gradientFn is a function  
    #to do
    
    #
    points = []
    points.append(list(trench))
    
    #set starting values
    #normal/= np.linalg.norm(normal)#unitize
    vertical = np.zeros(mesh.dim)
    vertical[-1] = -1.

    P0 = trench.copy()
    F = gradientFn(0.)
    #print(F)
    H = 0.
    V = 0.
    #print(normal, F)
    S = normal.copy()

    S[-1] = F     #assumes the last component is vertical
    S/= np.linalg.norm(S) #unitize

    
    while V < maxDepth:
        
        #Get next set of points
        #print(S*ds)
        P1 = P0 + S*ds
        points.append(list(P1))
        P0 = P1
        
        #update, H, V, F, S
        H +=  np.dot(S*ds, normal)
        V +=  abs(((S*ds)[-1]))
        F = gradientFn(H)        
        S = normal.copy()
        S[-1] = F     #assumes the last component is vertical
        S/= np.linalg.norm(S) #unitize

        
        
        
    return(np.array(points))
    


# In[100]:

def polyGradientFn(S):
    if S == 0.:
        return 0.
    else:
        return -1*(S/ndp.radiusOfCurv)**2


# In[101]:

ds = 5e3/sf.lengthScale
normal = [0.,1., 0.]

trenchxs = np.arange(mesh.minCoord[0], mesh.maxCoord[0], ds)
trenchys = np.ones(trenchxs.shape)*ndp.subZoneLoc
trenchzs = np.ones(trenchxs.shape)


trench = np.column_stack((trenchxs, trenchys,trenchzs))


# In[102]:

#trench


# In[103]:

slabdata = slab_top(trench, normal, polyGradientFn, ds, ndp.maxDepth, mesh)


# In[104]:

slabxs = slabdata[:,:,0].flatten()
slabys = slabdata[:,:,1].flatten()
slabzs = slabdata[:,:,2].flatten()


# In[105]:

#create the makerSurface

slabTop = markerSurface3D(mesh, velocityField, slabxs, slabys ,slabzs , thicknessAtTrench, 1.)



# In[106]:

#Assign the signed distance for the slab
#in this case we only want the portion where the signed distance is positive


#Note distance=2.*thicknessAtTrench: we actually want to allow distance greater than thicknessAtTrench in the kDTree query, 
#as some of these distances will not be orthogonal to the marker line
#The dot product in the function will project these distances onto the normal vector
#We'll cull distances greater than thicknessAtTrench with a numpy boolean slice
#this is a big advantage in parallel


sd, pts = slabTop.compute_signed_distance(swarm.particleCoordinates.data, distance=2.*thicknessAtTrench)
signedDistanceVariable.data[np.logical_and(sd>0, sd<=slabTop.thickness)] = sd[np.logical_and(sd>0, sd<=slabTop.thickness)]


#sp, pts0 = slabLine.compute_marker_proximity(swarm.particleCoordinates.data)
#plateVariable.data[np.logical_and(sd>0,sp == slabLine.ID)] = sp[np.logical_and(sd>0,sp == slabLine.ID)]



# In[107]:

#slabCirc = inCircleFnGenerator((ndp.subZoneLoc, 1.0), ndp.maxDepth)


bufferlength = 1e3/sf.lengthScale

plateDepthFn = fn.branching.conditional([(depthFn < thicknessAtTrench, depthFn),
                                        (True, 1.)])

#plateTempProxFn = fn.math.erf((depthFn*sf.lengthScale)/(2.3*fn.math.sqrt(dp.refDiffusivity*proxyageFn)))
plateTempProxFn = fn.math.erf((plateDepthFn)/(2.3*fn.math.sqrt(1.*proxyageFn)))

#slabTempProx  = fn.math.erf((signedDistanceVariable*sf.lengthScale)/(2.3*fn.math.sqrt(dp.refDiffusivity*proxyageFn)))
slabTempProx  = fn.math.erf((signedDistanceVariable)/(2.3*fn.math.sqrt(1.*proxyageFn)))


#proxytempConds = fn.branching.conditional([(signedDistanceVariable < bufferlength, plateTempProxFn),
#                          (slabCirc, fn.misc.min(slabTempProx , plateTempProxFn)),                 
#                          (True, plateTempProxFn)]) #take the min of the plate and slab thermal stencil 


proxytempConds = fn.branching.conditional([(signedDistanceVariable < bufferlength, plateTempProxFn), #What is this?
                          (depthFn > ndp.maxDepth, 1.),                 
                          (True, fn.misc.min(slabTempProx , plateTempProxFn)) ]) #take the min of the plate and slab thermal stenc





proxyTempVariable.data[:] = proxytempConds.evaluate(swarm)


# In[ ]:




# In[108]:

#proxyTempVariable.data.max()


# In[109]:

print('test Point')


# ## Mask variable for viz

# In[110]:

bBox = bbox(mesh)
bBox


# In[111]:

vizVariable      = swarm.add_variable( dataType="int", count=1 )


vizConds = fn.branching.conditional([(proxyTempVariable < 0.9*1., 1),
                          (True, 0)]) 

vizVariable.data[:] = vizConds.evaluate(swarm)


# In[112]:

#fn_mask=vizVariable


# In[113]:

swarmfig = glucifer.Figure(figsize=(800,400), boundingBox=bBox)
swarmfig.append( glucifer.objects.Points(swarm, proxyTempVariable, fn_mask=vizVariable) )
#swarmfig.show()
swarmfig.save_database('test.gldb')


# In[114]:

#print('got to first update')


# distance = w0
# d, p  = slabLine.kdtree.query( swarm.particleCoordinates.data, distance_upper_bound=distance )
# fpts = np.where( np.isinf(d) == False )[0]
# director = np.zeros_like(swarm.particleCoordinates.data)  # Let it be zero outside the region of interest
# director = slabLine.director.data[p[fpts]]
# vector = swarm.particleCoordinates.data[fpts] - slabLine.kdtree.data[p[fpts]]
# np.linalg.norm(vector, axis = 1)

# ## Fault / interface

# In[93]:

def copy_markerSurface3D(ml, thickness=False, ID=False):
    
    """
    I think this is safe in parallel...
    
    """
    if not thickness:
        thickness = ml.thickness
    if not ID:
        ID = -1*ml.ID
    new_line = markerSurface3D(mesh, velocityField, [], [],[], thickness,  ID)
    if ml.swarm.particleCoordinates.data.shape[0] > 0:
        new_line.swarm.add_particles_with_coordinates(ml.swarm.particleCoordinates.data.copy())
        
    new_line.rebuild()
    return new_line


# In[99]:

#Build fault
fault = markerSurface3D(mesh, velocityField, slabxs, slabys ,slabzs, ndp.faultThickness, 1.)


with fault.swarm.deform_swarm():
    fault.swarm.particleCoordinates.data[:] += fault.director.data*ndp.faultThickness
    
fault.rebuild()
fault.swarm.update_particle_owners()


# In[100]:

#inform the mesh of the fault

sd, pts0 = fault.compute_signed_distance(swarm.particleCoordinates.data, distance=thicknessAtTrench)
sp, pts0 = fault.compute_marker_proximity(swarm.particleCoordinates.data)

materialVariable.data[np.logical_and(sd<0,sp == fault.ID)] = sp[np.logical_and(sd<0,sp == fault.ID)]


if directorVector.data.shape[0]:
    dv, nzv = fault.compute_normals(swarm.particleCoordinates.data)
    if directorVector.data[nzv].shape[0]:
        directorVector.data[nzv] = dv[nzv]


# In[ ]:




# In[101]:

#Copy the fault and jitter, this is the swarm we'll capture inteface details on 

metricSwarm  = copy_markerSurface3D(fault)

ds = ndp.faultThickness/2.
with metricSwarm.swarm.deform_swarm():
    metricSwarm.swarm.particleCoordinates.data[...] -= metricSwarm.director.data[...]*ds


# In[104]:

swarmfig = glucifer.Figure(figsize=(800,400), boundingBox=bBox)
swarmfig.append( glucifer.objects.Points(swarm, proxyTempVariable, fn_mask=vizVariable) )
swarmfig.append( glucifer.objects.Points(fault.swarm) )
swarmfig.append( glucifer.objects.Points(slabTop.swarm) )


#swarmfig.show()
#swarmfig.save_database('test.gldb')


# In[ ]:




# ## Temperature field
# 
# This bit needs work

# In[105]:

mesh.minCoord[1]


# In[106]:

def swarmToTemp():

    _ix, _weights, _dist = nn_evaluation(swarm.particleCoordinates.data, mesh.data, n=2, weighted=True)


    temperatureField.data[:] = 1.0 #first set to dimensionless potential temp

    #this was to provide a bit of safety in case we don't use particles everywhere. Currently not needed
   # tempMapTol = 0.2
    #tempMapMask = _dist.min(axis=1) < tempMapTol*(1. - mesh.minCoord[1])/mesh.elementRes[1] 
    #temperatureField.data[:,0][tempMapMask] = np.average(proxyTempVariable.evaluate(swarm)[_ix][tempMapMask][:,:,0],weights=_weights[tempMapMask], axis=1)


    temperatureField.data[:,0] = np.average(proxyTempVariable.evaluate(swarm)[_ix][:,:,0],weights=_weights, axis=1)
    #now used IDW to assign temp from particles to Field
    #this is looking pretty ugly; nn_evaluation could use some grooming

    #now cleanup any values that have fallen outside the Bcs

    temperatureField.data[temperatureField.data > 1.] = 1.
    temperatureField.data[temperatureField.data < 0.] = 0.
    
    #and cleanup the BCs
    
    temperatureField.data[bWalls.data] = 1.
    temperatureField.data[tWalls.data] = 0.


# In[107]:

#map proxy temp (swarm var) to mesh variable
swarmToTemp()


# In[108]:

#fig= glucifer.Figure(quality=3, boundingBox=bBox)

#fig.append( glucifer.objects.Mesh(mesh ))
#fig.append( glucifer.objects.Points(swarm, temperatureField, pointSize=2,fn_mask=vizVariable))
#
#fig.show()
#fig.save_database('temp.gldb')


# In[ ]:




# ## adiabatic temp correction

# In[109]:

#(w0*sf.lengthScale)/(2.*np.sqrt(dp.refDiffusivity*ageAtTrenchSeconds))


# In[110]:

#Adiabatic correction: this is added to the arrhenius laws to simulate the adiabatic component
#We'll use a double linearisation of the adiabatic temp function:

#ndp.tempGradMantle linearised at the mantle potential temp
#dp.tempGradSlab linearised at typical slab temp


thicknessAtTrench = 2.3*math.sqrt(1.*ndp.slabMaxAge)
tempAtTrench  = math.erf((thicknessAtTrench)/(2.3*math.sqrt(1.*ndp.slabMaxAge))) #this is the isotherm used to define the slab / mantle boundary


dp.tempGradMantle, dp.tempGradSlab

if md.thermal:
    adiabaticCorrectFn = fn.branching.conditional([(temperatureField > tempAtTrench, depthFn*ndp.tempGradMantle), #idea is to make this arbitrarily complex
                                      (True, depthFn*ndp.tempGradSlab) ])
else:
    adiabaticCorrectFn = fn.branching.conditional([(proxyTempVariable > tempAtTrench, depthFn*ndp.tempGradMantle), #idea is to make this arbitrarily complex
                                      (True, depthFn*ndp.tempGradSlab) ])


#This will need alteration if we are using non Global particles. The adiabaticCorrectFn will need to be wrapped in a 
#swarm.fn_particle_found() conditional


# In[111]:

#fig= glucifer.Figure(quality=3, boundingBox= bBox)

#fig.append( glucifer.objects.Mesh(mesh ))
#fig.append( glucifer.objects.Points(swarm, proxyTempVariable + adiabaticCorrectFn, pointSize=1))

#fig.show()


# ## Swarm densification
# 
# Try to build our initial geometry with less particles that are required dynamically,
# 
# Then use population_control to fill out the swarm

# In[112]:

population_control = uw.swarm.PopulationControl(swarm, deleteThreshold=0.006, splitThreshold=0.25, maxDeletions=1, maxSplits=3, aggressive=True,aggressiveThreshold=0.9, particlesPerCell=int(md.ppc))


#


# def repopulate():
#     thresh = 5000.
#     diff = thresh + 1
#     count = 1
#     maxLoops = 10
#     pg = np.copy(swarm.particleGlobalCount)
#     while abs(diff) > thresh and count < maxLoops + 1:
#         population_control.repopulate()
#         diff = swarm.particleGlobalCount - pg
#         pg = swarm.particleGlobalCount
#         #print(str(count), str(pg), str(diff))
#         count += 1
#     

# In[113]:

#keep it simple for now
def repopulate():
    population_control.repopulate()


# In[114]:

repopulate()


# In[115]:

((float(swarm.particleGlobalCount)/mesh.elementsGlobal))/md.ppc


# ## MOR restriction Fn
# 
# Use the temperature gradient to define a restriction around the Ridges
# This coud be used to determin locations for crust creation, etc.

# In[116]:

#depthMorTest = 20e3/sf.lengthScale

#nearSurfTempGrad = fn.branching.conditional( ((depthFn < depthMorTest, temperatureField.fn_gradient[2] ), 
#                                           (True,                      0.)  ))

#morRestrictFn = fn.math.abs(nearSurfTempGrad) < 50.


# In[117]:

#fig= glucifer.Figure(quality=3)
#fig.append( glucifer.objects.Surface(mesh,repopMaskFn))
#fig.append( glucifer.objects.Points(swarm, temperatureField.fn_gradient[2], pointSize=2))

#fig.show()


# ##  Material Graph

# In[118]:

###################
#initial particle layout
###################

#Setup the graph object
MG = MatGraph()

#First thing to do is to add all the material types to the graph (i.e add nodes)
MG.add_nodes_from(material_list)

#mantle  => crust
MG.add_transition((mantleID,crustID), depthFn, operator.lt, ndp.mantleCrustDepth)
MG.add_transition((mantleID,crustID), yFn, operator.gt, 2.*thicknessAtTrench)   #This distance away from the ridge
MG.add_transition((mantleID,crustID), yFn, operator.lt, ndp.subZoneLoc)

                  
#crust  => mantle                
MG.add_transition((crustID, mantleID), depthFn, operator.gt, ndp.crustLimitDepth)


MG.build_condition_list(materialVariable)
materialVariable.data[:] = fn.branching.conditional(MG.condition_list).evaluate(swarm)


# In[119]:

#Final particle transformation rules
#restrict crust creation - avoid crust on the upper plate

MG.remove_edges_from([(mantleID,crustID)])

#mantle  => crust
MG.add_transition((mantleID,crustID), depthFn, operator.lt, ndp.mantleCrustDepth)
MG.add_transition((mantleID,crustID), yFn, operator.gt, 2.*thicknessAtTrench) 
MG.add_transition((mantleID,crustID), yFn, operator.lt, ndp.subZoneLoc/2.)   
MG.add_transition((mantleID,crustID), xFn, operator.lt, (mesh.minCoord[0] - ndp.subZoneLoc)/2.)

MG.build_condition_list(materialVariable)


# In[122]:

fig3= glucifer.Figure(quality=3, boundingBox=bBox)
fig3.append( glucifer.objects.Points(swarm,materialVariable, pointSize=2, fn_mask=vizVariable))
fig3.append( glucifer.objects.Mesh(mesh,opacity=0.2))

#fig1.append( glucifer.objects.Points(metricSwarm.swarm, pointSize=1))
#fig3.show()
#fig3.save_database('mat.gldb')


# ## choose temp field to use

# In[103]:

if md.thermal:
    temperatureFn = temperatureField
else:
    temperatureFn = proxyTempVariable


# ## Rheology

# In[54]:

#fault_coll = fault_collection([fault])

#Set up any functions required by the rheology
strainRate_2ndInvariant = fn.tensor.second_invariant( 
                            fn.tensor.symmetric( 
                            velocityField.fn_gradient ))

#total pressure (i.e dimensional total pressure, multiplied by the Stress scaling)
totalPressureFn = pressureField + (depthFn*dp.depth*dp.refDensity*dp.refGravity)*sf.stress #assumes surface dynamic pressure integral is set to zero


def safe_visc(func, viscmin=ndp.viscosityMin, viscmax=ndp.viscosityMax):
    return fn.misc.max(viscmin, fn.misc.min(viscmax, func))


#edotn_SFn, edots_SFn = fault_coll.global_fault_strainrate_fns(velocityField, directorVector, proximityVariable)


# In[55]:

##Diffusion Creep
diffusionUM = (1./ndp.diffusionPreExp)*            fn.math.exp( ((ndp.diffusionEnergy + (depthFn*ndp.diffusionVolume))/((temperatureFn+ adiabaticCorrectFn + ndp.surfaceTemp))))

diffusionLM = (1./ndp.lowerMantlePreExp)*            fn.math.exp( ((ndp.lowerMantleEnergy + (depthFn*ndp.lowerMantleVolume))/((temperatureFn+ adiabaticCorrectFn + ndp.surfaceTemp))))

diffusion = fn.branching.conditional( ((depthFn < ndp.lowerMantleDepth, diffusionUM ), 
                                           (True,                      diffusionLM )  ))
    
diffusion = safe_visc(diffusion, viscmax=1e5)
    
#Define the mantle Plasticity
ys =  ndp.cohesionMantle + (depthFn*ndp.frictionMantle)
ysf = fn.misc.min(ys, ndp.yieldStressMax)
yielding = ysf/(2.*(strainRate_2ndInvariant) + 1e-15) 

##Crust plasticity
crustys =  ndp.cohesionCrust + (depthFn*ndp.frictionCrust)
crustysf = fn.misc.min(crustys, ndp.yieldStressMax)
crustYielding = crustysf/(2.*(strainRate_2ndInvariant)) 



#combined rheologies

mantleViscosityFn = safe_visc(fn.misc.min(diffusion, yielding), viscmax=ndp.viscosityMax)
interfaceViscosityFn = safe_visc(fn.misc.min(diffusion, crustYielding), viscmax=ndp.viscosityMaxCrust)


# In[ ]:




# In[56]:

viscosityMapFn = fn.branching.map( fn_key = materialVariable,
                         mapping = {0:mantleViscosityFn,
                                    1:interfaceViscosityFn} )




# In[57]:

#fig= glucifer.Figure(quality=3)

#fig.append( glucifer.objects.Points(upperplate.swarm, pointSize=2))
#fig.append( glucifer.objects.Mesh(mesh, opacity=0.2))
#fig.append( glucifer.objects.Points(swarm, viscosityMapFn, pointSize=2, logScale=True))


#fig.show()
#fig.save_image('visc.png')


# ## Buoyancy 

# In[58]:

#md.compBuoyancy = True


# In[59]:

#Thermal Buoyancy

z_hat = ( 0.0,0.0, 1.0 )


if md.thermal:
    thermalBuoyancyFn = ndp.rayleigh*temperatureField
else:
    thermalBuoyancyFn = ndp.rayleigh*proxyTempVariable


# In[60]:

#Set up compositional buoyancy contributions


buoyancy_factor = (dp.refGravity*sf.lengthScale**3)/(dp.refViscosity*dp.refDiffusivity)

air_comp_buoyancy  = (dp.refDensity - 1000.)*buoyancy_factor          #roughly sea-water density
basalt_comp_buoyancy  = (dp.refDensity - 2940.)*buoyancy_factor       #
#harz_comp_buoyancy = (dp.refDensity - 3235.)*buoyancy_factor
pyrolite_comp_buoyancy = (dp.refDensity - 3300.)*buoyancy_factor


#if we're using a thicker crust for numerical resolution, make a bouyancy adjustment
averageCrustThickness = 6e3
basalt_comp_buoyancy *=(averageCrustThickness/dp.mantleCrustDepth)


# In[61]:

if not md.compBuoyancy:
    pyrolitebuoyancyFn =  (thermalBuoyancyFn)*z_hat
#    harzbuoyancyFn =      (ndp.RA*temperatureField) 
    basaltbuoyancyFn =    (thermalBuoyancyFn)*z_hat
    
    airbuoyancyFn =    (fn.misc.constant(air_comp_buoyancy))*z_hat

else : 
    pyrolitebuoyancyFn =  (thermalBuoyancyFn + pyrolite_comp_buoyancy)*z_hat
#    harzbuoyancyFn =      (ndp.RA*temperatureField*taFn) +\
#                           harz_comp_buoyancy
    basaltbuoyancyFn =    (thermalBuoyancyFn + basalt_comp_buoyancy)*z_hat
    
    airbuoyancyFn =    (fn.misc.constant(air_comp_buoyancy))*z_hat
                           
                           
                           
buoyancyMapFn = fn.branching.map( fn_key = materialVariable,
                         mapping = {airID:airbuoyancyFn ,
                                    crustID:basaltbuoyancyFn, 
                                    mantleID:pyrolitebuoyancyFn} )
#                                    harzIndex:harzbuoyancyFn} )


# In[ ]:




# In[62]:

#md.nonGlobalSwarm


# ## Any other functions we'll need

# In[63]:

###################
#Create integral, max/min templates 
###################


globRestFn = 1.

def volumeint(Fn = 1., rFn=globRestFn):
    return uw.utils.Integral( Fn*rFn,  mesh )

def surfint(Fn = 1., rFn=globRestFn, surfaceIndexSet=mesh.specialSets["MaxJ_VertexSet"]):
    return uw.utils.Integral( Fn*rFn, mesh=mesh, integrationType='Surface', surfaceIndexSet=surfaceIndexSet)

def maxMin(Fn = 1.):
    #maxMin(Fn = 1., rFn=globRestFn
    #vuFn = fn.view.min_max(Fn*rFn) #the restriction functions don't work with the view.min_max fn yet
    vuFn = fn.view.min_max(Fn)
    return vuFn


# In[ ]:




# ## Stokes system and solver

# In[64]:

#print('got to Stokes')


# In[85]:

stokes = uw.systems.Stokes( velocityField  = velocityField, 
                                   pressureField  = pressureField,
                                   conditions     = [freeslipBC,],
                                   fn_viscosity   = viscosityMapFn, 
                                   fn_bodyforce   = buoyancyMapFn )



# In[ ]:

# Create solver & solve
solver = uw.systems.Solver(stokes)


# In[ ]:

solver.options.main.Q22_pc_type='gkgdiag'
solver.options.scr.ksp_rtol=5e-5
solver.set_inner_method('mg')
solver.options.mg.levels = 3
#solver.set_penalty(10.)


# In[ ]:

#solver.set_inner_method("mumps")
#solver.options.scr.ksp_type="cg"
#solver.set_penalty(1.0e7)
#solver.options.scr.ksp_rtol = 1.0e-4


# In[ ]:


solver.solve(nonLinearIterate=True, nonLinearTolerance=5.0e-2,
              nonLinearMaxIterations=15)

#solver.print_stats()


# In[ ]:

fig= glucifer.Figure(quality=3)

#fig.append( glucifer.objects.Mesh(mesh, opacity=0.2))
fig.append( glucifer.objects.Points(swarm, proxyTempVariable, pointSize=2))
fig.append( glucifer.objects.VectorArrows(mesh, velocityField, scaling=0.00005))

#fig.show()
#fig.save_image('solve1.png')


# ## Setup advection-diffusion, swarm advection

# In[ ]:

advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )

if md.thermal:
    advDiff = uw.systems.AdvectionDiffusion( phiField       = temperatureField, 
                                         phiDotField    = temperatureDotField, 
                                         velocityField  = velocityField,
                                         fn_sourceTerm    = 0.0,
                                         fn_diffusivity = 1., 
                                         #conditions     = [neumannTempBC, dirichTempBC] )
                                         conditions     = [ dirichTempBC] )


# In[ ]:

#md.thermal


# In[ ]:

#print('past first solve')


# ## Viz

# #Build a depth dependent mask for the vizualisation
# 
# depthVariable      = swarm.add_variable( dataType="float", count=1 )
# depthVariable.data[:] = depthFn.evaluate(swarm)
# 
# vizVariable      = swarm.add_variable( dataType="int", count=1 )
# vizVariable.data[:] = 0
# 
# for index, value in enumerate(depthVariable.data[:]):
#     #print index, value
#     if np.random.rand(1)**5 > value/(mesh.maxCoord[1] - mesh.minCoord[1]):
#         vizVariable.data[index] = 1
#         
# del index, value    #get rid of any variables that might be pointing at the .data handles (these are!)

# In[52]:

#Set up the gLucifer stores

fullpath = os.path.join(outputPath + "gldbs/")
store1 = glucifer.Store(fullpath + 'subduction1.gldb')


fig1 = glucifer.Figure(store1, boundingBox=bBox)
if md.thermal:
    fig1.append( glucifer.objects.Points(swarm, temperatureField, pointSize=2, fn_mask=vizVariable))
else:
    fig1.append( glucifer.objects.Points(swarm, proxyTempVariable, pointSize=2, fn_mask=vizVariable))
fig1.append( glucifer.objects.Points(swarm, materialVariable, pointSize=2, fn_mask=vizVariable))
fig1.append( glucifer.objects.VectorArrows(mesh, velocityField,  scaling=0.0005))


# In[ ]:

#fig1.show()


# ## Integrals and metrics

# In[ ]:

_pressure = surfint(pressureField)
_surfLength = surfint()
surfLength = _surfLength.evaluate()[0]
pressureSurf = _pressure.evaluate()[0]   
pressureField.data[:] -= pressureSurf/surfLength


# ## Update functions for main loop

# In[ ]:

def main_update():
    
    
    if md.thermal:
        dt = advDiff.get_max_dt()*md.courantFac #md.courantFac helps stabilise advDiff
        advDiff.integrate(dt)
        
    else:
        dt = advector.get_max_dt()
        
    advector.integrate(dt)
    fault.advection(dt)
    metricSwarm.advection(dt)
    
    #remove drift in pressure
    pressureSurf = _pressure.evaluate()[0]   
    pressureField.data[:] -= pressureSurf/surfLength
    
    
    return time+dt, step+1
    


# In[ ]:

def viz_update():
    
    #vizVariable      = swarm.add_variable( dataType="int", count=1 )


    #vizConds = fn.branching.conditional([(proxyTempVariable < 0.9*1., 1),
    #                          (True, 0)]) 

    #vizVariable.data[:] = vizConds.evaluate(swarm)
    
    #save gldbs
    fullpath = os.path.join(outputPath + "gldbs/")
    
    store1.step = step
    fig1.save( fullpath + "Temp" + str(step).zfill(4))
    


# In[ ]:

def swarm_update():
    
    #run swarm repopulation
    repopulate()
    
    
    #rebuild the ridge mask guy
    #nearSurfTempGrad = fn.branching.conditional( ((depthFn < depthMorTest, temperatureField.fn_gradient[1] ), 
    #                                      (True,                      0.)  ))

    #morRestrictFn = fn.math.abs(nearSurfTempGrad) < 50.
    
    
    #rebuild the material graph condition list, and apply to swarm
    MG.build_condition_list(materialVariable)
    materialVariable.data[:] = fn.branching.conditional(MG.condition_list).evaluate(swarm)
    
    
    


# ## Main loop

# In[ ]:

time = 0.  # Initial time
step = 0   # Initial timestep
maxSteps = 100      # Maximum timesteps (201 is recommended)
steps_output = 5   # output every 10 timesteps
metrics_output = 5
files_output = 10


# In[ ]:

while step < maxSteps:
    
    solver.solve(nonLinearIterate=True, nonLinearTolerance=5.0e-2,
              nonLinearMaxIterations=15)
    
    # main
    time,step = main_update()
        
    #Viz
    if step % 5 == 0:
        viz_update()
        
    #particles
    if step % 5 == 0:
        swarm_update()
        
    print 'step =',step    

print 'step =',step


# In[ ]:



