"""

This script runs an Magnetic Amplitude Inversion (MAI) from TMI data.
Magnetic amplitude data are weakly sensitive to the orientation of
magnetization, and can therefore better recover the location and geometry of
magnetic bodies in the presence of remanence. The algorithm is inspired from
Li & Shearer (2008), with an added iterative sensitivity weighting strategy to
counter the vertical stretching that the original code suffered

This is done in three parts:

1- TMI data are inverted for an equivalent source layer.

2-The equivalent source layer is used to predict component data -> amplitude

3- Amplitude data are inverted in 3-D for an effective susceptibility model

Created on December 7th, 2016

@author: fourndo@gmail.com

"""
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, DataMisfit
from SimPEG import Inversion, Utils, Regularization
import SimPEG.PF as PF
import numpy as np
import matplotlib.pyplot as plt
import os

# targmis = simpeg.Directives.TargetMisfit()

work_dir = ".\\"
out_dir = "SimPEG_AMP_Inv\\"
input_file = "MB_100m_input_file.inp"

os.system('mkdir ' + work_dir+out_dir)

# %%
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)

# Access the mesh and survey information
mesh = driver.mesh
survey = driver.survey

# %% STEP 1: EQUIVALENT SOURCE LAYER
# The first step inverts for an equiavlent source layer in order to convert the
# observed TMI data to magnetic field Amplitude.

# Get the active cells for equivalent source is the top only
# active = driver.activeCells(layer=True)
active = driver.activeCells
surf = PF.MagneticsDriver.actIndFull2layer(mesh, active)

nC = len(surf)  # Number of active cells

# Create active map to go from reduce set to full
surfMap = Maps.InjectActiveCells(mesh, surf, -100)

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create static map
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap,
                                     actInd=surf, equiSourceLayer=True)
prob.solverOpts['accuracyTol'] = 1e-4

# Pair the survey and problem
survey.pair(prob)

# Create a regularization function, in this case l2l2
reg = Regularization.Simple(mesh, indActive=surf)
reg.mref = np.zeros(nC)

# Specify how the optimization will proceed, set susceptibility bounds to inf
opt = Optimization.ProjectedGNCG(maxIter=500, lower=-np.inf,
                                 upper=np.inf, maxIterLS=20,
                                 maxIterCG=20, tolCG=1e-3)

# Define misfit function (obs-calc)
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Create the default L2 inverse problem from the above objects
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Specify how the initial beta is found
betaest = Directives.BetaEstimate_ByEig()

# Beta schedule for inversion
betaSchedule = Directives.BetaSchedule(coolingFactor=2., coolingRate=1)

# Target misfit to stop the inversion,
# try to fit as much as possible of the signal, we don't want to lose anything
targetMisfit = Directives.TargetMisfit(chifact=0.1)

# Put all the parts together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, betaSchedule,
                                             targetMisfit])

# Run the equivalent source inversion
mstart = np.zeros(nC)
print('Number of Data for Inversion is: {:.1f}'.format(survey.nD))
mrec = inv.run(mstart)

pred = invProb.dpred
PF.Magnetics.writeUBCobs(work_dir + out_dir + 'EQS_predicted.pre',
                         survey, pred)

# Ouput result
Mesh.TensorMesh.writeModelUBC(mesh,
                              work_dir + out_dir + "EquivalentSource.sus",
                              surfMap*mrec)

# %% STEP 2: COMPUTE AMPLITUDE DATA
# Now that we have an equialent source layer, we can forward model alh three
# components of the field and add them up: |B| = ( Bx**2 + Bx**2 + Bx**2 )**0.5

# Won't store the sensitivity and output 'xyz' data.
prob.forwardOnly = True
pred_x = prob.Intrgl_Fwr_Op(m=mrec, recType='x')
pred_y = prob.Intrgl_Fwr_Op(m=mrec, recType='y')
pred_z = prob.Intrgl_Fwr_Op(m=mrec, recType='z')

ndata = survey.nD

d_amp = np.sqrt(pred_x**2. +
                pred_y**2. +
                pred_z**2.)

rxLoc = survey.srcField.rxList[0].locs

# Write data out
PF.Magnetics.writeUBCobs(work_dir + out_dir + 'Amplitude_data.obs',
                         survey, d_amp)

# %% STEP 3: RUN AMPLITUDE INVERSION
# Now that we have |B| data, we can invert. This is a non-linear inversion,
# which requires some special care for the sensitivity weights (see Directives)

# Re-set the active cells to entire mesh
# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, active, -100)
nC = len(active)

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)

mstart = np.ones(len(active))*1e-4

# Create the forward model operator
prob = PF.Magnetics.MagneticAmplitude(mesh, chiMap=idenMap,
                                      actInd=active)
prob.model = mstart

# Change the survey to xyz components
survey.srcField.rxList[0].rxType = 'xyz'

# Pair the survey and problem
survey.unpair()
survey.pair(prob)

# Re-set the observations to |B|
survey.dobs = d_amp

# Create a sparse regularization
# Create a sparse regularization
reg = Regularization.Sparse(mesh, indActive=active, mapping=idenMap)
reg.mref = np.zeros(nC)
reg.norms = driver.lpnorms
if driver.eps is not None:
    reg.eps_p = driver.eps[0]
    reg.eps_q = driver.eps[1]

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1/survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=40, lower=0., upper=1.,
                                 maxIterLS=10, maxIterCG=30,
                                 tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Here is the list of directives
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e1)

# Specify the sparse norms
IRLS = Directives.Update_IRLS(f_min_change=1e-3,
                              minGNiter=3, coolingRate=3, chifact=1.,
                              maxIRLSiter=10)

# Special directive specific to the mag amplitude problem. The sensitivity
# weights are update between each iteration.
update_SensWeight = Directives.UpdateSensWeighting()
update_Jacobi = Directives.UpdatePreCond()

saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
saveModel.fileName = work_dir + out_dir + 'AmpInvTemp'

# Put all together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_SensWeight,
                                             update_Jacobi, saveModel])

# Invert
mrec = inv.run(mstart)

# Output results
if getattr(invProb, 'l2model', None) is not None:
    Mesh.TensorMesh.writeModelUBC(mesh,
                                  work_dir + out_dir + "Amplitude_l2l2.sus",
                                  actvMap*invProb.l2model)

Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + "Amplitude_lplq.sus",
                              actvMap*invProb.model)
PF.Magnetics.writeUBCobs(work_dir+out_dir+'Amplitude_Inv.pre',
                         survey, invProb.dpred)
