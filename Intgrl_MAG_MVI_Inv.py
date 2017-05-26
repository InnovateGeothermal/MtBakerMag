"""
This script runs a Magnetic Vector Inversion - Cartesian formulation. The code
is used to get the magnetization orientation and makes no induced assumption.
The inverse problem is three times larger than the usual susceptibility
inversion, but won't suffer the same issues in the presence of remanence. The
drawback of having 3x the number of model parameters is in non-uniqueness of
the solution. Models have a tendency to be overlly complicated.

To counter this issue, the inversion can be done cooperatively with the
Magnetic Amplitude Inverion (MAI) that uses sparse norms to recover compact
bodies. The information about the location and shape of magnetic bodies is
used by the MVI code through a cell-based wieght.

This script will run both inversions in sequence in order to compare the
results if the flag CMI is activated

-------------------------------------------------------------------------------
 !! IMPORTANT: PLease run Intgrl_MAG_AMP_Inv.py before running this script !!
-------------------------------------------------------------------------------


Created on Thu Sep 29 10:11:11 2016

@author: dominiquef
"""
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization
from SimPEG import DataMisfit, Inversion, Regularization
import SimPEG.PF as PF
import numpy as np
import os

# Define the inducing field parameter
work_dir = ".\\"
out_dir = "SimPEG_MVI_Inv\\"
input_file = "MB_50m_input_file.inp"

os.system('mkdir ' + work_dir+out_dir)
CMI = True

# %% INPUTS
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)

#%% Access the mesh and survey information
mesh = driver.mesh
survey = driver.survey

# Extract active region
actv = driver.activeCells

# Set starting mdoel
mstart = np.ones(3*len(actv))*1e-4

# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, 0)
nC = int(len(actv))

# Create identity map
# Create wires to link the regularization to each model blocks
wires = Maps.Wires(('prim', nC),
           ('second', nC),
           ('third', nC))

# Create identity map
idenMap = Maps.IdentityMap(nP=3*nC)

mstart= np.ones(3*len(actv))*1e-4

# Create the forward model operator
prob = PF.Magnetics.MagneticVector(mesh, chiMap=idenMap,
                                     actInd=actv)

# Pair the survey and problem
survey.pair(prob)


# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std


# Create sensitivity weights from our linear forward operator
wr = np.zeros(prob.F.shape[1])
for ii in range(survey.nD):
    wr += prob.F[ii, :]**2.
wr = wr**0.5
wr = (wr/np.max(wr))


# Create a regularization
reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.prim)
reg_p.cell_weights = wires.prim* wr

reg_s = Regularization.Sparse(mesh, indActive=actv, mapping=wires.second)
reg_s.cell_weights = wires.second * wr

reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.third)
reg_t.cell_weights = wires.third * wr

reg = reg_p + reg_s + reg_t


# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=30, lower=-10., upper=10.,
                                 maxIterCG=20, tolCG=1e-3)


invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

betaCool = Directives.BetaSchedule(coolingFactor=2., coolingRate=1)

update_Jacobi = Directives.UpdatePreCond()
targetMisfit = Directives.TargetMisfit()

saveModel = Directives.SaveUBCModelEveryIteration(mapping = actvMap)
saveModel.fileName = work_dir + out_dir + 'MVI'
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, update_Jacobi, betaCool,
                                             targetMisfit, saveModel])
mrec = inv.run(mstart)


obs_loc = survey.srcField.rxList[0].locs
PF.Magnetics.writeUBCobs(work_dir+out_dir+'MVI.pre', survey, invProb.dpred)

#%% Re-run the inversion if the CMI flag is activated.
# The script will try to first load the amplitude model.
# This part will break if the amplitude model is not in the working directory

if CMI:

    # Try to load amplitude model.
    MAI_m = Mesh.TensorMesh.readModelUBC(mesh, work_dir + 'Amplitude_lplq.sus')

    # Create rescaled weigths
    mamp = (MAI_m[actv]/MAI_m[actv].max() + 1e-2)**-1.

    # Update the sensitivity weights with amplitude weights added
    wr = np.sum(prob.F**2., axis=0)**0.5
    wr = (wr/np.max(wr))*np.r_[mamp, mamp, mamp]

    # Create a regularization
    reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.prim)
    reg_p.cell_weights = wires.prim* wr

    reg_s = Regularization.Sparse(mesh, indActive=actv, mapping=wires.second)
    reg_s.cell_weights = wires.second * wr

    reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.third)
    reg_t.cell_weights = wires.third * wr

    reg = reg_p + reg_s + reg_t

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1./survey.std

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=30, lower=-10., upper=10.,
                                     maxIterCG=20, tolCG=1e-3)

    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    betaCool = Directives.BetaSchedule(coolingFactor=2., coolingRate=1)

    update_Jacobi = Directives.UpdatePreCond()
    targetMisfit = Directives.TargetMisfit()

    saveModel = Directives.SaveUBCModelEveryIteration(mapping = actvMap)
    saveModel.fileName = work_dir + out_dir + 'CMI'
    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[betaest, update_Jacobi, betaCool,
                                                 targetMisfit, saveModel])
    mrec = inv.run(mstart)

    PF.Magnetics.writeUBCobs(work_dir+out_dir+'CMI.pre',survey,invProb.dpred)

    obs_loc = survey.srcField.rxList[0].locs

