# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:40:47 2016

@author: Jeff Witter
"""

import os
import SimPEG.PF as PF
from SimPEG import Maps, Regularization, Optimization, DataMisfit,\
                   InvProblem, Directives, Inversion, Mesh
import matplotlib.pyplot as plt
import numpy as np


# def run(plotIt=True):
"""
PF: Mag: Mount Baker TMI data
Coarse, 100m cell size inversion
=============================================

This script uses the SimPEG code to invert TMI
data collected at Mt. Baker volcano, Washington by William
Schermerhorn in August 2016.
The model AOI is 2.5 x 3.5 km in the vicinity of Baker Hot Springs.
The 3D model is also 2.5 km thick.

We run the inversion in two steps.  Firstly creating a L2 model and
then applying an Lp norm to produce a compact model.
"""

# %% User input
# Plotting parameters, max and min Mag Susc in SI units
vmin = 0.0
vmax = 0.1

# weight exponent for default weighting
wgtexp = 3.

# Define the inducing field parameter (Total Field, Incl, Decl)
H0 = (54318, 70.139, 16.063)

# Start by importing files HOW DO I DO THIS??
# DOM: Just point to your working directory where your input files are.
# Since you move them to the current directory, it is just .\
work_dir = ".\\"
out_dir = "SimPEG_PF_Inv\\"
input_file = "MB_100m_input_file.inp"
# %%
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
os.system('mkdir ' + work_dir+out_dir)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)
# %%
# Now we need to create the survey and model information.

# Access the mesh and survey information
mesh = driver.mesh
survey = driver.survey

# define magnetic survey locations - one point for each gridded TMI map cell
rxLoc = survey.srcField.rxList[0].locs

# define magnetic data and errors
d = survey.dobs
wd = survey.std

# Get the active cells
active = driver.activeCells
nC = len(active)  # Number of active cells

# Create active map to go from reduce set to full
activeMap = Maps.InjectActiveCells(mesh, active, -100)

# Create static map
static = driver.staticCells
dynamic = driver.dynamicCells

staticCells = Maps.InjectActiveCells(None,
                                     dynamic, driver.m0[static], nC=nC)
mstart = driver.m0[dynamic]

# Get index of the center
midx = int(mesh.nCx/2)

# %%
# Now that we have a model and a survey we can build the linear system ...
# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=staticCells,
                                  actInd=active)
prob.solverOpts['accuracyTol'] = 1e-4

# Pair the survey and problem
survey.pair(prob)

# Apply depth weighting
# wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, active, wgtexp,
#                                np.min(mesh.hx)/4.)
# wr = wr**2.
# DOM: I recommand just using the sensitivity for the MAG problem
wr = np.sum(prob.F**2., axis=0)**0.5
wr = (wr/np.max(wr))

# %% Create inversion objects
reg = Regularization.Sparse(mesh, indActive=active,
                            mapping=staticCells)
reg.norms = driver.lpnorms

if driver.eps is not None:
    reg.eps_p = driver.eps[0]
    reg.eps_q = driver.eps[1]

reg.mref = driver.mref[dynamic]
reg.cell_weights = wr

# Specify how the optimization will proceed
opt = Optimization.ProjectedGNCG(maxIter=500, lower=driver.bounds[0],
                                 upper=driver.bounds[1], maxIterLS=50,
                                 maxIterCG=20, tolCG=1e-3)

# Define misfit function (obs-calc)
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./wd

# create the default L2 inverse problem from the above objects
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Specify how the initial beta is found
betaest = Directives.BetaEstimate_ByEig()


# IRLS sets up the Lp inversion problem
# Set the eps parameter parameter in Line 11 of the
# input file based on the distribution of model (DEFAULT = 95th %ile)
IRLS = Directives.Update_IRLS(f_min_change=1e-2, maxIRLSiter=20,
                              minGNiter=5)

# Preconditioning refreshing for each IRLS iteration
update_Jacobi = Directives.UpdatePreCond()

saveModel = Directives.SaveUBCModelEveryIteration(mapping=activeMap)
saveModel.fileName = work_dir + out_dir + 'MagSus'

# Create combined the L2 and Lp problem
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_Jacobi, saveModel])

# %%
# Run L2 and Lp inversion
mrec = inv.run(mstart)

# %%
# Plot observed data
PF.Magnetics.plot_obs_2D(rxLoc, d, 'Observed Data')

# %%
# Write output model and data files and print misft stats.

if getattr(invProb, 'l2model', None) is not None:
    # reconstructing l2 model mesh with air cells and active dynamic cells
    L2out = activeMap * invProb.l2model
    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'L2_model.sus', L2out)
    pred = invProb.dpred
    PF.Magnetics.writeUBCobs(work_dir + out_dir + 'UBC_Model.pre', survey, pred)

# reconstructing lp model mesh with air cells and active dynamic cells
Lpout = activeMap*mrec
Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'Lp_model.sus', Lpout)