# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:40:47 2016

@author: Jeff Witter
"""

import os
import SimPEG.PF as PF
from SimPEG import Maps, Regularization, Optimization, DataMisfit,\
                   InvProblem, Directives, Inversion
from SimPEG.Utils.io_utils import remoteDownload
import matplotlib.pyplot as plt
import numpy as np


def run(plotIt=True):
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
    
        # Load a MESH file?????

        # Load a TOPO file?????

        # Load a TMI OBS file?????

        # Load an INPUT file?????

# %%
    # Read in the input file which included all parameters at once
    # (mesh, topo, model, survey, inv param, etc.)
    driver = PF.MagneticsDriver.MagneticsDriver_Inv(MB_100m_input_file)
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
    prob = PF.Gravity.GravityIntegral(mesh, mapping=staticCells,
                                      actInd=active)
    prob.solverOpts['accuracyTol'] = 1e-4

    # Pair the survey and problem
    survey.pair(prob)

    # Apply depth weighting
    wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, active, wgtexp,
                                   np.min(mesh.hx)/4.)
    wr = wr**2.

    # %% Create inversion objects
    reg = Regularization.Sparse(mesh, indActive=active,
                                mapping=staticCells)
    reg.mref = driver.mref[dynamic]
    reg.cell_weights = wr * mesh.vol[active]

    # Specify how the optimization will proceed
    opt = Optimization.ProjectedGNCG(maxIter=150, lower=driver.bounds[0],
                                     upper=driver.bounds[1], maxIterLS=20,
                                     maxIterCG=20, tolCG=1e-3)

    # Define misfit function (obs-calc)
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.Wd = 1./wd

    # create the default L2 inverse problem from the above objects
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

    # Specify how the initial beta is found
    betaest = Directives.BetaEstimate_ByEig()
    
    
    # IRLS sets up the Lp inversion problem
    # Set the eps parameter parameter in Line 11 of the
    # input file based on the distribution of model (DEFAULT = 95th %ile)
    IRLS = Directives.Update_IRLS(norms=driver.lpnorms, eps=driver.eps,
                                  f_min_change=1e-2, maxIRLSiter=20,
                                  minGNiter=5)

    # Preconditioning refreshing for each IRLS iteration
    update_Jacobi = Directives.Update_lin_PreCond()

    # Create combined the L2 and Lp problem
    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[IRLS, update_Jacobi, betaest])

    # %%
    # Run L2 and Lp inversion
    mrec = inv.run(mstart)
    
    # %%
    if plotIt:
        # Plot observed data
        PF.Magnetics.plot_obs_2D(rxLoc, d, 'Observed Data')

        # %%
        # Write output model and data files and print misft stats.

        # reconstructing l2 model mesh with air cells and active dynamic cells
        L2out = activeMap * reg.l2model

        # reconstructing lp model mesh with air cells and active dynamic cells
        Lpout = activeMap*mrec

        # %%
        # Plot out sections and histograms of the smooth l2 model.
        # The ind= parameter is the slice of the model from top down.
        yslice = midx + 1
        L2out[L2out == -100] = np.nan  # set "air" to nan

        plt.figure(figsize=(10, 7))
        plt.suptitle('Smooth Inversion: Depth weight = ' + str(wgtexp))
        ax = plt.subplot(221)
        dat1 = mesh.plotSlice(L2out, ax=ax, normal='Z', ind=-16,
                              clim=(vmin, vmax), pcolorOpts={'cmap': 'bwr'})
        plt.plot(np.array([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 np.array([mesh.vectorCCy[yslice], mesh.vectorCCy[yslice]]),
                 c='gray', linestyle='--')
        plt.scatter(rxLoc[0:, 0], rxLoc[0:, 1], color='k', s=1)
        plt.title('Z: ' + str(mesh.vectorCCz[-16]) + ' m')
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.gca().set_aspect('equal', adjustable='box')
        cb = plt.colorbar(dat1[0], orientation="vertical",
                          ticks=np.linspace(vmin, vmax, 4))
        cb.set_label('MagSusc (SI)')

        ax = plt.subplot(222)
        dat = mesh.plotSlice(L2out, ax=ax, normal='Z', ind=-27,
                             clim=(vmin, vmax), pcolorOpts={'cmap': 'bwr'})
        plt.plot(np.array([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 np.array([mesh.vectorCCy[yslice], mesh.vectorCCy[yslice]]),
                 c='gray', linestyle='--')
        plt.scatter(rxLoc[0:, 0], rxLoc[0:, 1], color='k', s=1)
        plt.title('Z: ' + str(mesh.vectorCCz[-27]) + ' m')
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.gca().set_aspect('equal', adjustable='box')
        cb = plt.colorbar(dat1[0], orientation="vertical",
                          ticks=np.linspace(vmin, vmax, 4))
        cb.set_label('MagSusc (SI)')

        ax = plt.subplot(212)
        mesh.plotSlice(L2out, ax=ax, normal='Y', ind=yslice,
                       clim=(vmin, vmax), pcolorOpts={'cmap': 'bwr'})
        plt.title('Cross Section')
        plt.xlabel('Easting(m)')
        plt.ylabel('Elevation')
        plt.gca().set_aspect('equal', adjustable='box')
        cb = plt.colorbar(dat1[0], orientation="vertical",
                          ticks=np.linspace(vmin, vmax, 4), cmap='bwr')
        cb.set_label('MagSusc (SI)')

        # %%
        # Make plots of Lp model
        yslice = midx + 1
        Lpout[Lpout == -100] = np.nan  # set "air" to nan

        plt.figure(figsize=(10, 7))
        plt.suptitle('Compact Inversion: Depth weight = ' + str(wgtexp) +
                     ': $\epsilon_p$ = ' + str(round(reg.eps_p, 1)) +
                     ': $\epsilon_q$ = ' + str(round(reg.eps_q, 2)))
        ax = plt.subplot(221)
        dat = mesh.plotSlice(Lpout, ax=ax, normal='Z', ind=-16,
                             clim=(vmin, vmax), pcolorOpts={'cmap': 'bwr'})
        plt.plot(np.array([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 np.array([mesh.vectorCCy[yslice], mesh.vectorCCy[yslice]]),
                 c='gray', linestyle='--')
        plt.scatter(rxLoc[0:, 0], rxLoc[0:, 1], color='k', s=1)
        plt.title('Z: ' + str(mesh.vectorCCz[-16]) + ' m')
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.gca().set_aspect('equal', adjustable='box')
        cb = plt.colorbar(dat[0], orientation="vertical",
                          ticks=np.linspace(vmin, vmax, 4))
        cb.set_label('MagSusc (SI)')

        ax = plt.subplot(222)
        dat = mesh.plotSlice(Lpout, ax=ax, normal='Z', ind=-27,
                             clim=(vmin, vmax), pcolorOpts={'cmap': 'bwr'})
        plt.plot(np.array([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 np.array([mesh.vectorCCy[yslice], mesh.vectorCCy[yslice]]),
                 c='gray', linestyle='--')
        plt.scatter(rxLoc[0:, 0], rxLoc[0:, 1], color='k', s=1)
        plt.title('Z: ' + str(mesh.vectorCCz[-27]) + ' m')
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.gca().set_aspect('equal', adjustable='box')
        cb = plt.colorbar(dat[0], orientation="vertical",
                          ticks=np.linspace(vmin, vmax, 4))
        cb.set_label('MagSusc (SI)')

        ax = plt.subplot(212)
        dat = mesh.plotSlice(Lpout, ax=ax, normal='Y', ind=yslice,
                             clim=(vmin, vmax), pcolorOpts={'cmap': 'bwr'})
        plt.title('Cross Section')
        plt.xlabel('Easting (m)')
        plt.ylabel('Elevation (m)')
        plt.gca().set_aspect('equal', adjustable='box')
        cb = plt.colorbar(dat[0], orientation="vertical",
                          ticks=np.linspace(vmin, vmax, 4))
        cb.set_label('MagSusc (SI)')

if __name__ == '__main__':
    run()
    plt.show()