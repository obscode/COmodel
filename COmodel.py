#!/usr/bin/env python
# coding: utf-8

import pickle, os, re
import numpy as np
import astropy.units as u
from astropy.constants import h, c, k_B
from astropy.modeling import models, Fittable1DModel, Parameter
from astropy.io import ascii
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
import george
from george import kernels
from matplotlib import pyplot as plt
import corner

# Pre-compute constants and Units because all the .to() calls are freakin'
# expensive! First:  hc/k_T
hck = (h*c/k_B).to('AA K').value
# plank function constants and units in CGS
hc2 = (2*h*c**2*u.AA**(-5)).to('erg / (s cm2 AA)').value
# conversion factor from cm^2/g/Mpc**2 to 1/Msun
conv = (1*u.cm**2/u.g/u.Mpc**2).to('1 / Msun').value

class FluxModel(Fittable1DModel):

   # The model paramters
   T = Parameter(default=3000)              # Temperature in 10^3 K
   vel = Parameter(default=2000)            # velocity in 10^3 km/s
   lco = Parameter(default=-2.99)           # log(CO+/CO)
   A = Parameter(default=4000)              # Amplitude later converted to mass
   z = Parameter(default=0.00113, bounds=(0, 0.005)) # Extra redshift
   b = Parameter(default=2, bounds=(0,10))  # R-J continuum in 10^-15 

   def __init__(self, pkl, verbose=False, dist=3.4, scale=1, *args, **kwargs):
      '''Initialize a CO model instance.

      Arguments:
        pkl (str):  Name of the pickle file containing Peter's grid of models
        verbose (bool):  Be verbose
        dist (float):  Distance to object in Mpc
        scale (float):  model will be in units of scale*FLAM/Msun. Only matters
                        for computing physical units (i.e., computeMCO())

      Returns:
        model instance
      '''

      self.dist = dist             # Distance in Mpc
      self.scale = scale           # Arbitrary scale for the spectra to 
                                   # avoid small numbers

      # Build the opacities interpolator
      self.readModels(pkl, verbose=verbose)
      self.makeInterpolator()

      super().__init__(*args, **kwargs)
       
      # Set limits on paramters according to Peter's grid limits
      self.T.min = self.interp.grid[1].min()
      self.T.max = self.interp.grid[1].max()
      self.vel.min = self.interp.grid[2].min()
      self.vel.max = self.interp.grid[2].max()
      self.lco.min = self.interp.grid[3].min()
      self.lco.max = self.interp.grid[3].max()
       
   def BB(self, wave, T):
      ''' Return the Blakcbody (F_\lambda) for temperature T'''
      arg = hck/(wave*T)
      BB = hc2*np.power(wave,-5)/(np.exp(arg) - 1)
      return BB

   def opacity(self, wave, T, vel, lco):
      '''Compute opacity from the interpolation of Peter's grid of models
      Arguments:
         wave (float array):  Wavelengths in AA
         T (float):  Temperature in K
         vel (float):  velocity in km/s
         lco (float):  log(CO+/CO)
         z (float):  Extra redshif:  wave is blueshifted by this much
      Returns:
         Kappa (float array):  opacity (cm**2/g) evaluated at wave
      '''
      
      wave = np.atleast_1d(wave)
      # Setup the interpolator evaluation grid
      evp = np.zeros((wave.shape[0],4))
      evp[:,0] = wave
      evp[:,1] = T
      evp[:,2] = vel
      #evp[:,3] = logn
      evp[:,3] = lco
      # Peter's Model
      kappa = self.interp(evp) # Peter's models in cm^2/g
      return kappa

   def evaluate(self, wave, T, vel, lco, A, z, b):
      '''Used internally by scipy.modeling to evaluate the model given 
      the paramters'''
      
      # De-Redshift the wavelenghts (which redshifts the model)
      zwave = np.atleast_1d(wave)/(1+z)
      kappa = self.opacity(zwave, T, vel, lco)
       
      # Only half the photons are viewed, so 2pi instead of 4pi
      hmod = 2*np.pi*self.BB(zwave, T)*kappa/(4*np.pi*(self.dist)**2)
      hmod = hmod*conv    # Now in FLAM/Msun

      self.hscale = sum(hmod)        # This removes the T^4 dependence
      hmod = hmod/self.hscale        # We'll need this scale later to convert
                                     # back to real physical units
      
      # RJ continuum
      RJ = b*np.power(zwave/zwave[0],-2)

      model = A*hmod + RJ
      return model
   
   def computeMCO(self):
      '''Given the current state of the model, compute the CO mass'''
      T,vel,lco,A,z,b = self.parameters

      # Normally, A would be in units of Msun, but we re-scaled the model
      # in the evaluate() function by a factor of self.hscale, so we need to 
      # undo that. And also use flux scale.
      
      return A*self.scale/self.hscale
      #return (A * self.dist**2 / const).value / self.scale / self.hscale
   
   @property
   def variables(self):
      '''Gives a list of parameters that are variables.'''
      return [par for par in self.param_names if not self.fixed[par]]

   def set_pars(self, p):
      '''Given a set of parameters, p, change those variables that are not
         fixed with updated values. To be used with emcee'''
      if len(self.variables) != len(p):
          raise ValueError("Error! Length of p does not match number "                             "of free paramters")
      for i,par in enumerate(self.variables):
          getattr(self, par).value = p[i]

   
   def readModels(self, pkl, verbose=False):
      '''Given a directory or a pickle file, load the models into a grid'''
      with open(pkl, 'rb') as fin:
          self.wave,self.grid,self.temps,self.vels,self.lcos = \
                pickle.load(fin)
                   
   def makeInterpolator(self):
      '''With the model on a grid, setup an interpolator. For now, use 
      linear interp.  We could use something else (GP?) but that might be 
      too slow.'''
      self.interp = RegularGridInterpolator((self.wave, self.temps, self.vels,
                            self.lcos), self.grid, method='linear')                

def lnprior(p):
   '''Return priors for p. For now, just uniform on parameters to keep them
     in the grid.'''
   pars = [par for par in model.param_names if not model.fixed[par]]
   for i,par in enumerate(pars):
      pmin,pmax = model.bounds[par]
      if pmin is not None and p[i] < pmin: return -np.inf
      if pmax is not None and p[i] > pmax:  return -np.inf
   return 0

def lnprob(p):
   lp = lnprior(p)
   if not np.isfinite(lp):
      return -np.inf,-np.inf,-1.0
   model.set_pars(p)
   diff = (flux - model(wave))

   lp += np.sum(norm.logpdf(diff, loc=0, scale=eflux))
   MCO = model.computeMCO()
    
   # return two copies of lp and MCO so they are available as "blobs" 
   #   for post-processing
   return lp,lp,MCO

def plotTraces(samples, model, outfile="traces.pdf"):

   Nvar = len(model.variables)
   fig,axes = plt.subplots(Nvar,1, figsize=(5,10/6*Nvar))
   for i in range(Nvar):
      axes[i].plot(samples[:,:,i], color='k', alpha=0.1)
      axes[i].set_ylabel(model.variables[i])
      axes[i].xaxis.set_visible(False)
   plt.subplots_adjust(left=0.15)
   fig.savefig(outfile)

def plotCorner(samples, meds, model, outfile="corner.pdf"):

   labs = {'T':"$T$", "vel":"$v$", "lco":r"$\log(CO_+/CO)$",
         'z':"$z$", "b":"$b$", 'A':"$M_{CO}$"}
   
   f = corner.corner(samples, #labels=model.variables,
         labelpad=0.2, truths=meds, label_kwargs={'fontsize':16},
         labels=[labs[var] for var in model.variables])
   plt.subplots_adjust(bottom=0.1, left=0.1)
   f.savefig(outfile)
   
def plotFit(wave, flux, meds, model, outfile='flux_fit.pdf'):

   fig,ax = plt.subplots()
   ax.plot(wave, flux)
   model.set_pars(meds)
   ax.plot(wave, model(wave))
   ax.set_xlabel('Wavelength (Anstroms)')
   ax.set_ylabel('flux (scaled to maximum')
   fig.savefig(outfile)

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser(description="Model the CO band using Peter"\
         " Hoeflich's models")
   parser.add_argument('spec', help='Spectrum as ASCII file with two columns')
   parser.add_argument('-grid', help="File containing the model grid pickle",
         default=None)
   parser.add_argument('-wmin', help="Minimum of wavelength range to fit",
         default=22000, type=float)
   parser.add_argument('-wmax', help="Maximum of wavelength range to fit",
         default=25000, type=float)
   parser.add_argument('-Nwalk', help="Number of walkers",
         default=200, type=int)
   parser.add_argument('-output', help="Where to store he MCMC samples",
         default="samples.pkl")
   parser.add_argument('-scale', help="Scale the spetra by this much",
         default=1, type=float)
   parser.add_argument('-Niter', help="Number of iterations",
         default=1000, type=int)
   parser.add_argument('-burn', help="Number of iterations to discard",
         default=500, type=int)
   parser.add_argument('-filt', help="Filter out 'lost' chains?",
         action="store_true")

   args = parser.parse_args()

   import emcee
   import multiprocessing as mp

   Pool = mp.get_context('fork').Pool

   tab = ascii.read(args.spec)
   wave,flux = tab['wave'].value, tab['flux'].value/args.scale
   gids = np.greater(wave, args.wmin) & np.less(wave, args.wmax)
   wave,flux = wave[gids],flux[gids]
   mflux = np.convolve(flux, np.ones(101)/101, mode='same')
   eflux = np.sqrt(np.convolve((flux-mflux)**2, np.ones(101)/101, mode='same'))

   model = FluxModel(args.grid, scale=args.scale)
   
   gp = george.GP(0.06 * kernels.Matern32Kernel(65.0))
   gp.compute(wave, eflux)

   #Ndim = len(model.variables) + 2
   Ndim = len(model.variables) 
   Nwalkers = args.Nwalk
   p0 = []
   for i in range(Nwalkers):
      p0.append([])
      for par in model.variables:
         p = getattr(model, par)
         if p.min is not None and p.max is not None:
            # Draw random number uniformly from bounds
            p0[-1].append(np.random.uniform(p.min, p.max))
         else:
            # raw random number uniformly from 0.1 -> 10 X default value
            p0[-1].append(np.random.uniform(p.value/10, p.value*10))
      #p0[-1].append(np.random.uniform(-2,2))
      #p0[-1].append(np.random.uniform(-2,2))
   p0 = np.array(p0)


   with Pool() as pool:
      sampler = emcee.EnsembleSampler(Nwalkers, Ndim, lnprob,
                                pool=pool)
      sampler.run_mcmc(p0, args.Niter, progress=True)

   # Dump the chains to a file.
   samples = sampler.get_chain()
   blobs = sampler.get_blobs()
   with open(args.output, 'wb') as fout:
      pickle.dump(dict(samples=samples, wave=wave, flux=flux,
                       eflux=eflux, wmin=args.wmin, wmax=args.wmax,
                       blobs=blobs, grid=args.grid), fout)

   # Plot the raw traces (no burn removed and 'A' instead of M_CO
   plotTraces(samples, model, outfile="traces.pdf")

   # Get into a more covenient shape (chain, iteration, paramter)
   #samples = np.transpose(samples, (1,0,2))
   #blobs = np.transpose(blobs, (1,0,2))

   # Now replace 'A' with M_CO
   idx = model.variables.index('A')
   samples[:,:,idx] = blobs[:,:,1]

   # Take out the burn-in
   samples = samples[args.burn:,:,:]

   # Medians and Standard deviations
   meds = np.median(samples, axis=(1,0))
   stds = 1.5*np.median(np.absolute(samples - meds[np.newaxis,np.newaxis,:]), 
          axis=(1,0))
   
   # If asked for, filter out lost chains
   gids = np.ones(samples.shape[1]).astype(bool)
   if args.filt:
      # Filter out lost chains
      for i in range(samples.shape[2]):
         devs = np.absolute(meds[i]-np.median(samples[:,:,i], axis=0))
         bids = np.nonzero(np.greater(devs, stds[i]*3))
         gids[bids] = False

   # Re-compute with good stuff
   gsamples = samples[:,gids,:]
   meds = np.median(gsamples, axis=(1,0))
   stds = 1.5*np.median(np.absolute(gsamples - meds[np.newaxis,np.newaxis,:]), 
          axis=(1,0))
   
   plotCorner(gsamples, meds, model, outfile="corner.pdf")
   plotFit(wave, flux, meds, model, outfile='flux_fit.pdf')

   with open("best_fit.txt", 'w') as fout:
      for i,par in enumerate(model.variables):
         print("{:20s} {} +/- {}".format(par,meds[i], stds[i]))
         fout.write("{:20s} {} +/- {}\n".format(par,meds[i], stds[i]))
