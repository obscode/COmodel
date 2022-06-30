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


# Constants in BB multiplied by AA^-5 (for the lambda^-5 factor)
const = 2*h*c**2*u.AA**-5
# multiply by kappa units
const = const*u.cm**2/u.g
# Now convert to the units we want
const = const.to('erg Mpc2 / (solMass s AA cm2 )')

class FluxModel(Fittable1DModel):
   FILENAME_FORMAT = re.compile("^plot\.dat_(\d+)_(\d+)_(\d+\.\d*)_(.+)$")

   # The model paramters
   T = Parameter(default=3000)              # Temperature in 10^3 K
   vel = Parameter(default=2000)            # velocity in 10^3 km/s
   logn = Parameter(default=8, fixed=True)  # Log density
   lco = Parameter(default=-2.99)           # log(CO+/CO)
   A = Parameter(default=4000)              # Amplitude later converted to mass
   z = Parameter(default=0.00113, bounds=(0, 0.005)) # Extra redshift
   b = Parameter(default=2, bounds=(0,10))  # R-J continuum in 10^-15 
                                            #  erg/s/cm2/AA

   def __init__(self, pkl, verbose=False, dist=3.4, scale=1, *args, **kwargs):

      self.dist = dist*u.Mpc       # Distance in Mpc
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
      self.logn.min = self.interp.grid[3].min()
      self.logn.max = self.interp.grid[3].max()
      self.lco.min = self.interp.grid[4].min()
      self.lco.max = self.interp.grid[4].max()
       
   def evaluate(self, wave, T, vel, logn, lco, A, z, b): #, sigma):
      '''Used internally by scipy.modeling to evaluate the model given 
      the paramters'''
      
      # De-Redshift the wavelenghts (which redshifts the model)
      zwave = np.atleast_1d(wave)/(1+z)
      
      # Setup the interpolator evaluation grid
      evp = np.zeros((zwave.shape[0],5))
      evp[:,0] = zwave
      evp[:,1] = T
      evp[:,2] = vel
      evp[:,3] = logn
      evp[:,4] = lco
      # Peter's Model
      hmod = self.interp(evp)
      #self.hscale = hmod.max()         # Removes T^4 dependence
      self.hscale = sum(hmod)
      #self.hscale = 1e-22
      hmod = hmod/self.hscale        # We'll need this scale later
      
      # RJ continuum
      RJ = b*np.power(zwave/zwave[0],-2)
      model = A*hmod + RJ
      return model
   
   def computeMCO(self):
      '''Given the current state of the model, compute the CO mass'''
      T,vel,logn,lco,A,z,b = self.parameters
      
      return (A * self.dist**2 / const).value / self.scale / self.hscale
   
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
          self.wave,self.grid,self.temps,self.vels,self.lns,self.lcos = \
                pickle.load(fin)
                   
   def makeInterpolator(self):
      '''With the model on a grid, setup an interpolator. For now, use 
      linear interp.  We could use something else (GP?) but that might be 
      too slow.'''
      self.interp = RegularGridInterpolator((self.wave, self.temps, self.vels,
                            self.lns, self.lcos), self.grid, method='linear')                

def lnprior(p):
   '''Return priors for p. For now, just uniform on paramters to keep them
     in the grid. The last two are for the GP noise model'''
   pars = [par for par in model.param_names if not model.fixed[par]]
   for i,par in enumerate(pars):
      pmin,pmax = model.bounds[par]
      if pmin is not None and p[i] < pmin: return -np.inf
      if pmax is not None and p[i] > pmax:  return -np.inf
   #lna,lntau = p[-2:]
   #lna,lntau = p[-2:]
   #if not ( -5 < lna < 5):  return -np.inf
   #if not (-5 < lntau < 5):  return -np.inf
      
   return 0

def lnprob(p):
   #global model, wave, flux, eflux
   lp = lnprior(p)
   if not np.isfinite(lp):
      return -np.inf,-np.inf,-1.0

   #model.set_pars(p[:-2])
   model.set_pars(p)
   diff = (flux - model(wave))

   #a, tau = np.exp(p[-2:])
   #gp = george.GP(a * kernels.Matern32Kernel(tau))
   #gp.compute(wave, eflux)
   lp += np.sum(norm.logpdf(diff, loc=0, scale=eflux))
   #lp += gp.lnlikelihood(diff)
   MCO = model.computeMCO()
    
   # return two copies of lp and MCO so they are available as "blobs" 
   #   for post-processing
   return lp,lp,MCO

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

   args = parser.parse_args()

   import emcee
   import multiprocessing as mp

   Pool = mp.get_context('fork').Pool

   tab = ascii.read(args.spec)
   wave,flux = tab['wave'].value, tab['flux'].value*args.scale
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

   samples = sampler.get_chain()
   blobs = sampler.get_blobs()
   with open(args.output, 'wb') as fout:
      pickle.dump(dict(samples=samples, wave=wave, flux=flux,
                       eflux=eflux, wmin=args.wmin, wmax=args.wmax,
                       blobs=blobs), fout)

