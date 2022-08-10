import numpy as np
from numpy.random import binomial
import numba
from numba import jit, prange
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import basinhopping

@jit(nopython=True, cache=True)
def _log_sum_exp(arr):
  a = np.max(arr)
  x = arr - a
  tot = a + np.log(np.sum(np.exp(x)))
  return(tot)

@jit(nopython=True, cache=True)
def _emission_helper(a, hij, eps):
  # Calculating emission probability here
  emiss_prob = (1. - eps) * (a == hij) + eps * (a != hij)
  return(np.log(emiss_prob))

@jit(nopython=True, cache=True)
def _forward_algo(haps, positions, test_hap, eps, nsamples, nsnps, scale):
  """ Helper function to compute the forward algorithm """
#   assert((eps >= 0.0) & (eps < 1.0))
#   assert(haps.shape[1] == positions.size)
  assert(test_hap.size == positions.size)
  assert(scale > 0)
  alphas_null = [_emission_helper(test_hap[0], haps[j,0], eps) for j in range(nsamples)]
  alphas = np.zeros(shape=(nsamples,nsnps), dtype=np.float64)
  alphas[:,0] = np.array(alphas_null)
  start_pos = positions[0]
  # Iterating through the snps
  for i in range(1, nsnps):
    cur_pos = positions[i]
    # Calculate distance between samples
    dij = cur_pos - start_pos
    # Probability of recombination before coalescence
    rate = scale*dij
    # Probability of moving away (maybe use log1p here?)
    pj = 1.0 - np.exp(-rate)
    # This second term is the log-transitions
    second_term = alphas[:,i-1] - np.log(nsamples) + np.log(pj)
    a = _log_sum_exp(second_term)
#     print(a, pj, dij, rate)
    # Calculating the underlying likelihood
    for j in range(nsamples):
      # Calculate probability of no transition away from the state
      no_trans = np.log(1.0 - pj) + alphas[j,i-1]
      # Calculate updated alpha parameter (with emission from current site)
      alphas[j,i] = _emission_helper(test_hap[i], haps[j,i], eps) + _log_sum_exp(np.array([a, no_trans]))
    start_pos = cur_pos
  return(alphas)

@jit(nopython=True, cache=True)
def _backward_algo(haps,positions, test_hap, eps, nsamples, nsnps, scale):
  """ Helper function for computing the backward algorithm """
  assert((eps >= 0.0) & (eps < 1.0))
  betas_null = [_emission_helper(test_hap[0], haps[j,-1], eps) for j in range(nsamples)]
  betas = np.zeros(shape=(nsamples,nsnps))
  betas[:,-1] = np.array(betas_null)
  start_pos = positions[-1]
  for i in range(nsnps-2,-1,-1):
    cur_pos = positions[i]
    # Calculating position in reverse...
    dij = start_pos - cur_pos
    # Multiply centimorgan-scale to true scale?
    rate = scale*dij
    pj = 1.0 - np.exp(-rate)
    second_term = betas[:,i+1] - np.log(nsamples) + np.log(pj)
    b = _log_sum_exp(second_term)
    for j in range(nsamples):
      notrans = np.log(1.0 - pj) + betas[j,i+1]
      betas[j,i] = _emission_helper(test_hap[i], haps[j,i], eps) + _log_sum_exp(np.array([b,notrans]))
    start_pos = cur_pos
  return(betas)

@jit(nopython=True, cache=True)
def _fwd_bwd_algo(haps, positions,test_hap, eps, nsamples, nsnps, scale):
  """ Function to compute the forward-backward algorithm for Li-Stephens """
  # Forward algorithm step
  alphas = _forward_algo(haps, positions,test_hap, eps, nsamples, nsnps, scale)
  # Backward algorithm step
  betas = _backward_algo(haps, positions,test_hap, eps, nsamples, nsnps, scale)
  # gammas are still in log-space
  gammas = alphas + betas
  true_gammas = np.zeros(gammas.shape)
  for i in range(gammas.shape[1]):
    true_gammas[:,i] = gammas[:,i]  - _log_sum_exp(gammas[:,i])
    true_gammas[:,i] = np.exp(true_gammas[:,i])
  return(true_gammas)

def _viterbi_helper (n_states, n_snps, positions, eps, scale, test_hap, panel_haps):
  """helper function to compute the viterbi algorithm"""
  # initialize
  delta = np.zeros(shape=(n_states, n_snps))
  delta[:,0] = [_emission_helper(test_hap[0], panel_haps[j,0], eps)-np.log(n_states) for j in range(n_states)]
  arrows = np.ndarray(shape=(n_states, n_snps), dtype=object)
  arrows[:,0] = 0

  # main loop
  previous_pos = positions[0]

  for i in range(1, n_snps):

    # calculate genetic dist between snps
    gen_dist = positions[i] - previous_pos
    rate = gen_dist*scale

    # calculate log transition probs
    log_trans = np.log((1-np.exp(-rate))/n_states)
    log_no_trans = np.log(np.exp(-rate)+(1-np.exp(-rate))/n_states)

    # create log transition matrix
    trans_mat = np.full((n_states,n_states), log_trans)
    np.fill_diagonal(trans_mat, log_no_trans)

    for j in range(0, n_states):

      # store prob
      emission_prob = _emission_helper(test_hap[i], panel_haps[j][i], eps)
      delta[j][i] = np.max(delta[:,i-1] + trans_mat[:,j] + emission_prob)

      # store arrow
      arrows[j][i] = np.argmax(delta[:,i-1] + trans_mat[:,j] + emission_prob)

      # update genetic map position
      previous_pos = positions[i]

  # bactracking
  path = []

  max_state = np.argmax(delta[:, -1]) # Find the index of the max value in the last column of delta
  max_value = delta[max_state, -1] # Find the max value in the last column of delta

  path.append(str(max_state))

  old_state = max_state

  for i in range(n_snps-2, -1, -1): # desde la penultima snp hasta la primera

    current_state = arrows[old_state][i+1] 
    path.append(str(current_state))
    old_state = current_state 
    
    # return negloglikelihood and path
  return max_value, path[::-1]
 # return max_value, "".join(reversed(path))

def _viterbi_baseline (n_states, n_snps, eps, test_hap, panel_haps):
  "viterbi baseline with equal probability to copy each haplotype"
  # initialize
  delta = np.zeros(shape=(n_states, n_snps))
  delta[:,0] = [_emission_helper(test_hap[0], panel_haps[j,0], eps)-np.log(n_states) for j in range(n_states)]
  arrows = np.ndarray(shape=(n_states, n_snps), dtype=object)
  arrows[:,0] = 0


  # main loop
  for i in range(1, n_snps):
    for j in range(0, n_states):

      # store prob
      emission_prob = _emission_helper(test_hap[i], panel_haps[j][i], eps)
      delta[j][i] = np.max(delta[:,i-1] + np.log(1/n_states) + emission_prob)

      # store arrow
      arrows[j][i] = np.argmax(delta[:,i-1] + np.log(1/n_states) + emission_prob)

  # bactracking
  path = []
  max_state = np.argmax(delta[:, -1]) # Find the index of the max value in the last column of delta
  max_value = delta[max_state, -1] # Find the max value in the last column of delta
  path.append(str(max_state))
  old_state = max_state

  for i in range(n_snps-2, -1, -1): # desde la penultima snp hasta la primera

    current_state = arrows[old_state][i+1] 
    path.append(str(current_state))
    old_state = current_state 

  # return log posterior and path
  return max_value, path[::-1]


class LiStephensHMM:
  """ Simple class to apply Li-Stephens HMM """

  def __init__(self, haps, positions):
    assert(haps.shape[1] == positions.size)
    self.haps = haps
    self.positions = positions
    self.n_snps = haps.shape[1]
    self.n_samples = haps.shape[0]
    self.rho = 0.0
    self.theta = 0.0
    self.eps = 0.0

  def _set_params(self, rho, theta):
    """ Set parameters to use as initial values  """
    self.rho = rho
    self.theta = theta

  def _infer_theta(self):
    """ Estimate theta using watterson's estimator (like LS-model) """
    S = np.sum(1.0/np.arange(1,self.n_samples))
    S = 1.0 / S
    return(S)

  def _forward(self, test_hap, scale=1.0, eps=0.001):
    """ Computing the forward algorithm for a given test haplotype as the outcome variable """
    haps = self.haps
    positions = self.positions
    theta = self.theta
    nsamples = self.n_samples
    nsnps = self.n_snps
    # Running compiled helper function with passed arguments for speed
    alphas = _forward_algo(haps, positions, test_hap, eps, nsamples, nsnps, scale)
    return(alphas)

  def _negative_logll(self, test_hap, scale=1.0, eps=0.001):
    alphas = self._forward(test_hap, scale, eps)
    neg_logll =  -np.nansum(alphas[:,-1])
    return(neg_logll)

  def _backward(self, test_hap, scale=1.0, eps=0.001):
    """ Computing the backward algorithm for a given test haplotype as the outcome variable """
    haps = self.haps
    positions = self.positions
    rho = self.rho
    theta = self.theta
    nsamples = self.n_samples
    nsnps = self.n_snps
    # Running compiled helper function with passed arguments for speed
    betas = _backward_algo(haps, positions, test_hap, eps, nsamples, nsnps, scale)
    return(betas)

  def _viterbi(self, test_hap, scale=1.0, eps=0.001):
    """ Viterbi algorithm for a test haplotype as the outcome variable """
    haps = self.haps
    positions = self.positions
    rho = self.rho
    theta = self.theta
    nsamples = self.n_samples
    nsnps = self.n_snps
    # (n_states, n_snps, positions, eps, scale, test_hap, panel_haps)
    posterior, path = _viterbi_helper(nsamples, nsnps, positions, eps, scale, test_hap, haps)
    baseline_post, baseline_path = _viterbi_baseline(nsamples, nsnps, eps, test_hap, haps)
    return(posterior, path, baseline_post, baseline_path)

  def _infer_scale(self, test_hap, eps=0.001, **kwargs):
    """ Inferring scale by numerically minimizing the negative log-likelihood """
    f = lambda s : self._negative_logll(test_hap, scale=s, eps=eps)
    ta = minimize_scalar(f, **kwargs)
    return(ta)

  def _infer_eps(self, test_hap, scale=1e2, **kwargs):
    """ Inferring error by numerically minimizing the negative log-likelihood """
    f = lambda e : self._negative_logll(test_hap, scale=scale, eps=e)
    ta = minimize_scalar(f, **kwargs)
    return(ta)

  def _infer_params(self,test_hap, **kwargs):
    """ Infer error parameter and scale parameter using numerical optimization """
    f = lambda x : self._negative_logll(test_hap, scale=x[0], eps=x[1])
    x = minimize(f, **kwargs)
    return(x)

  # TODO  : is there a more efficient way to do this?
  def _sim_haplotype(self, scale=1e2, eps=1e-2, seed=None):
    """ Simulate a haplotype from the LS model to test against """
    if seed is not None:
        np.random.seed(seed)
    haps = self.haps
    rec_pos = self.positions
    nhaps, nsnps = haps.shape[0], haps.shape[1]
    sim_hap = np.zeros(nsnps)
    idxs = np.zeros(nsnps, dtype=np.uint32)
    # setup the initial random sample
    cur_pos = rec_pos[0]
    idxs[0] = np.random.randint(nhaps)
    sim_hap[0] = np.random.choice([haps[idxs[0],0], 1 - haps[idxs[0],0]], p = [eps, 1. - eps])
    for i in range(1,nsnps):
    # What is the probability that you would jump?
      dij = rec_pos[i] - cur_pos
      pj = 1. - np.exp(-(scale*dij))
      if np.random.binomial(1,pj):
        idxs[i] = np.random.randint(nhaps)
      else:
        idxs[i] = idxs[i-1]

      # set the current position
      cur_pos = rec_pos[i]
    for i in range(1,nsnps):
      sim_hap[i] = np.random.choice([haps[idxs[i],i], 1 -haps[idxs[i],i]], p=[1.-eps, eps])
    sim_hap = sim_hap.astype(np.int8)
    return(sim_hap, idxs)

  def _fill_haplotypes_hwe(self):
    """
      Use HWE within the haplotype panel to fill in any missing variants
      missing variants are encoded as an nan within the panel
    """
    assert(np.sum(self.haps < 0) > 0)
    # Estimate frequencies otherwise
    freqs = np.sum(self.haps, axis=0)
    ns = np.sum((self.haps < 0), axis=0)
    freqs = freqs / (self.n_samples - ns)
    assert(freqs.size == ns.size)
    assert(freqs.size == self.n_snps)
    haps_hwe_filled = np.copy(self.haps)
    for i in range(self.n_snps):
      if freqs[i] <= 0.:
        geno = np.zeros(ns[i], dtype=np.int8)
      elif freqs[i] >= 1.:
        geno = np.ones(ns[i], dtype=np.int8)
      else:
        geno = binomial(1, p=freqs[i], size=ns[i])
      haps_hwe_filled[self.haps[:,i] < 0,i] = geno
    # Fill in the haplotypes via the sampling procedure
    self.haps = haps_hwe_filled.astype(np.int8)

  def _filter_sites(self, test_hap):
    """
      Filter to only sites on the haplotype panel that are typed in the test haplotype
        untyped variants are encoded as np.nan
      TODO : filter out sites that are monomorphic across the panel as well
    """
    assert(test_hap.size == self.n_snps)
    idx = np.where(test_hap >= 0)[0]
    pos = np.copy(self.positions)
    haps = np.copy(self.haps)
    # Doing the indexing ...
    haps = haps[:,idx]
    pos = pos[idx]

    # resetting the system level variables
    self.haps = haps.astype(np.int8)
    self.positions = pos.astype(np.float64)
    self.n_snps = pos.size
