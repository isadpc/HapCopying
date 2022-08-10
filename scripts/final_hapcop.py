#!/usr/bin/env python3

import os
import sys
import numpy as np
import tszip
from tqdm import tqdm

from li_stephens import LiStephensHMM

# Using the splines to estimate the
from scipy.interpolate import UnivariateSpline

import argparse 

parser = argparse.ArgumentParser(description='mle estimation of jump rate and copying error parameters + viterbi algorithm')
parser.add_argument('-f', '--file', metavar='', help='file containing reference modern haplotypes plus test haplotypes from an ancient individual')
parser.add_argument('-o', '--output', metavar='', help='output path')
args = parser.parse_args()

def ascertain_variants(hap_panel, pos, maf=0.05):
    """Ascertaining variants based on frequency
    NOTE: performs additional filter for non-zero recombination map distance
    """
    assert (maf < 0.5) & (maf > 0)
    mean_daf = np.mean(hap_panel, axis=0)
    af_idx = np.where((mean_daf > maf) | (mean_daf < (1.0 - maf)))[0]
    # filter positions that are not recombinationally distant
    pos_diff = pos[1:] - pos[:-1]
    idx_diff = pos_diff > 0.0
    idx_diff = np.insert(idx_diff, True, 0)
    # Treat this as the logical and of the MAF check and
    # ascertainment checks ...
    idx = np.logical_and(af_idx, idx_diff)
    asc_panel = hap_panel[:, idx]
    asc_pos = pos[idx]
    return (asc_panel, asc_pos, idx)

def calc_se_finite_diff(f, mle_x, eps=1e-1):
    """f is the loglikelihood function."""
    xs = np.array([mle_x - 2 * eps, mle_x - eps, mle_x, mle_x + eps, mle_x + 2 * eps])
    ys = np.array(
        [
            f(mle_x - 2 * eps),
            f(mle_x - eps),
            f(mle_x),
            f(mle_x + eps),
            f(mle_x + 2 * eps),
        ]
    )
    dy = np.diff(ys, 1)
    dx = np.diff(xs, 1)
    yfirst = dy / dx
    xfirst = 0.5 * (xs[:-1] + xs[1:])
    dyfirst = np.diff(yfirst, 1)
    dxfirst = np.diff(xfirst, 1)
    ysecond = dyfirst / dxfirst
    se = 1.0 / np.sqrt(-ysecond[1])
    return (ysecond, se)

rec_rate = 10**-8

ts = tszip.decompress(args.file)

hap_panel_test = ts.genotype_matrix().T

phys_pos = np.array([v.position for v in ts.variants()])
rec_pos = phys_pos * rec_rate
pos = rec_pos
avg_gen_dist = np.mean(rec_pos, axis = 0)

node_ids = [s for s in ts.samples()]
tree = ts.first()
times = np.array([tree.time(x) for x in node_ids])

seed = 100

mod_idx = np.where(times == 0)[0]

# they are taking only one test haplotype (haploid L&S?)
ta_test = times[np.where(times != 0)[0][0]]
ta_idx = np.where(times != 0)[0][0]


# Extracting the panel
modern_hap_panel = hap_panel_test[mod_idx, :]
# Test haplotype
test_hap = hap_panel_test[ta_idx, :]

mod_asc_panel, asc_pos, asc_idx = ascertain_variants(
    modern_hap_panel, pos, maf= 5 / 100.0
)
anc_asc_hap = test_hap[asc_idx]

# Edit distance of ancient to panel normalized by length (n_snps)
dists = np.abs(np.subtract(mod_asc_panel, anc_asc_hap)).sum(axis=1)/len(anc_asc_hap)

afreq_mod = np.sum(mod_asc_panel, axis=1)

cur_hmm = LiStephensHMM(haps=mod_asc_panel, positions=asc_pos)
#cur_hmm.theta = cur_hmm._infer_theta()

scales = np.logspace(2, 6, 30)
neg_log_lls = np.array(
    [
        cur_hmm._negative_logll(anc_asc_hap, scale=s, eps=1e-2)
        for s in tqdm(scales)
    ]
)
min_idx = np.argmin(neg_log_lls)
print(scales, neg_log_lls)
scales_bracket = (1.0, scales[min_idx] + 1.0)
neg_log_lls_brack = (0, neg_log_lls[min_idx])
print(ta_test, scales_bracket, neg_log_lls_brack)
mle_scale = cur_hmm._infer_scale(
    anc_asc_hap, eps=1e-2, method="Brent", bracket=scales_bracket
)

# Calculate the marginal standard error
f = lambda x: -cur_hmm._negative_logll(anc_asc_hap, scale=x, eps=1e-2)
_, se_finite_diff = calc_se_finite_diff(f, mle_scale.x)

# Estimating both error and scale parameters jointly
bounds = [(10, 1000000e4), (1e-4, 0.25)]
mle_params = cur_hmm._infer_params(
    anc_asc_hap, x0=[mle_scale.x, 1e-2], bounds=bounds
)
cur_params = np.array([np.nan, np.nan])
se_params = np.array([np.nan, np.nan])
if mle_params["success"]:
    cur_params = mle_params["x"]
    se_params = np.array(
        [
            np.sqrt(mle_params.hess_inv.todense()[0, 0]),
            np.sqrt(mle_params.hess_inv.todense()[1, 1]),
        ]
    )
    
    posterior, path, post_baseline, path_baseline = cur_hmm._viterbi(anc_asc_hap, scale=cur_params[0], eps=cur_params[1])
    
else:
    print(mle_params)

model_params = np.array([mod_asc_panel.shape[0], asc_pos.size, ta_test])
print('MLE jump rate:', mle_scale["x"], 'SE:', se_finite_diff)
print('MLEs:', cur_params)
print('SE:', se_params)
print(model_params)


def dist_to_path (path):
    "calculate avg edit distance to viterbi path"
    path = np.array(path, dtype=('int64'))
    states = np.unique(path)
    n_cp_states = len(states)

    dist_cp = 0
    for state in states:
        state_idx = np.where(path==state)
        mod_hap = mod_asc_panel[state]
        dist_cp += np.abs(np.subtract(mod_hap[state_idx], anc_asc_hap[state_idx])).sum()
    avg_dist_cp = dist_cp/len(anc_asc_hap)
    return avg_dist_cp, n_cp_states

"""
def dist_to_path(path):
    path = np.array(path, dtype=('int64'))
    dist = 0
    avg_dist = 0 
    length = 0
    n_jumps = 0
    for i in range(len(path)):
        length += 1
        if path[i] != path[i-1]:
            n_jumps += 1
            avg_dist += dist/length
            length = 0 
            
        if mod_asc_panel[path[i]][i] != anc_asc_hap [i]:
            dist += 1 
    return avg_dist/(n_jumps+1), n_jumps

"""
avg_dist_cp, n_cp_states = dist_to_path (path)
avg_dist_baseline, n_cp_states_bl = dist_to_path (path_baseline)

def count_jumps (path):
    # counting number of jumps
    n_jumps = 0
    for i in range(len(path)):
        if path[i] != path[i-1]:
            n_jumps += 1
    return n_jumps

n_jumps = count_jumps(path)
n_jumps_bl = count_jumps(path_baseline)

np.savez(
    args.output + str(int(ta_test)) +"_"+ str(len(modern_hap_panel)) +".npz",
    loglls=-neg_log_lls,
    scales = scales,
    scale=mle_scale["x"],
    se_scale_finite_diff=se_finite_diff,
    mle_params = mle_params,
    params=cur_params,
    se_params=se_params,
    model_params=model_params,
    path_post = posterior,
    path=path,
    path_bl = path_baseline,
    post_bl = post_baseline,
    time = ta_test,
    n_init_snps = len(phys_pos),
    n_asc_snps = len(asc_pos),
    avg_gen_dist = avg_gen_dist,
    edit_dist = dists,
    n_copied_states = np.unique(path),
    cp_states_bl = np.unique(path_baseline), 
    avg_edit_dist = avg_dist_cp,
    avg_edit_bl = avg_dist_baseline,
    jumps = n_jumps,
    jumps_bl = n_jumps_bl 
)

