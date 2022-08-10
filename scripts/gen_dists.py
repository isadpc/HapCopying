#!/usr/bin/env python3

import os
import sys
import numpy as np
import tszip
import argparse

parser = argparse.ArgumentParser(description='extraction of genetic distances from tsfile')
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


np.savez(args.output + str(int(ta_test)) +"_gendist.npz",
    dist=asc_pos)
