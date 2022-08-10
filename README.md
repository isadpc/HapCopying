# HapCopying

This repository contains the code and data used in my master's thesis project, titled 'the Limits of Haplotype-Based approaches: Exploring the Applicability of the Li&Stephens Haplotype-Copying Model to Ancient Samples'.

## Abstract

In the era of next-generation sequencing, ancient human genomes are becoming readily available. However, issues like postmortem degradation or contamination from modern-day human DNA still pose a barrier in its analysis. Many computational methods have surged to deal with these issues; recently, some do so within the framework of the Li and Stephens haplotype-copying model. Well-known to population geneticists, this framework provides way to reconstruct a target haplotype as an imperfect mosaic of a set of reference haplotypes
in a copying process dictated by a hidden Markov model. This project investigates the applicability of the model to ancient target haplotypes, as no previous extensive research
supports it. 
The model’s robustness was evaluated on simulated genetic variation data at the population level, using a reference panel of modern haplotypes and sampling ancient target haplotypes at different time points. Estimates of the model parameters were obtained for each target haplotype through a maximum likelihood approach. The model’s performance was
then compared to that of a baseline model with the freedom to copy from any reference haplotype at any locus.
The results point to good model performance for target haplotypes as old as 900,000 years in the simplest-case scenario of constant-sized continuous populations. Although this
suggests the model’s applicability to ancient DNA from anatomically modern humans, a more definitive answer should be reached by considering scenarios that reflect the complexity of the demographic history of humans. Nevertheless, this work provides a starting point for assessing the haplotype-copying framework when applied to ancient DNA data.
