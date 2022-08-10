# loading libraries ------------------------------------------------------------
library(slendr)
library(ggplot2)
library(dplyr)
library(reticulate)
library(tidyverse)

# setting up virtual environment and checking requirements ---------------------
#setup_env("./venv")
use_virtualenv("./venv")
check_env()

set.seed(100)


# defining pops ----------------------------------------------------------------
AF_toy <- population('African', time=200000, N=14375, parent='ancestor')
EU_toy <- population('European', time=100000, N=11250, parent=AF)
AS_toy <- population('Asian', time=23000, N=10000, parent=EU)
#AM <- population('American', time=22000, N=7500, parent=AS)

# defining model ---------------------------------------------------------------
model_toy <- compile_model(
  populations = list(AF_toy, EU_toy, AS_toy),
  generation_time = 30,
  path = paste0("/home/projects/HapCopying/simulations/toy_1MB"),
  overwrite = TRUE
)

# schedule sampling -----------------------------------------------------------
af_samples_toy <- schedule_sampling(model_toy, 
                                times=seq(from=30, to=200000, by=600),
                                list(AF_toy,20))

eu_samples_toy <- schedule_sampling(model_toy, 
                                times=seq(from=30, to=100000, by=600),
                                list(EU_toy,20))

as_samples_toy <- schedule_sampling(model_toy, 
                                times=seq(from=30, to=23000, by=600),
                                list(AS_toy,20))

#am_samples <- schedule_sampling(model_4pop, 
#times=seq(from=30, to=22000, by=900),
#list(AM,20))

modern_samples_toy <- schedule_sampling(model_toy,
                                    times=0,
                                    list(AF_toy,2000),
                                    list(EU_toy,2000),
                                    list(AS_toy,2000))

samples_toy <- rbind(af_samples_toy, eu_samples_toy, as_samples_toy, modern_samples_toy)

# visualizing demographic model ------------------------------------------------
cowplot::plot_grid(
  plot_model(model_toy, sizes = TRUE)
)


# running simulation------------------------------------------------------------
slim(model_toy, 
     sequence_length = 1e6, 
     recombination_rate = 1e-8, 
     sampling = samples_toy,
     output = file.path(model_toy$path, "output_toy"),
     coalescent_only = FALSE,
     random_seed = 300)


ts_file_toy <- file.path(model_toy$path, "output_toy_slim.trees")
file.exists(ts_file_toy)

# recapitation -----------------------------------------------------------------
ts_toy_simp <- ts_load(model_toy, recapitate = TRUE, mutate = TRUE, simplify = TRUE,
                        recombination_rate = 1e-8, Ne = 10000, mutation_rate = 2e-8,
                        random_seed = 100)


data_toy <- ts_data(ts_toy_simp)

# importing tszip python module and saving tree sequence file -----------------
tszip <- import_from_path("tszip", 
                          path="/net/dslave2/home/people/s202406/.local/lib/python3.8/site-packages/")
# import tskit module
tskit <- import_from_path("tskit", path="/home/people/s202406/.local/lib/python3.8/site-packages/")

# Efficient way to save the ts files ------------------------------------------

## panel with haplotypes from all different individuals, ie only retaining one
## node per individuals (trick is to retain even node ids only) 

modern_nodes <- as_tibble(data_toy) %>%
  filter (pop == 'African' & remembered == TRUE & time == 0 & (node_id %% 2) == 0 ) 


## test individuals -----------------------------------------------------------
try <- as_tibble(data_toy)  %>% 
  filter (pop == 'African' & remembered == TRUE)

# change the by = to save more or less individuals, multiple of 600
for (time_point in seq(from=630, to=199830, by=3000)){
  node <- try %>%
    filter (time == time_point) %>%
    slice(1) %>%
    select (node_id)
  all_nodes <- c(modern_nodes$node_id[seq(from=1, to=2000, by=20)], node$node_id[1])
  cur_ts <- tskit$TreeSequence$simplify(ts_toy_simp, samples=all_nodes)
  tszip$compress(cur_ts, 
                 paste ("/home/projects/HapCopying/ts_files/1MB_100panel/toy_",time_point,".trees.tsz", sep=""))
}


# saving all modern individuals to check whether they have coalesced -----------
modern_afr <- data_toy %>%
  filter(pop == 'African' & remembered==TRUE & time==0)

ts_modern_afr <- ts_simplify (ts_toy_simp, 
                          simplify_to = c(modern_afr$name),
                          keep_input_roots = TRUE)
tszip$compress(ts_modern_afr, "/home/projects/HapCopying/ts_files/all_modern_afr.trees.tsz")

# same with only the panel of 100 haps -----------------------------------------
modern_100_ts <- tskit$TreeSequence$simplify(ts_toy_simp, samples=c(modern_nodes$node_id[1:100]))
tszip$compress(modern_100_ts, "/home/projects/HapCopying/ts_files/100_modern_afr.trees.tsz")

# no recapitation -------------------------------------------------------------
ts_toy_norecap <- ts_load(model_toy, recapitate = FALSE, mutate = TRUE, simplify = TRUE,
                       recombination_rate = 1e-8, Ne = 10000, mutation_rate = 2e-8,
                       random_seed = 100)
data_norecap <- ts_data(ts_toy_norecap)

modern_afr <- data_norecap %>%
  filter(pop == 'African' & remembered==TRUE & time==0)

ts_modern_afr <- ts_simplify (ts_toy_norecap, 
                              simplify_to = c(modern_afr$name),
                              keep_input_roots = TRUE)
tszip$compress(ts_modern_afr, "/home/projects/HapCopying/ts_files/all_modern_afr_norecap.trees.tsz")
