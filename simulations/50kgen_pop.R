# loading libraries
# Importing packages ----------------------------------------------------------
library(slendr)
library(ggplot2)
library(dplyr)
library(reticulate)
library(tidyverse)

# setting up virtual environment and checking requirements --------------------
#setup_env("./venv")
use_virtualenv("./venv")
check_env()

set.seed(100)


# Defining populations --------------------------------------------------------
ancient <- population('Ancient', time=1500000, N=14375, parent='ancestor')

# Compiling demographic model -------------------------------------------------
model_50kgen <- compile_model(
  populations = list(ancient),
  generation_time = 30,
  direction = 'backward',
  path = paste0("/home/projects/HapCopying/simulations/50kgen_1MB"),
  overwrite = TRUE
)

# Schedule sampling -----------------------------------------------------------
ancient_samples_50kgen <- schedule_sampling(model_50kgen, 
                                    times=seq(from=30, to=1500000, by=600),
                                    list(ancient,1))

modern_samples_50kgen <- schedule_sampling(model_50kgen,
                                        times=0,
                                        list(ancient,4000))

samples_50kgen <- rbind(ancient_samples_50kgen,
                     modern_samples_50kgen)

# Visualizing demographic model -----------------------------------------------
cowplot::plot_grid(
  plot_model(model_50kgen, sizes = TRUE)
)

# Running simulation ----------------------------------------------------------
slim(model_50kgen, 
     sequence_length = 1e6, 
     recombination_rate = 1e-8, 
     sampling = samples_50kgen,
     output = file.path(model_50kgen$path, "output_50kgen"),
     coalescent_only = FALSE,
     random_seed = 300)

# Loading ts file -------------------------------------------------------------
ts_file_50kgen <- file.path(model_50kgen$path, "output_50kgen_slim.trees")
file.exists(ts_file_50kgen)

## Recapitation, simplification and adding mutations --------------------------
ts_50kgen_simp <- ts_load(model_50kgen, 
                       recapitate = TRUE, 
                       mutate = TRUE, 
                       simplify = TRUE,
                       recombination_rate = 1e-8, 
                       Ne = 10000, 
                       mutation_rate = 2e-8,
                       random_seed = 100)

## Loading simulation data ----------------------------------------------------
data_50kgen <- ts_data(ts_50kgen_simp)

# Saving tree sequence files --------------------------------------------------
## Importing python modules ---------------------------------------------------
tszip <- import_from_path("tszip", 
                          path="/net/dslave2/home/people/s202406/.local/lib/python3.8/site-packages/")

tskit <- import_from_path("tskit", 
                          path="/home/people/s202406/.local/lib/python3.8/site-packages/")


# panel with haplotypes from all different individuals, ie only retaining one
# node per individuals (trick is to retain even node ids only)
modern_nodes <- as_tibble(data_50kgen) %>%
  filter (pop == 'Ancient' & remembered == TRUE & time == 0 & (node_id %% 2) == 0 ) 


# test individuals
test_nodes <- as_tibble(data_50kgen)  %>% 
  filter (pop == 'Ancient' & remembered == TRUE)

# change the by = to save more or less individuals, multiple of 600
for (time_point in seq(from=630, to=1499430, by=6000)){
  node <- test_nodes %>%
    filter (time == time_point) %>%
    slice(1) %>%
    select (node_id)
  all_nodes <- c(modern_nodes$node_id[seq(from=1, to=4000, by=40)], node$node_id[1])
  cur_ts <- tskit$TreeSequence$simplify(ts_50kgen_simp, samples=all_nodes)
  tszip$compress(cur_ts, 
                 paste ("/home/projects/HapCopying/ts_files/50kgen_1MB_100panel/50kgen_",time_point,".trees.tsz", sep=""))
}




