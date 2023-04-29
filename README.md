# Repository for "Redrawing attendance boundaries to promote racial and ethnic diversity in elementary schools"

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-xie2018" class="csl-entry">

**Citation:**
Nabeel Gillani, Doug Beeferman, Christine Vega-Pourheydarian, Cassandra Overney, Pascal Van Hentenryck, Deb Roy. *Redrawing attendance boundaries to promote racial and ethnic diversity in elementary schools*. Forthcoming in *Educational Researcher*.
<https://www.ssrn.com/abstract=4387899>.

</div>

</div>

The notes below describe the key code and data files used in / produced from this study.  They likely miss some details and context that might be helpful for replicating and/or building off of this work.  If you find that's the case, please do not hesitate to reach out to <n.gillani@northeastern.edu> with any questions, comments, etc.!

# Code

Python package dependencies can be found in `requirements.txt`.  The key code folders/files are as follows:

* `utils/` -> contains code to run the data pipeline (see `data_prep_pipeline.py` in this folder).  Note: travel time matrices were computed on an external server due to large memory requirements.  Files produced by this pipeline can be found in the data folders named `data/derived_data` and `models/solver_files`(more info on these and other data folders found below)
* `models/` -> contains code for the rezoning algorithm (see `assignment_cp_sat.py`).  `simulation_sweeps.py` enables the creation of different parameter configurations to run through the optimization model in the former file (e.g., using a parallel computing cluster)
* `analysis/` -> contains code for sifting through outputs from different simulation configurations/runs and computing key outcome measures.  The file `analyze_model_solution.py` does this and outputs a set of CSVs per district, which are then read in by `analyze_rezoning_sim_outputs.Rmd` for further processing and analysis.  This R markdown file is also responsible for generating the main plots in the paper.  
* `frontend/` -> contains code for the dashboard hosted at <https://schooldiversity.org>, which shows simulation results for different districts across the US.  The dashboard was built using the data dashboarding library Streamlit, and hence, includes only Python code.  

# Data

* `data.zip` -> This should be unzipped and stored in the root folder of the repository.  It contains two main subfolders:
	- `derived_data/` -> this includes files per state output from the first four functions called in `utils/data_prep_pipeline.py` (described above).  Namely, it includes a file that represents an estimate of how many students per racial/ethnic group attend each each zoned (boundary-assigned) school from each school (`blocks_file_for_assignment.csv`).  This also constitutes the block-to-school mapping derived from the 21/22 school attendance boundaries data described in the paper.  **NOTE:** due to gaps in data availability in the common core of data, the `num_frl` (free/reduced lunch) and `num_ell` (English-language learner) estimates are unreliable, and hence, not used in the paper.
	- `school_covariates/` -> this includes several files used for the school choice optout analysis (`school_choice_analysis.csv` -> produced by `analysis/analyze_districts.py`) and other district-level covariates used in the analyses described in the main text.
	- `prepped_csvs_for_analysis` -> this includes the resulting CSVs per district produced by running `analysis/analyze_model_solution.py` over a given set of simulation outputs found in `simulation_outputs/`.  These CSVs represent the expected impacts of each hypothetical rezoning (represented by a row in each CSV) on integration, travel times, and other outcomes of interest.  The CSV files are then loaded into `analysis/analyze_rezoning_sim_outputs.Rmd` for further processing and visualization.
* `solver_files.zip` -> This should be unzipped and stored in the `models/` folder.  The folder contains files needed to run `models/assignment_cp_sat.py` -- i.e., the rezoning algorithm.  In particular, there are three files per district: 1) `prepped_file_for_solver_2791448.csv`, which is a processed version of `blocks_file_for_assignment.csv` (processed by `utils/model_utils.py`)
* `simulation_outputs.zip` -> This should be unzipped and stored in the root folder of the repository.  It contains ...
