> # Maintaining Hierarchy of Time Series with bottom forcasts only



## Raw_implementation
Model implemented from the scratch using pytorch , main changes are held in this part of the repo.


#### _Model.py 
- Consists of lstm based architecture with modified loss values, not yet generalised for all the data set, Different Ipynb files are pushed with the same functions inside to track and debug while creating a more generalised loss and Model.

#### _utils.py
- Consists of utility functions for  calculation of  error metrics and spliting batches while scaling the batch data
- This was basically done with more of a dependent dataset, right now we are not working with covariates ,
   so each series is considered independent.

>         Split batch function: inputs a batch and divides into input, target and covariates.
>         Invert Scale : Applied to the target time series for calculation of  the loss functions on the un-scaled time series.
#### _paper_params.py 
- Initally contained paper parameters now is used for optimisation
#### _test.py 
- Contains Initial experimentation and prediction for multivariate series.

# To-do
- ~~Removed _pycache and data files due to size limitations will add url based ver in future.~~
- ~~Point pred and prob both can be implemented with a simple change, here prob ver is done.~~
- ~~Model paramenters used from baselines~~
- ~~Training on more epochs~~
- Test with more datasets
- Generalize the model
- parameter tuning
- organize the code
- weighted loss testing
# Test_Files
- **base_single_run.ipynb** :First test run with updated loss with generated dataset
- **NLL_base_single_generalised.ipynb**  : Second test run with updated loss with generated dataset with a more generalised approach
- **NL+meanbase_single_run_sigma_2_lvl_hiererchy.ipynb** : Not pure NLL, Bottom level Loss with NLL, Hierarchy simple bottom prediction aggregation - truth values 
- **base_single_run_tourism.ipynb**: First test run with updated loss with Tourism dataset
	bottom forcasts showed more uncertainity


