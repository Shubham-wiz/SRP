> # Maintaining Hierarchy of Time Series
> ## with bottom forcasts only

### Student Research Project
##### Probabilistic Forecasting of Hierarchical Time Series with Neural Networks Under supervision of Vijaya Krishna Yalavarthi
##### First Pitch: [https://www.overleaf.com/read/jqbbwsbzjvgk](https://www.overleaf.com/read/jqbbwsbzjvgk)
-----------------------------------------------------------------------------------------------------------------------
##### Team Members

    Deepalakshmi Murugesu -313265
    Najeebullah Hussaini  -310850
    Shubham Dwivedi -311620
    Syed Hammad Jaffery -312441
    Usama Abdul samad -311458


#### Updations:
-   ~~try on other datasets as well.~~
-   ~~Returning to get closer to baselines~~
-   ~~Some issues persist with embed dimensions~~
-  ~~tested on 3 more datasets different from baselines~~
-   ~~data seasonality drastically changed due to covid( new or older data to be considered w/o anomalies)~~
-  ~~Lr find seems to give different results #~~
-   ~~NLL updation #~~
 


# Code Understanding
- **Gluon_ts_implementation**
	 The implemntation of  paper "End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series" presented at ICML 2021, Model was implemented in amazons properitery Mxnet/Gluonts , dropped due to lack of customization Possibilities
	 
- **Pytorch Forcasting Implementation**
	_PyTorch Forecasting_ is a PyTorch-based package for forecasting time series with state-of-the-art network architectures.  
	Multivariate predictions were on point but backpropogation of modified loss is created issues.
- **Raw_implementation**
	Model implemented from the scratch using pytorch , main changes are held in this part of the repo.



## Raw_implementation

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




## Method Organization

 - Size of the entire context window utilized during training can be
   divided into 2 sections - first: conditioning context with input data
   points and network predictions conditioned on real data; and other
   one is prediction context, where network predictions are conditioned
   on network output from the ealrier steps.

####  Training Loop: Basic Structure
           For every epoch in Total of epochs:
			Split batches into input and target, (covariates if not 0)
			Take values <-Conditioning window input,Prediction window , 
			Condition Model output on previous values from previos steps
			Scale input if needed
			Optimizer.zeroGrad-> restarts looping without losses from the last step(initiall tensor of zeros)
			Output<-model
			Invert Scale(Output) if required
			Calculate loss->NLL
			Backpropogate loss
  
  ####  Model:  Structure->Probabalistic model with likelihood function
           Model<-Number of Lstm units,Input Dimension, Output Dimensions, hidden Dimensions
	           _init_ everything  using recieved parameters
			      define Stacked lstm Structure
				  μ and σ distibution -> next point for every t_i.prediction length is from output dimensions     
	       Forward<-Input ,covariates (if required)
	       basic idea : h(hidden state),c(cell state)-> output of Lstm cell, h(i)-> passed to lstm(i+1) cell,h(i) and c(i) are recurrent for lstm cell(i+1)
	       define cell-state(c), hidden-state(h)
	       concat covariates and Inputs
		     looping though input chunks: 
				 where first cell ->concat of input and covariates, recurrent input
				 looping again through all lstm units:
					 susquent cells-> prev. output from stack, recurrent input
	       obtain output according to future length given to model
	       obtain μ and σ <-outputs (use softplus if infinites hit) 
           
  ####  Loss: Structure-> Calculates Negative log liklihood for all hierarchies based on bottom forcasts only     
	Loss<-Output,Ground Truth,Hierarchy data
	μ and σ<-from output
            calculate loss for base forcast using Negative liklihood formula
            looping through hierarchies:
	            calculate Negative liklihood loss between truth values and related bottom values
	        total loss= base forcast loss+ hierarchial loss     
            


# To-do
- Test with more datasets
- Generalize the model
- parameter tuning



##  Basic Arch:

![](https://mermaid.ink/img/pako:eNqtU01PwzAM_StRzhtSOfbAATiAtO2yCg4URV6TrdaSpsrH-Cr_nbhFtAMmLuQSO3nPjp_jN15ZqXjOy2ar7VNVgwusuCwbxnzc7By0Nbtt2hjohDGJTlUBbfOJmaDE4NMaLcweVqxAo5hXDpVnd6Bj2nL2KrJZmGdntF5FQ_bjyNtmD0t2ZQ_gEIIaWc_E6jnPwszC4yTVkHjwVSO_vW4doNorKRY-mJHzsxxai3WxzObzi67OROjIO08e6-rzT1esjgrts33LJ9EH3ESKLVpwYPxfWe-FiaP3JDzuDByX8xXd220QrY7_EXSMulI7CHhQQtud0LjXWFsr_0qxAU8M70UFuooaCEHq3aBy4KoaQbNFumeT-66wAXRPO3rNsJElGGk-bVvfkdSBX6X9eUhw0olkYsSbivY7fBm7kxp8sfu4JONpbOGiEof-zyb0KRifcaOcAZRp_N6o5pKHWhlV8jyZEty-TGP5nnAQg12_NBXPQwo947GVaSyuEVLjDM-3oH06VRKDdcthnvuxfv8AcyBF9g?type=png)





