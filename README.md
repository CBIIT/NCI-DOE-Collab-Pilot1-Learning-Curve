# NCI-DOE-Collab-Pilot1-Learning_Curves

### Description:
Learning curves is an empirical method that allow to evaluate whether a supervised learning model can be further improved with more training data. The trajectory of the curves provides a forward-looking metric for analyzing prediction performance and can serve as a co-design tool to guide experimental biologists and computational scientists in the design of future experiments in prospective research studies.

### User Community:
Researchers interested in the following topics:
* Primary: Cancer biology data modeling
* Secondary: Machine Learning; Bioinformatics; Computational Biology

### Usability:	
The current code can be used by a data scientist experienced in Python and the domain.

### Uniqueness:	
Learning curves is a general method that can be applied to any supervised learning model. In this repo we utilize this method to generate learning curves for two drug response prediction models: LightGBM regressor and a neural network regressor. These curves can be used to evaluate and compare the data scaling properties of prediction modeles across a range of training set sizes rather for a fixed sample size. 

### Components:	
This capability provides the following components:
* Scripts that implements the learning curve method using two machine learning models: LightGBM and Neural Networks.
* Examples on how to apply the learning curve method for models that predict the drug response using data from CTRP. 

### Publication: 
Partin, A., Brettin, T., Evrard, Y.A. et al. Learning curves for drug response prediction in cancer cell lines. BMC Bioinformatics 22, 252 (2021). [https://doi.org/10.1186/s12859-021-04163-y](https://doi.org/10.1186/s12859-021-04163-y)


### Technical Details:
Refer to this [README](./src/README.md).
