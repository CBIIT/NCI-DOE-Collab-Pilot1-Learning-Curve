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
Learning curves is a general method that can be applied to any supervised learning model. In this repo we utilize this method to generate learning curves for two drug response prediction models: LightGBM Regressor and a neural network regressor. These curves can be used to evaluate and compare the data scaling properties of prediction modeles across a range of training set sizes rather for a fixed sample size. 

### Components:	
The following components are in the Learning Curves dataset in the Model and Data Clearinghouse (MoDaC):
* Tabular data: 
  * Machine learning data for training prediction models models and generating learning curves.
* Learning curve data:
  * Raw learning curve data (lc.out.ctrp.lgb) from training LightGBM regressor with the CTRP drug response dataset.
  * Raw learning curve data (lc.out.ctrp.nn_reg0) from training a neural network regressor with the CTRP drug response dataset.

### Technical Details:
Refer to this [README](./src/README.md).
