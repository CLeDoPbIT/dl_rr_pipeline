## Detecting Atrial Fibrillation from heart rate data using deep learning

This is the code repository for attempts to overcome the results of the following paper:

"Automated detection of atrial fibrillation using long short-term memory network with RR interval signals"

https://www.sciencedirect.com/science/article/pii/S0010482518301847


#### Data

Authors have provided the processed data sequences (in csv format) as a zip file in the data directory of this repository.

Data is in 100 beat sequences with a 99 beat overlap.  The first element of every row in the data file specifies how many beats in each sequence were annotated as exhibiting signs of atrial fibrillation.

To recreate the datasets used in the paper just extract the zip file somewhere, fix the directory in the code, and run the data exploration scripts in the data directory.  As an added bonus you should get some nice plots too! :)


#### Prepare to use:
  * git clone https://github.com/CLeDoPbIT/dl_rr_pipeline
  * cd dl_rr_pipeline
  * pip install requirements.txt  
 
#### Usage:
  * ProcessorRegistry.json contains a sequential list of processors to be started. If you want to add a new module, stick to the old logic. Modules with "forceCreate": "True" will start anyway. Otherwise, the module will not start if data has already been created as a result of its operation.
  * The current release implements data preprocessing for two neural networks: Fully connected network and ResNet_1D
  * Insert your module to models directory. Don't forget to create configs as in current examples.
  * You will also need to add your constants to the constants.json
  * Just run main.py
