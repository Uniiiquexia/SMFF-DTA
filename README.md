# SMFF-DTA

# step 1: Data Preparation
# The datasets we used are Davis and KIBA. Due to the memory limitation of github, we have separated KIBA into 4 parts. So when you try to run the code with our data, please firstly combine KIBA1, KIBA2, KIBA3 and KIBA4 together.

# step 2: Input Preparation
# As shown in our paper, our model have 3 inputs respectively for both drugs and targets: sequence, structural information and physicochemical properties.
# For target, the structural information is represented by secondary structures, which has been already extracted and provided in data file.
# For drug, you have to prepare fingerprint and atom feature information by running extract_fingerprint.py and extract_atomFeatures.py separatly.

# step 3: Model Training
# Run main.py.
