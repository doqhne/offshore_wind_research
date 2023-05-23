# LLJ_research

This repository contains my code for my research on how offshore wind farms might affect low level jets (LLJ's).

#### Note: Data files and output are not included in this repository.

### Description of files in this repository:
- detection.py: Code that detects low level jets using WRF outputs for a given wind farm parametrization. Output is a csv 
  file that contains information on LLJ-classification, nose heights, nose wind speed, nose wind direction, and shear and 
  veer values for various layers of the atmosphere.
- compare.py: Code that compares two files containing information on LLJ's and produces plots to summarize the data
- run_compare.sh: Script to easily run compare.py for multiple file pairs.
- utils.py: File containing functions used in detection.py
- detection2/detection2.py: Another (better) LLJ detection script that has a lower shear threshold, and also includes Obukhov length and BVF metrics
- detection2/utils.py: Utility functions for detection2.py
- notebooks/ttest_noseheights.ipynb: Test if NWF and WF data have significantly different nose heights
- notebooks/sampleprofile.ipynb: Create a labelled sample wind profile for an LLJ
