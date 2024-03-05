# offshore_wind_research

This repository contains my code for my research on how offshore wind farms might affect local meteorology.

### Description of files in this repository:
- detection2/ contains scripts used for my analysis on how offshore wind farms modify low-level jets. 
    - detection2.py : code to detect LLJ profiles from wrfout files
    - utils.py : Additional functions used in detection2.py
    - LLJ_files/ contains code to plot results for a large array of points
        - bubble_plots.ipynb : Make maps of LLJ occurrences, nose heights, etc for array of locations. Code used to generate figures 2, 6, 7, 8, 12, 13, 14, and 15 in Offshore wind farms modify low-level jets.
- micro_work/ contains scripts used for my analysis on the micrometeorological impacts of offshore wind farms
    - pblh_wakes3.py : Code to find the wind farm wake area and distance for all stable times, also generates plots showing detected wake
    - plots2.py : Make plots comparing the NWF and LA100 simulations for different variables
    - wake_detection_clean.ipynb : A demonstration of the method used to determine the wake area/distance in pblh_wakes3.py
    - plot_panels.py : Make panel plots comparing the NWF and LA100 simulations for differernt variables
- notebooks/ contains notebooks used to analyze LLJ data. 
    - d2_panels.ipynb : Code used to create figures 10, 11, 16, 17, 18, 19, and 20 in Offshore wind farms modify low-level jets
    - fast_LLJ.ipynb : Code to create fig 21
    - general_plots.ipynb : Code to create fig 5
    - loc_map.ipynb : Code to create fig 1
    - rmol_scatter.ipynb : Code to create fig 4
    - sample_profile.ipynb : code to create fig 3
    - ttest_nose_heights.ipynb : A students-t test to validate nose height results
    
    
