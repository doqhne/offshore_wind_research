#!/bin/bash

python compare.py \
    --nwf_file NWF_vwcent.csv \
    --wf_file VW100_vwcent.csv \
    --wf_name VW100 \
    --location "Vineyard winds centroid" \
    --plot_path output_plots/compare_vwcent_plots

python compare.py \
    --nwf_file NWF_nebuoy.csv \
    --wf_file CA100_nebuoy.csv \
    --wf_name CA100 \
    --location "NE buoy" \
    --plot_path output_plots/compare_nebuoy_plots

python compare.py \
    --nwf_file NWF_long_island.csv \
    --wf_file CA100_long_island.csv \
    --wf_name CA100 \
    --location "Long Island" \
    --plot_path output_plots/compare_longisland_plots

python compare.py \
    --nwf_file NWF_marthas_vineyard.csv \
    --wf_file VW100_marthas_vineyard.csv \
    --wf_name VW100 \
    --location "Martha's Vineyard" \
    --plot_path output_plots/compare_marthasvineyard_plots