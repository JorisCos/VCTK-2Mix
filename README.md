# VCTK-2Mix

### About the dataset
VCTK-2Mix is an open source dataset for source separation in noisy 
environments. It is derived from VCTK signals and WHAM noise.
It is meant as a test set. It will also enable cross-dataset experiments.

### Generating VCTK-2Mix
To generate VCTK-2Mix, clone the repo and run the main script : 
[`generate_VCTK-2.sh`](./generate_VCTK-2.sh)

```
git clone https://github.com/JorisCos/VCTK-2Mix
cd VCTK-2Mix
./generate_VCTK-2mix.sh storage_dir
```
You can either change `storage_dir` by hand in 
the script or use the command line.  
By default, VCTK-2mix will be generated for 2,
at both 16Khz and 8kHz, 
for min max modes, and all mixture types will be saved (mix_clean, 
mix_both and mix_single). This represents around **7GB** 
of data.
You will also need to store VCTK and wham_noise during
generation for an additional **18GB** and **35GB**.

 
### Note on scripts
For the sake of transparency, we have released the metadata generation 
scripts. However, the dataset shouldn't be changed under any 
circumstance.