#!/bin/bash
set -e  # Exit on error

storage_dir=$1

echo "Download wham_noise into $storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 https://storage.googleapis.com/whisper-public/wham_noise.zip -P $storage_dir
unzip -qn $storage_dir/wham_noise.zip -d $storage_dir
rm -rf $storage_dir/wham_noise.zip


echo "Download VCTK into $storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y -P $storage_dir
unzip -qn $storage_dir/VCTK-Corpus.zip -d $storage_dir
rm -rf $storage_dir/wham_noise.zip

python