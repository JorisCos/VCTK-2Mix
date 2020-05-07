from scipy.signal import resample_poly
import os
import argparse
import soundfile as sf
import glob
from tqdm import tqdm
import librosa

RATE = 48000
freq = 16000

threshold = 20

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--VCTK_dir', type=str, required=True,
                    help='Path to VCTK root directory')


def main(args):
    VCTK_dir = args.VCTK_dir
    preproccess_VCTK(VCTK_dir)


def preproccess_VCTK(VCTK_dir):
    dir_path = os.path.join(VCTK_dir, 'wav48')
    save_path_cut = os.path.join(VCTK_dir, 'wav16_cut')
    os.makedirs(save_path_cut, exist_ok=True)

    # Recursively look for .wav files in current directory
    sound_paths = glob.glob(os.path.join(dir_path, '**/*.wav'),
                            recursive=True)
    # Go through the sound file list
    for sound_path in tqdm(sound_paths, total=len(sound_paths)):
        source, _ = sf.read(sound_path)
        # Detect silences
        silences = librosa.effects.split(source, top_db=threshold)
        # Cut silence at the beginning
        source_cut, _ = sf.read(sound_path, start=silences[0][0])
        # Downsample to 16 KHz
        resampled_source = resample_poly(source_cut, freq, RATE)
        sound__name_cut = os.path.join(save_path_cut,
                                       os.path.split(sound_path)[1])
        # Save audio file
        sf.write(sound__name_cut, resampled_source, freq)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
