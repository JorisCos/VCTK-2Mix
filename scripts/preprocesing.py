from scipy.signal import resample_poly
import os
import argparse
import soundfile as sf
import glob
import tqdm.contrib.concurrent
import librosa
import functools

RATE = 48000
freq = 16000

threshold = 20

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--VCTK_dir', type=str, required=True,
                    help='Path to VCTK root directory')
parser.add_argument('-j', '--parallel', type=int, default=1,
                    help='Number of threads to use')


def main(args):
    VCTK_dir = args.VCTK_dir
    n_workers = args.parallel
    preprocess_VCTK(VCTK_dir, n_workers)


def preprocess_file(sound_path, save_path_cut):
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


def preprocess_VCTK(VCTK_dir, n_workers):
    dir_path = os.path.join(VCTK_dir, 'wav48')
    save_path_cut = os.path.join(VCTK_dir, 'wav16_cut_fixed')
    os.makedirs(save_path_cut, exist_ok=True)

    # Recursively look for .wav files in current directory
    sound_paths = glob.glob(os.path.join(dir_path, '**/*.wav'),
                            recursive=True)
    # Go through the sound file list
    tqdm.contrib.concurrent.thread_map(
        functools.partial(preprocess_file, save_path_cut=save_path_cut),
        sound_paths,
        max_workers=n_workers
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
