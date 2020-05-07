import os
import argparse
import soundfile as sf
import pandas as pd
import glob
from tqdm import tqdm

# Global parameter
NUMBER_OF_SECONDS = 3
RATE = 16000

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--VCTK_dir', type=str, required=True,
                    help='Path to VCTK root directory')


def main():
    VCTK_dir = args.VCTK_dir
    md_dir = os.path.join(VCTK_dir, 'metadata')
    os.makedirs(md_dir, exist_ok=True)
    create_VCTK_metadata(VCTK_dir, md_dir)


def create_VCTK_metadata(VCTK_dir, md_dir):
    """ Generate metadata corresponding to downloaded data in VCTK """
    # Get speakers metadata
    speakers_metadata = create_speakers_dataframe(VCTK_dir)
    # Generate the dataframe relative to the directory
    dir_metadata = create_VCTK_dataframe(VCTK_dir, speakers_metadata)
    # Filter out files that are shorter than 3s
    num_samples = NUMBER_OF_SECONDS * RATE
    dir_metadata = dir_metadata[
        dir_metadata['length'] >= num_samples]
    # Sort the dataframe according to ascending Length
    dir_metadata = dir_metadata.sort_values('length')
    # Write the dataframe in a .csv in the metadata directory
    save_path = os.path.join(md_dir, 'VCTK_md.csv')
    dir_metadata.to_csv(save_path, index=False)


def create_speakers_dataframe(VCTK_dir):
    """ Read metadata from the VCTK_dir dataset and collect infos
    about the speakers """
    speaker_md_path = os.path.join(VCTK_dir, 'speaker-info.txt')
    md_file = pd.read_table(speaker_md_path, engine='python',
                            names=[1, 2, 3, 4, 5, 6],
                            delim_whitespace='True', skiprows=1)
    md_file = pd.concat([md_file.iloc[:, 0], md_file.iloc[:, 2]], axis=1)
    md_file.columns = ['speaker_ID', 'sex']
    return md_file


def create_VCTK_dataframe(VCTK_dir, speakers_md):
    """ Generate a dataframe that gather infos about the sound files in a
    VCTK subdirectory """

    print(f"Creating metadata file")
    # Get the current directory path
    dir_path = os.path.join(VCTK_dir, 'wav16_cut')
    # Recursively look for .flac files in current directory
    sound_paths = glob.glob(os.path.join(dir_path, '*.wav'),
                            recursive=True)
    # Create the dataframe corresponding to this directory
    dir_md = pd.DataFrame(columns=['speaker_ID', 'sex',
                                   'length', 'origin_path'])

    # Go through the sound file list
    for sound_path in tqdm(sound_paths, total=len(sound_paths)):
        # Get the ID of the speaker
        spk_id = os.path.split(sound_path)[1].split('_')[0].replace('p', '')
        # Missing metadata
        if spk_id == '280':
            continue
        # Find Sex according to speaker ID in VCTK metadata
        sex = speakers_md[speakers_md['speaker_ID'] == int(spk_id)].iat[0, 1]
        # Get its length
        length = len(sf.SoundFile(sound_path))
        # Get the sound file relative path
        rel_path = os.path.relpath(sound_path, VCTK_dir)
        # Add information to the dataframe
        dir_md.loc[len(dir_md)] = [spk_id, sex, length, rel_path]
    return dir_md


if __name__ == "__main__":
    args = parser.parse_args()
    main()
