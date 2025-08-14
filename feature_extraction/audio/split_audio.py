import os
import tqdm
import glob
import sys

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_file_path)))
import config
import argparse


def func_split_audio_from_video_16k(video_root, save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for video_path in tqdm.tqdm(glob.glob(video_root + "/*")):
        videoname = os.path.basename(video_path)[:-4]
        audio_path = os.path.join(save_root, videoname + ".wav")
        if os.path.exists(audio_path):
            continue
        cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" % (
            config.PATH_TO_FFMPEG,
            video_path,
            audio_path,
        )  # linux
        # cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" %(config.PATH_TO_FFMPEG_Win, video_path, audio_path) # windows
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="/sda/xyy/mer/MERTools/MER2023-Dataset-Extended/mer2023-dataset-process/",
        help="file name",
    )
    args = parser.parse_args()

    dataset = args.dataset

    video_root = os.path.join(dataset, "video")
    save_root = os.path.join(dataset, "audio")
    func_split_audio_from_video_16k(video_root, save_root)
