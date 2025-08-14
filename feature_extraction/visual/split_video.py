import os
import os.path as osp
import cv2
from moviepy.editor import VideoFileClip
import argparse
import sys

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
import config

if __name__ == "__main__":
    split_t = config.VIDEO_SPLIT_T  # 分割的间断时长
    duration_t = config.VIDEO_DURATION_T  # 每段分割的持续时长

    # ----------- 需要指定的路径
    video_dir = config.PATH_TO_VIDEO  # 文件的初始文件夹
    save_basedir = config.PATH_TO_VIDEO_CLIP  # 剪辑片段的保存文件夹

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="test",
        help="file name",
    )
    args = parser.parse_args()

    ftitle = args.dataset

    # 视频文件的路径
    video_path = osp.join(video_dir, ftitle + ".mp4")

    # 保存路径
    output_folder = osp.join(save_basedir, ftitle)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # 加载视频
    clip = VideoFileClip(video_path)

    # 视频总时长
    duration = clip.duration

    # 分割视频的起始时间
    start_time = 0

    # 循环分割视频，直到视频结束
    cnt = 0
    for i in range(0, int(duration // split_t)):
        # 计算结束时间，确保不超过视频总时长
        end_time = min(start_time + duration_t, duration)

        # 创建子剪辑
        subclip = clip.subclip(start_time, end_time)

        # 生成输出文件名
        output_filename = os.path.join(output_folder, f"clip_{cnt}.mp4")

        # 保存子剪辑
        subclip.write_videofile(output_filename, codec="libx264")

        # 更新起始时间
        start_time += split_t
        print(f"{cnt} done!")

        cnt += 1

    # 释放资源
    clip.close()

    print("视频分割完成")
