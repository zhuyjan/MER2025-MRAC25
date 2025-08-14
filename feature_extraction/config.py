# *_*coding:utf-8 *_*
import os
import sys
import socket
import os.path as osp

############ global path ##############
# PATH_TO_PROJECT = (
#     "/sda/xyy/mer/MERTools/MER2023-Dataset-Extended/mer2023-dataset-process/"
# )
# PATH_TO_VIDEO = osp.join(PATH_TO_PROJECT, "video")  # 文件的初始文件夹

# PATH_TO_FEATURES = osp.join(PATH_TO_PROJECT, "features")  # 剪辑片段的特征的保存文件夹


# PATH_TO_RAW_FACE_Win = PATH_TO_VIDEO
# PATH_TO_FEATURES_Win = PATH_TO_FEATURES

############ Models ##############

# pre-trained models, including supervised and unsupervised
PATH_TO_PRETRAINED_MODELS = "/sda/xyy/mer/MERTools/MER2024/tools"
PATH_TO_OPENSMILE = "/sda/xyy/mer/MERTools/MER2024/tools/opensmile-2.3.0/"
PATH_TO_FFMPEG = "/sda/xyy/mer/MERTools/MER2024/tools/ffmpeg-4.4.1-i686-static/ffmpeg"
PATH_TO_WENET = (
    "/sda/xyy/mer/MERTools/MER2024/tools/wenet/wenetspeech_u2pp_conformer_libtorch"
)


PATH_TO_OPENFACE_Win = "/sda/xyy/mer/tools/OpenFace_2.2.0"

PATH_TO_FFMPEG_Win = (
    "/sda/xyy/mer/MERTools/MER2024/tools/ffmpeg-3.4.1-win32-static/bin/ffmpeg"
)
