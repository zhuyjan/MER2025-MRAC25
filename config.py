# ----------------- path to train/test info
PATH_TO_TRAIN_LST = "/mnt/public/gxj_2/EmoNet_Pro/lst_train/mer25_train_val"
PATH_TO_TEST_LST = "/mnt/public/gxj_2/EmoNet_Pro/lst_test/"

# ----------------- emo / index matching rules
EMO_RULE = {
    "MER": ["neutral", "angry", "happy", "sad", "worried", "surprise"],
    "CREMA-D": ["neutral", "angry", "happy", "sad", "fear", "disgust"],
    "TESS": ["neutral", "angry", "happy", "sad", "fear", "disgust"],
    "RAVDESS": [
        "neutral",
        "angry",
        "happy",
        "sad",
        "fear",
        "disgust",
        "surprised",
        "calm",
    ],
}

# ----------------- features can be used
FEAT_VIDEO_DICT = {
    "senet50face_UTT": [
        512,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/senet50face_UTT",
    ],
    "resnet50face_UTT": [
        512,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/resnet50face_UTT",
    ],
    "clip-vit-large-patch14-UTT": [
        768,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/clip-vit-large-patch14-UTT",
    ],
    "clip-vit-base-patch32-UTT": [
        512,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/clip-vit-base-patch32-UTT",
    ],
    "videomae-large-UTT": [
        1024,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/videomae-large-UTT",
    ],
    "videomae-base-UTT": [
        768,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/videomae-base-UTT",
    ],
    "manet_UTT": [
        1024,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/manet_UTT",
    ],
    "emonet_UTT": [
        256,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/emonet_UTT",
    ],
    "dinov2-large-UTT": [
        1024,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/dinov2-large-UTT",
    ],
    "InternVL_2_5_HiCo_R16-UTT": [
        4096,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/InternVL_2_5_HiCo_R16-UTT",
    ],
}

FEAT_AUDIO_DICT = {
    "chinese-hubert-large-UTT": [
        1024,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/chinese-hubert-large-UTT",
    ],
    "Qwen2-Audio-7B-UTT": [
        1280,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/Qwen2-Audio-7B-UTT",
    ],
    "chinese-hubert-base-UTT": [
        768,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/chinese-hubert-base-UTT",
    ],
    "whisper-large-v2-UTT": [
        1280,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/whisper-large-v2-UTT",
    ],
    "chinese-wav2vec2-large-UTT": [
        1024,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/chinese-wav2vec2-large-UTT",
    ],
    "chinese-wav2vec2-base-UTT": [
        768,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/chinese-wav2vec2-base-UTT",
    ],
    "wavlm-base-UTT": [
        768,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/wavlm-base-UTT",
    ],
}

FEAT_TEXT_DICT = {
    "chinese-roberta-wwm-ext-large-UTT": [
        1024,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/chinese-roberta-wwm-ext-large-UTT",
    ],
    "chinese-roberta-wwm-ext-UTT": [
        768,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/chinese-roberta-wwm-ext-UTT",
    ],
    "chinese-macbert-large-UTT": [
        1024,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/chinese-macbert-large-UTT",
    ],
    "chinese-macbert-base-UTT": [
        768,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/chinese-macbert-base-UTT",
    ],
    "bloom-7b1-UTT": [
        4096,
        "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/bloom-7b1-UTT",
    ],
}

MODEL_DIR_DICT = {}
