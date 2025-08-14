
project_dir="/sda/xyy/mer/MERTools/MER2023-Dataset-Extended/mer2023-dataset-process/"

# --------------- 视觉预处理
# --- 提取人脸
# cd ./visual
# python extract_openface.py --dataset=$project_dir --type="videoOne"
# cd ../
# #
# # --- 提取视觉特征
# cd ./visual
# python -u extract_vision_huggingface.py --dataset=$project_dir --feature_level='UTTERANCE' --model_name='clip-vit-large-patch14' --gpu=0 
# cd ../

# # --------------- 音频预处理
# # --- 分离音频
# cd ./audio
# python split_audio.py --dataset=$project_dir
# cd ../
# #
# # --- 提取音频特征
# cd ./audio
# python -u extract_audio_huggingface.py --dataset=$project_dir --feature_level='UTTERANCE' --model_name='chinese-hubert-large' --gpu=0
# cd ../

# # --------------- 文本预处理
# #--- 获取文本
# cd ./text
# python split_asr.py  --dataset=$project_dir
# cd ../
# #
# # --- 提取文本特征
cd ./text
python extract_text_huggingface.py --dataset=$project_dir --feature_level='UTTERANCE' --model_name='bloom-7b1' --gpu=0
cd ../

