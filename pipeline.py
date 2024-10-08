from RobustVideoMatting.inference import  convert_video
from RobustVideoMatting.model import MattingNetwork
from replace_color import replace
from chroma_key_replacer import chroma_replace
import torch
import os


model = MattingNetwork('resnet50').eval().cpu()  # or "resnet50"
model.load_state_dict(torch.load('rvm_resnet50.pth'))
original_video_path = 'headphones.mp4'
output_video = f'{os.path.splitext(os.path.basename(original_video_path))[0]}_output.mp4'
output_composition = f'./sequence/{os.path.splitext(output_video)[0]}'
full_background_path = f'full_backgroud.mp4'
phone_background_path = f'phone_background.mp4'


"""
convert_video(
    model,                           # The model, can be on any device (cpu or cuda).
    #input_source='./../BackgroundMattingV2/test/headphones.mp4',        # A video file or an image sequence directory.
    input_source='./../output.mp4',        # A video file or an image sequence directory.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition='com.mov',    # File path if video; directory path if png sequence.
    output_alpha="pha.mov",          # [Optional] Output the raw alpha prediction.
    output_foreground="output_foreground.mov",          # [Optional] Output the raw alpha prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
    seq_chunk=24,                    # Process n frames at once for better parallelism.
)
"""

if not os.path.exists(output_video):
    replace(original_video_path)

if not os.path.exists(output_composition):
    os.makedirs(output_composition)

    convert_video(
        model,  # The model, can be on any device (cpu or cuda).
        # input_source='./../BackgroundMattingV2/test/headphones.mp4',        # A video file or an image sequence directory.
        input_source=f'{output_video}',  # A video file or an image sequence directory.
        output_type='png_sequence',  # Choose "video" or "png_sequence"
        output_composition=output_composition,  # File path if video; directory path if png sequence.
        output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
        downsample_ratio=None,  # A hyperparameter to adjust or use None for auto.
        seq_chunk=24,  # Process n frames at once for better parallelism.
    )

chroma_replace(output_video, full_background_path, phone_background_path)


# Исходное видео передается в реплейс_колор
# Вызов нейронки конверта видео
# Получаем пнг секвенцию с кадрами из видео альфа каналов
# вызов эплай хромакей, передаю бекграунд и передаешь путь к папке с пнг секвенцией