import os
import torch
import argparse
from RobustVideoMatting.inference import convert_video
from RobustVideoMatting.model import MattingNetwork
from replace_color import replace
from chroma_key_replacer import chroma_replace
from configs.editor_config import EditorConfig
from find_elements import get_video
from subtitles import add_audio_and_subtitles
from speed_up import speed_up

def parse_arguments():
    parser = argparse.ArgumentParser(description="Video Processing with Matting and Chroma Key Replacement")

    parser.add_argument('--original_video', type=str, required=True, help="Path to the original video file")
    parser.add_argument('--full_background', type=str, required=True, help="Path to the full background video file")
    parser.add_argument('--phone_background', type=str, help="Path to the phone background video file")
    parser.add_argument('--model_path', type=str, default='rvm_resnet50.pth', help="Path to the model file")
    parser.add_argument('--zoom_scale', type=float, default=1.0, help="Initial zoom scale for chroma key effect")
    parser.add_argument('--zoom_increment', type=float, default=0.2, help="Zoom increment during the effect")
    parser.add_argument('--output_video_name', type=str, default='output_video', help="Title of the final video without extension")
    parser.add_argument('--lower_blue', type=int, nargs=3, default=[80, 50, 80], help="Lower HSV range for blue")
    parser.add_argument('--upper_blue', type=int, nargs=3, default=[130, 255, 255], help="Upper HSV range for blue")
    parser.add_argument('--lower_green', type=int, nargs=3, default=[35, 40, 40], help="Lower HSV range for green")
    parser.add_argument('--upper_green', type=int, nargs=3, default=[85, 255, 255], help="Upper HSV range for green")
    parser.add_argument('--robust_output_type', type=str, default='png', help="Video or png (default: png)")
    parser.add_argument('--robust_model', type=str, default='resnet50', help="Robust model. resnet50 or mobilenetv3")
    parser.add_argument('--processing_model', type=str, default='cpu', help='Processing model: cpu or gpu')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--start_phone_video', action='store_true', help="Start the clip immediately after the phone screen appears in the frame. Default: false.")
    parser.add_argument('--no_resize', type=str, default='',
                        help="Comma-separated list of videos not to resize: 'phone', 'background' or both.")
    parser.add_argument('--music_path', type=str, help="Path to the music file")

    return parser.parse_args()


def main():
    args = parse_arguments()
    editor_config = EditorConfig(args)

    if not os.path.exists(editor_config.replace_output_video_folder_path):
        print('1. Replace colors in file')
        os.makedirs(editor_config.replace_output_video_folder_path)
        replace(editor_config)
    else:
        print(f"Founded color replaced file {editor_config.replace_output_video_folder_path}, skipped...")

    # Проверка наличия директории композиций
    if not os.path.exists(editor_config.output_composition_path):
        print('2. Removing background from video')
        os.makedirs(editor_config.output_composition_path)

        # Загрузка модели матирования
        if editor_config.processing_model == 'cpu':
            model = MattingNetwork(args.robust_model).eval().cpu()
        elif editor_config.processing_model == 'gpu':
            model = MattingNetwork(args.robust_model).eval().cuda()
        else:
            raise TypeError('Unsupported processing model')

        model.load_state_dict(torch.load(args.model_path))

        if editor_config.robust_output_type == "png":
            convert_video(
                model=model,
                input_source=editor_config.replace_output_video_path,
                output_type='png_sequence',
                output_composition=editor_config.output_composition_path,
                output_video_mbps=4,
                downsample_ratio=None,
                seq_chunk=24
            )
        elif editor_config.robust_output_type == "video":
            convert_video(
                model=model,
                input_source=editor_config.replace_output_video_path,
                output_type='video',
                output_composition=f'{editor_config.output_composition_path}/com.mp4',
                output_alpha=f"{editor_config.output_composition_path}/pha.mp4",
                output_foreground=f"{editor_config.output_composition_path}/fgr.mp4",
                output_video_mbps=4,
                downsample_ratio=None,
                seq_chunk=24
            )
    else:
        print(f'Founded removed background video {editor_config.output_composition_path}, skipped...')

    print('3. CHROMA KEY REPLACING')
    fps, phone_start_second, back_video_start_second = chroma_replace(editor_config)

    data = [
        {'template_path': './src/share/big-share-white.png', 'resize': {'min': 80, 'max': 150}, 'threshold': 0.7},
        {'template_path': './src/link/tiktok_link.png', 'resize': {'min': 180, 'max': 350}, 'threshold': 0, 'background_hex_color': '#2764FB', 'template_skip_frames': 47}
    ]
    output_path = os.path.join(editor_config.main_folder_path, editor_config.output_video_name)
    share_start_second, link_start_second = get_video(f'{output_path}.mp4', f'{output_path}_with_elements.mp4', data, fps)

    print('\nThe second the phone appears: ', phone_start_second, '\nOne second of the beginning of the clip: ', back_video_start_second, '\nA second of the emergence of the share template:', share_start_second, '\nThe second the link appears:', link_start_second)

    subtitles_data = [
        {"start_second": 3, "subtitle_text": "Listen to this track on your favorite platform in just two clicks!"},
    ]

    with_elements_video = f'{output_path}_with_elements.mp4'
    subtitles_video = f'{output_path}.mp4'
    speed_up_video = f'{output_path}_speed_up.mp4'

    print('4. ADDING AUDIO AND SUBTITLES')
    add_audio_and_subtitles(
        input_video_path=with_elements_video,
        output_video_path=subtitles_video,
        main_audio_path=editor_config.music_path,
        subtitles_data=subtitles_data,
        language='en',
        clean_temp=True
    )

    print('5. SPEEDING UP THE VIDEO')
    speed_up(
        output_video=speed_up_video,
        input_video=subtitles_video,
        start_time=15,
        speed_factor=3.2,
        audio_speed=False,
        end_image='./_lab/audio_for_videos/end2.png',
        end_time=3,
        end_animated=True,
        end_animated_time=2
    )

if __name__ == "__main__":
    main()
