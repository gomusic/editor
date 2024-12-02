import torch
import soundfile as sf
import whisperx
import subprocess
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
import os


def add_audio_and_subtitles(input_video_path, output_video_path, main_audio_path, subtitles_data, clean_temp=False):
    """
    Adds voiceover and synchronized subtitles for each object in subtitles_data.

    :param input_video_path: Path to the input video.
    :param output_video_path: Path to save the output video.
    :param main_audio_path: Path to the main audio track.
    :param subtitles_data: List of objects {start_second: int, subtitle_text: str}.
    :param clean_temp: Whether to delete temporary files.
    """
    print('Processing voiceovers...')
    input_video_path = os.path.abspath(input_video_path)
    output_video_path = os.path.abspath(output_video_path)
    main_audio_path = os.path.abspath(main_audio_path)
    temp_audio_files = []

    language = 'ru'
    model_id = 'v3_1_ru'
    sample_rate = 48000
    speaker = 'kseniya'
    put_accent = True
    put_yo = True
    device = torch.device('cpu')

    # Load the Silero TTS model
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language=language,
                              speaker=model_id)
    model.to(device)

    # Create temporary files for generated voices
    for i, subtitle in enumerate(subtitles_data):
        text = subtitle['subtitle_text']
        start_time = subtitle['start_second']
        subtitle_voice_path = os.path.splitext(output_video_path)[0] + f'_subtitle_voice_{i}.mp3'
        temp_audio_files.append(subtitle_voice_path)

        # Generate voice
        audio = model.apply_tts(text=text,
                                speaker=speaker,
                                sample_rate=sample_rate,
                                put_accent=put_accent,
                                put_yo=put_yo)
        file_save(subtitle_voice_path, audio, sample_rate)

    # Create subtitles synchronized with the generated voices
    subtitle_video_path = os.path.splitext(output_video_path)[0] + '_subtitles.mp4'
    create_synchronized_subtitles(input_video_path, subtitle_video_path, subtitles_data, temp_audio_files)

    # Overlay the audio
    add_multiple_audio(subtitle_video_path, temp_audio_files, main_audio_path, subtitles_data)

    if clean_temp:
        cleanup_temp_files(temp_audio_files + [subtitle_video_path])


def file_save(filename, audio, sample_rate):
    sf.write(os.path.abspath(filename), audio, sample_rate)


def cleanup_temp_files(temp_files):
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Temporary file deleted: {file_path}")
        except Exception as e:
            print(f"Error while deleting {file_path}: {e}")


def create_synchronized_subtitles(input_video_path, output_video_path, subtitles_data, audio_files):
    """
    Creates synchronized subtitles based on voiceover timing.
    """
    print('Processing subtitles...')
    output_path = os.path.splitext(output_video_path)[0] + '.mp4'
    font_path = os.path.abspath('./arial.ttf')  # Specify the path to the font

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    font = ImageFont.truetype(font_path, 120)

    # WhisperX parameters
    batch_size = 32
    compute_type = "float32"
    device = "cpu"

    subtitel_model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    word_segments = []
    for subtitle, audio_path in zip(subtitles_data, audio_files):
        start_time_offset = subtitle["start_second"]
        audio = whisperx.load_audio(audio_path)
        result = subtitel_model.transcribe(audio, batch_size=batch_size, language="ru")

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=True)

        for word in result["word_segments"]:
            word["start"] += start_time_offset
            word["end"] += start_time_offset
            word_segments.append(word)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Find words to display at the current moment
        for word in word_segments:
            if word["start"] <= current_time <= word["end"]:
                text = word["word"]
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)

                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = (frame_pil.width - text_width) // 2
                text_y = int(frame_pil.height * 0.8) - (text_height // 2)

                draw.text(
                    (text_x, text_y),
                    text,
                    font=font,
                    fill="white",
                    stroke_width=25,
                    stroke_fill="#5a3be2"
                )
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                break

        out.write(frame)

    cap.release()
    out.release()


def add_multiple_audio(input_video, voice_files, main_audio, subtitles_data):
    print('Create video...')
    fade_volume = 0.07
    output_video = re.sub(r'_subtitles', '', input_video)

    # Define filter parameters
    filter_complex_parts = []
    amix_inputs = []
    index_offset = 2

    # Generate filters for reducing the main audio volume
    volume_modifiers = []
    for i, (voice_file, subtitle_data) in enumerate(zip(voice_files, subtitles_data)):
        start_time = subtitle_data["start_second"]

        # Determine the duration of the current voice file
        probe_command = [
            'ffprobe', '-i', voice_file,
            '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0'
        ]
        voiceover_duration = float(subprocess.check_output(probe_command).decode().strip())
        end_time = start_time + voiceover_duration

        # Add a condition to reduce volume
        volume_modifiers.append(
            f"volume=enable='between(t,{start_time},{end_time})':volume={fade_volume}"
        )

        # Generate filter for each voice file
        filter_complex_parts.append(
            f"[{index_offset + i}:a]adelay={int(start_time * 1000)}|{int(start_time * 1000)}[voiceover{i}]"
        )
        amix_inputs.append(f"[voiceover{i}]")

    # Add the main audio with reduced volume
    main_volume_filter = f"[1:a]{','.join(volume_modifiers)}[main_modified]"
    filter_complex_parts.append(main_volume_filter)

    # Mix all audio tracks
    filter_complex_parts.append(
        f"{''.join(amix_inputs)}[main_modified]amix=inputs={len(amix_inputs) + 1}:duration=longest[audio_out]"
    )

    # Construct the ffmpeg command
    command = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-i", main_audio
    ]
    # Add all voice files
    for voice_file in voice_files:
        command.extend(["-i", voice_file])

    # Add the filter
    command.extend([
        "-filter_complex", ";".join(filter_complex_parts),
        "-map", "0:v",
        "-map", "[audio_out]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_video
    ])

    # Execute the command
    subprocess.run(command)

if __name__ == ('__main__'):
    subtitles_data = [
        {"start_second": 1, "subtitle_text": "Первый субтитр"},
        {"start_second": 5, "subtitle_text": "Второй субтитр"},
    ]
    add_audio_and_subtitles(input_video_path='../../headphones.mp4', output_video_path='D:/test_video', main_audio_path='main_music.mp3', subtitles_data=subtitles_data)