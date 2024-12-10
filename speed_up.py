import subprocess


def get_video_duration(input_video):
    """Get the duration of the video using ffprobe."""
    probe_command = [
        'ffprobe', '-i', input_video,
        '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0'
    ]
    try:
        duration = float(subprocess.check_output(probe_command).decode().strip())
        return duration
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0


def speed_up(output_video, input_video, start_time, speed_factor, audio_speed=False,
             end_image=None, end_time=None, end_animated=False, end_animated_time=2):

    # Calculate the inverse speed factor for the setpts filter
    pts_multiplier = 1 / speed_factor

    # Get the original video duration
    original_duration = get_video_duration(input_video)

    # Calculate the new duration for the accelerated part
    normal_duration = start_time
    accelerated_duration = (original_duration - start_time) / speed_factor
    new_duration = normal_duration + accelerated_duration

    # Calculate the duration for the image insertion if end_image is provided
    if end_image:
        if end_time is None:
            end_time = max(0, original_duration - new_duration)
        total_end_time = end_animated_time + end_time  # Include animation time

    # Build the video filter
    video_filter = (
        f"[0:v]trim=0:{start_time},setpts=PTS-STARTPTS[v1];"
        f"[0:v]trim={start_time},setpts=PTS-STARTPTS,setsar=1/1[v2];"
        f"[v2]setpts={pts_multiplier}*PTS[v2fast];"
    )

    # Build the audio filter depending on audio_speed
    if audio_speed:
        audio_filter = (
            f"[0:a]atrim=0:{start_time},asetpts=PTS-STARTPTS[a1];"
            f"[0:a]atrim={start_time},asetpts=PTS-STARTPTS[a2];"
            f"[a2]atempo={speed_factor}[a2fast];"
            f"[a1][a2fast]concat=n=2:v=0:a=1[outa];"
        )
    else:
        audio_filter = (
            f"[0:a]atrim=0:{start_time},asetpts=PTS-STARTPTS[a1];"
            f"[0:a]atrim={start_time},asetpts=PTS-STARTPTS[a2];"
            f"[a1][a2]concat=n=2:v=0:a=1[outa];"
        )

    # Add filter for inserting an image if `end_image` is provided
    if end_image:
        fps = 25  # Assumed frame rate
        loop_count = int(fps * total_end_time)  # Number of repetitions

        if end_animated:
            # Smooth appearance of the image
            video_filter += (
                f"[1:v]loop=loop={loop_count}:size=1,trim=duration={total_end_time},"
                f"setpts=PTS-STARTPTS,fade=t=in:st=0:d={end_animated_time}[endimg];"
            )
        else:
            # Abrupt insertion of the image
            video_filter += (
                f"[1:v]loop=loop={loop_count}:size=1,trim=duration={total_end_time},setpts=PTS-STARTPTS[endimg];"
            )

        video_filter += f"[v1][v2fast][endimg]concat=n=3:v=1:a=0[outv];"
    else:
        video_filter += "[v1][v2fast]concat=n=2:v=1:a=0[outv];"

    # Build the complete filter
    filter_complex = video_filter + audio_filter

    # Command for FFmpeg
    command = [
        "ffmpeg",
        "-y",
        "-i", input_video
    ]

    # If an image is provided, add it to the command
    if end_image:
        command += ["-i", end_image]

    command += [
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-shortest",
        output_video
    ]

    # Run the process
    subprocess.run(command)


if __name__ == "__main__":
    speed_up(
        output_video='test.mp4',
        input_video='fourth_var_new.mp4',
        start_time=15,
        speed_factor=3,
        audio_speed=False,
        end_image='end2.png',
        end_time=5,
        end_animated=True,
        end_animated_time=2
    )
