import cv2
from cv2 import VideoWriter





# def save_video(out_vid_frames, out_video_path, fps):
#     if not out_vid_frames:
#         print("No frames to write to video.")
#         return

#     # Use the .mp4 file extension and H.264 codec
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     height, width, layers = out_vid_frames[0].shape
#     out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

#     if not out.isOpened():
#         print("Error: VideoWriter not opened.")
#         return

#     for frame in out_vid_frames:
#         out.write(frame)
    
#     out.release()  # Ensure to release the writer object
#     print(f"Video saved to {out_video_path}")




    


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the original video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()  # Ensure to release the capture object
    return frames, fps  # Return frames and frame rate

def save_video(out_vid_frames, out_video_path, fps):
    if not out_vid_frames:
        print("No frames to write to video.")
        return

    # Use the original frame size
    height, width, layers = out_vid_frames[0].shape
    frame_size = (width, height)

    # Print debug information
    print(f"Output video path: {out_video_path}")
    print(f"Frame size: {frame_size}")
    print(f"FPS: {fps}")
    print(f"Number of layers (channels): {layers}")

    # Ensure FPS is valid
    if fps <= 0:
        print("Invalid FPS value.")
        return

    # Test with different codecs
    codecs = ['mp4v']
    for codec in codecs:
        codec_output_path = out_video_path.replace(".mp4", f"_{codec}.mp4")
        print(f"Testing with codec: {codec}")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(codec_output_path, fourcc, fps, frame_size)

        # Check if VideoWriter is successfully opened
        if not out.isOpened():
            print(f"Error: VideoWriter not opened with codec {codec}.")
            continue

        for frame in out_vid_frames:
            out.write(frame)

        out.release()  # Ensure to release the writer object
        print(f"Video saved to {codec_output_path} using codec {codec}")
        return

    print("Failed to open VideoWriter with all tested codecs.")





   
