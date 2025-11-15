import cv2
import os

def extract_key_frames(video_path, output_dir, frames_to_extract=5):
    """
    Extracts a fixed number of key frames from a video and saves them as images.
    
    Args:
        video_path (str): The path to the input video file.
        output_dir (str): The directory where the extracted frames will be saved.
        frames_to_extract (int): The number of frames to extract (evenly spaced).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- Frame Index Calculation ---
    # Calculate which frames to extract (evenly spaced).
    # We divide the total frames by the number of extractions + 1 
    # to find the step size, ensuring the frames are spread out.
    step = frame_count // (frames_to_extract + 1)
    
    # Frame indices are calculated starting from 1 * step up to frames_to_extract * step
    frame_indices = [step * i for i in range(1, frames_to_extract + 1)]

    print(f"Video has {frame_count} total frames.")
    print(f"Frames to be extracted: {frame_indices}")

    # Extract and save frames
    for i, frame_index in enumerate(frame_indices):
        # Set the video cursor to the desired frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if ret:
            # Save the frame as a PNG file
            frame_filename = os.path.join(output_dir, f"frame_{i+1:02d}.png")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {i+1} at index {frame_index} to {frame_filename}")
        else:
            print(f"Warning: Could not read frame at index {frame_index}")

    cap.release()
    print("\n✅ Frame extraction complete.")

if __name__ == "__main__":
    # IMPORTANT: Replace the video path with the actual location of your video file.
    video_file = "" 
    output_directory = "extracted_frames_PnPCabToCounter"
    
    # You can change the number of frames you want to extract
    extract_key_frames(video_file, output_directory, frames_to_extract=5)