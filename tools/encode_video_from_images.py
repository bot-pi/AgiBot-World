# given episode directories, encode videos from image frames
import os 
import argparse
import subprocess
import logging
import glob
import re

def encode_videos_from_images_by_frame_index(episode_dir, output_dir, camera_name="hand_right_color", framerate=30):
    """
    Encode images to video by their frame index, handling variable digit frame numbers.
    
    Args:
        episode_dir: Path to episode directory containing camera/{frame}/hand_right_color.jpg
        output_dir: Directory to save the output video
        camera_name: Name of the camera image file (default: "hand_right_color")
        framerate: Video framerate (default: 30)
    """
    camera_dir = os.path.join(episode_dir, 'camera')
    
    if not os.path.exists(camera_dir):
        logging.error(f"Camera directory not found: {camera_dir}")
        return False
    
    # Find all frame directories and extract frame numbers
    frame_pattern = os.path.join(camera_dir, "*", f"{camera_name}.jpg")
    image_files = glob.glob(frame_pattern)
    print(f"Found {len(image_files)} images matching pattern: {frame_pattern}")
    
    if not image_files:
        logging.error(f"No images found matching pattern: {frame_pattern}")
        return False
    
    # Extract frame numbers and sort them
    frame_info = []
    for img_path in image_files:
        frame_dir = os.path.dirname(img_path)
        frame_num_str = os.path.basename(frame_dir)
        try:
            frame_num = int(frame_num_str)
            frame_info.append((frame_num, img_path))
        except ValueError:
            logging.warning(f"Skipping non-numeric frame directory: {frame_num_str}")
    
    if not frame_info:
        logging.error(f"No valid frame numbers found in {camera_dir}")
        return False
    
    # Sort by frame number
    frame_info.sort(key=lambda x: x[0])
    print(frame_info[:10])
    
    # Determine file extension from the first image
    first_img_path = frame_info[0][1]
    file_extension = os.path.splitext(first_img_path)[1]  # .jpg or .png
    print(f"Detected file extension: {file_extension}")
    
    # Create a temporary directory with symlinks for ffmpeg
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create symlinks with sequential naming for ffmpeg
        for i, (frame_num, img_path) in enumerate(frame_info):
            # Use absolute path for symlink target
            absolute_img_path = os.path.abspath(img_path)
            symlink_path = os.path.join(temp_dir, f"frame_{i:06d}{file_extension}")
            os.symlink(absolute_img_path, symlink_path)
            if i < 5:  # Only print first few for debugging
                print(f"Created symlink: {symlink_path} -> {absolute_img_path}")
                print(f"  Symlink exists: {os.path.exists(symlink_path)}")
        
        # List all files in temp directory for debugging
        temp_files = os.listdir(temp_dir)
        print(f"Files in temp directory: {sorted(temp_files)[:10]}")  # Show first 10
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct ffmpeg command
        output_path = os.path.join(output_dir, f"{os.path.basename(episode_dir)}.mp4")
        ffmpeg_pattern = os.path.join(temp_dir, f"frame_%06d{file_extension}")
        print(f"FFmpeg input pattern: {ffmpeg_pattern}")
        
        command = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-framerate", str(framerate),
            "-start_number", "0",  # Start from frame 0
            "-i", ffmpeg_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",  # Constant rate factor for good quality
            output_path
        ]
        
        logging.info(f"Encoding video for {episode_dir} with {len(frame_info)} frames...")
        logging.info(f"Frame range: {frame_info[0][0]} to {frame_info[-1][0]}")
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"Video saved to {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg error: {e.stderr}")
            return False

def encode_videos_from_images(episode_dirs, output_dir):
    """
    Legacy function that uses the new flexible encoding approach.
    """
    success_count = 0
    for episode_dir in episode_dirs:
        if encode_videos_from_images_by_frame_index(episode_dir, output_dir):
            success_count += 1
        else:
            logging.error(f"Failed to encode video for {episode_dir}")
    
    logging.info(f"Successfully encoded {success_count}/{len(episode_dirs)} videos")

episodes = [
    "dataset/Manipulation-SimData/make_a_sandwich/2810183/3361360/A2D0015AB00061/12088405",
]

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Encode videos from image frames')
    parser.add_argument('--episode-dirs', nargs='+', default=episodes,
                        help='Episode directories containing camera frames')
    parser.add_argument('--output-dir', default='output_videos',
                        help='Output directory for videos')
    parser.add_argument('--framerate', type=int, default=30,
                        help='Video framerate (default: 30)')
    parser.add_argument('--camera-name', default='head_color',
                        help='Camera image filename without extension (default: head_color)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if len(args.episode_dirs) == 1:
        # Single episode, use the flexible function directly
        success = encode_videos_from_images_by_frame_index(
            args.episode_dirs[0], 
            args.output_dir, 
            args.camera_name, 
            args.framerate
        )
        if success:
            logging.info("Video encoding completed successfully!")
        else:
            logging.error("Video encoding failed!")
    else:
        # Multiple episodes, use batch function
        encode_videos_from_images(args.episode_dirs, args.output_dir)

if __name__ == "__main__":
    main()
else:
    # For backward compatibility when imported
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    encode_videos_from_images(episodes, "output_videos")