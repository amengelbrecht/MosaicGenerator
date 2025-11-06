#!/usr/bin/env python3
"""
MosaicGenerator - Create photomosaic images from video frames.

This tool extracts frames from videos, analyzes their colors, and uses them
to recreate a target image as a mosaic of video frames.
"""

import argparse
import os
import sys
from typing import Optional, Tuple
import copy

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_frames(
    skip_frames: int = 1,
    image_size: int = 64,
    source_dir: str = './source'
) -> None:
    """
    Extract frames from video files and save them as square images.

    Reads all video files from the source directory, extracts frames with center
    cropping to create square images, and saves them to the frames directory.

    Args:
        skip_frames: Number of frames to skip between extractions (default: 1)
        image_size: Size of output square images in pixels (default: 64)
        source_dir: Directory containing source video files (default: './source')

    Raises:
        FileNotFoundError: If source directory doesn't exist
        ValueError: If image_size or skip_frames are invalid
    """
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if skip_frames < 1:
        raise ValueError(f"skip_frames must be at least 1, got {skip_frames}")

    # Check that source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"{source_dir} does not exist. Please create it and add video files.")

    # Check for video files
    video_files = [f for f in os.listdir(source_dir) if not f.startswith('.')]
    if not video_files:
        print(f"Warning: No files found in {source_dir}")
        return

    # Make frames directory if it doesn't exist
    os.makedirs('./frames', exist_ok=True)

    frame_count = 0

    for file in video_files:
        filename = os.fsdecode(file)
        video_path = os.path.join(source_dir, filename)

        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Warning: Could not open {filename}, skipping...")
            continue

        print(f"Extracting frames from {filename}...")
        success, image = cap.read()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc=filename) as pbar:
            while success:
                # Center crop to square
                height, width = image.shape[:-1]
                crop_dim = int(min(height, width) / 2)
                mid_row = int(height / 2)
                mid_col = int(width / 2)

                cropped = image[
                    mid_row - crop_dim:mid_row + crop_dim,
                    mid_col - crop_dim:mid_col + crop_dim,
                    :
                ]

                # Resize to target size
                resized = cv2.resize(cropped, (image_size, image_size))

                # Save frame
                output_path = f'./frames/{frame_count}.jpg'
                cv2.imwrite(output_path, resized)

                # Skip frames as specified
                for _ in range(skip_frames):
                    pbar.update(1)
                    success, image = cap.read()
                    if not success:
                        break

                frame_count += 1

        cap.release()

    print(f"Extraction complete! Saved {frame_count} frames to ./frames/")


def process_images() -> pd.DataFrame:
    """
    Analyze all extracted frames and compute their average colors.

    Reads all images from the frames directory and calculates the mean
    BGR color values for each image.

    Returns:
        DataFrame with columns: filename, blue, green, red containing
        the filename and average color channels for each frame.

    Raises:
        FileNotFoundError: If frames directory doesn't exist or is empty
    """
    frames_dir = './frames'

    if not os.path.exists(frames_dir):
        raise FileNotFoundError(
            f"{frames_dir} directory not found. Run extract_frames() first."
        )

    frame_files = [f for f in os.listdir(frames_dir) if not f.startswith('.')]

    if not frame_files:
        raise FileNotFoundError(
            f"{frames_dir} is empty. Run extract_frames() first."
        )

    filenames, blue_values, green_values, red_values = [], [], [], []

    print(f"Analyzing {len(frame_files)} frames...")

    for file in tqdm(frame_files, desc="Processing"):
        filename = os.fsdecode(file)
        image_path = os.path.join(frames_dir, filename)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Could not read {filename}, skipping...")
            continue

        filenames.append(filename)
        blue_values.append(image[:, :, 0].mean())
        green_values.append(image[:, :, 1].mean())
        red_values.append(image[:, :, 2].mean())

    df = pd.DataFrame({
        'filename': filenames,
        'blue': blue_values,
        'green': green_values,
        'red': red_values
    })

    print(f"Analysis complete! Processed {len(df)} frames.")
    return df


def generate_image(
    target_filename: str,
    df: Optional[pd.DataFrame] = None,
    single_use: bool = False,
    output_filename: str = './output.png'
) -> np.ndarray:
    """
    Generate a mosaic image by matching target image pixels to similar frames.

    Takes a target image and recreates it as a mosaic where each pixel is
    replaced by the frame with the closest matching average color.

    Args:
        target_filename: Path to the target image to recreate
        df: DataFrame with frame color data (if None, will be generated)
        single_use: If True, each frame can only be used once (default: False)
        output_filename: Path where output image will be saved (default: './output.png')

    Returns:
        The generated mosaic image as a numpy array

    Raises:
        FileNotFoundError: If target image doesn't exist
        ValueError: If DataFrame is invalid or frames directory is missing
    """
    # Validate target image exists
    if not os.path.exists(target_filename):
        raise FileNotFoundError(f"Target image not found: {target_filename}")

    # Generate DataFrame if not provided
    if df is None:
        print("No DataFrame provided. Analyzing frames now...")
        df = process_images()

    # Validate DataFrame
    required_cols = ['filename', 'blue', 'green', 'red']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    if len(df) == 0:
        raise ValueError("DataFrame is empty. No frames to generate mosaic.")

    # Create a working copy for single_use mode
    source_df = copy.deepcopy(df)

    # Load target image
    target_image = cv2.imread(target_filename)
    if target_image is None:
        raise ValueError(f"Could not read target image: {target_filename}")

    height, width = target_image.shape[:2]

    print(f"Target image size: {width}x{height}")
    print(f"Using {len(df)} frames for mosaic generation")
    print(f"Output will contain {width * height} frame tiles")

    # Cache for loaded images to avoid repeated disk I/O
    image_cache = {}

    def get_frame_image(filename: str) -> np.ndarray:
        """Load frame image from cache or disk."""
        if filename not in image_cache:
            frame_path = f'./frames/{filename}'
            img = cv2.imread(frame_path)
            if img is None:
                raise FileNotFoundError(f"Frame not found: {frame_path}")
            image_cache[filename] = img
        return image_cache[filename]

    print("Building mosaic...")
    mosaic_rows = []

    for row in tqdm(range(height), desc="Generating"):
        mosaic_cols = []

        for col in range(width):
            # Get target pixel BGR values
            target_blue = target_image[row, col, 0]
            target_green = target_image[row, col, 1]
            target_red = target_image[row, col, 2]

            # Calculate color distance (Manhattan distance) for all frames
            color_diff = (
                abs(source_df['blue'] - target_blue) +
                abs(source_df['green'] - target_green) +
                abs(source_df['red'] - target_red)
            )

            # Find best matching frame
            best_match_idx = color_diff.idxmin()
            best_match_filename = source_df.iloc[best_match_idx]['filename']

            # Load the matching frame
            frame_image = get_frame_image(best_match_filename)
            mosaic_cols.append(frame_image)

            # Mark frame as used if single_use mode
            if single_use:
                source_df.iloc[best_match_idx] = [
                    '0.jpg',
                    float('inf'),
                    float('inf'),
                    float('inf')
                ]

        # Stack all frames in this row horizontally
        mosaic_rows.append(np.hstack(mosaic_cols))

    # Stack all rows vertically to create final mosaic
    output_image = np.vstack(mosaic_rows)

    # Save output image
    print(f"Saving mosaic to {output_filename}...", end=' ')
    if cv2.imwrite(output_filename, output_image):
        print("Done!")
        print(f"Output size: {output_image.shape[1]}x{output_image.shape[0]} pixels")
    else:
        print("Failed!")
        raise IOError(f"Failed to write output image to {output_filename}")

    return output_image


def main() -> int:
    """Main entry point for CLI interface."""
    parser = argparse.ArgumentParser(
        description='MosaicGenerator - Create photomosaic images from video frames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames from videos (every 30th frame)
  python mosaic_generator.py extract --skip-frames 30

  # Analyze extracted frames
  python mosaic_generator.py analyze

  # Generate mosaic from target image
  python mosaic_generator.py generate target.jpg

  # Generate with single-use frames
  python mosaic_generator.py generate target.jpg --single-use
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Extract command
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract frames from video files'
    )
    extract_parser.add_argument(
        '--skip-frames',
        type=int,
        default=1,
        help='Number of frames to skip between extractions (default: 1)'
    )
    extract_parser.add_argument(
        '--image-size',
        type=int,
        default=64,
        help='Size of output square images in pixels (default: 64)'
    )
    extract_parser.add_argument(
        '--source-dir',
        type=str,
        default='./source',
        help='Directory containing source videos (default: ./source)'
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze extracted frames and compute average colors'
    )
    analyze_parser.add_argument(
        '--output',
        type=str,
        default='./frame_colors.csv',
        help='Save analysis results to CSV file (default: ./frame_colors.csv)'
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate mosaic image from target'
    )
    generate_parser.add_argument(
        'target',
        type=str,
        help='Path to target image'
    )
    generate_parser.add_argument(
        '--single-use',
        action='store_true',
        help='Each frame can only be used once in the mosaic'
    )
    generate_parser.add_argument(
        '--output',
        type=str,
        default='./output.png',
        help='Output mosaic filename (default: ./output.png)'
    )
    generate_parser.add_argument(
        '--colors',
        type=str,
        help='Path to pre-computed colors CSV (optional)'
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == 'extract':
            extract_frames(
                skip_frames=args.skip_frames,
                image_size=args.image_size,
                source_dir=args.source_dir
            )

        elif args.command == 'analyze':
            df = process_images()
            df.to_csv(args.output, index=False)
            print(f"Saved color analysis to {args.output}")

        elif args.command == 'generate':
            # Load pre-computed colors if provided
            df = None
            if args.colors:
                if os.path.exists(args.colors):
                    df = pd.read_csv(args.colors)
                    print(f"Loaded color data from {args.colors}")
                else:
                    print(f"Warning: {args.colors} not found, will analyze frames")

            generate_image(
                target_filename=args.target,
                df=df,
                single_use=args.single_use,
                output_filename=args.output
            )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
