# MosaicGenerator

Create stunning photomosaic images by reconstructing a target image using frames from your videos.

## What is it?

MosaicGenerator transforms your videos into a library of small image tiles, then uses those tiles to recreate any target image as a photomosaic. Each "pixel" in the output is actually a tiny frame from your video, carefully selected to match the color of the original image at that position.

**Example:** You could recreate the Mona Lisa using frames from your vacation videos, or turn your portrait into a mosaic made from your favorite movie scenes!

## How It Works

MosaicGenerator works in three simple steps:

1. **Extract** - Pull frames from video files, center-crop them to squares, and save as a tile library
2. **Analyze** - Calculate the average color (BGR values) of each extracted frame
3. **Generate** - Match each pixel of your target image to the closest-colored frame and assemble the mosaic

## Installation

### Requirements

- Python 3.8 or higher
- pip or uv for package installation

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/MosaicGenerator.git
cd MosaicGenerator
```

2. Install dependencies:
```bash
# Or using pip
pip install -r requirements.txt
```

Dependencies:
- opencv-python (video/image processing)
- tqdm (progress bars)
- pandas (data manipulation)
- numpy (array operations)

## Quick Start

### 1. Prepare Your Videos

Create a `source` directory and add your video files:

```bash
mkdir source
# Copy your .mp4, .avi, or .mov files into the source/ directory
```

### 2. Extract Frames

Extract frames from your videos. The `--skip-frames` parameter lets you control density (higher = fewer frames, faster processing):

```bash
# Extract every 30th frame as 256x256 tiles
python mosaic_generator.py extract --skip-frames 30
```

This creates a `frames/` directory with numbered JPG files.

### 3. Generate Your Mosaic

Point the tool at any target image:

```bash
python mosaic_generator.py generate path/to/target.jpg
```

Your mosaic will be saved as `output.png`!

## Usage

### Command-Line Interface

MosaicGenerator provides three commands:

#### Extract Frames

```bash
python mosaic_generator.py extract [OPTIONS]
```

Options:
- `--skip-frames N` - Extract every Nth frame (default: 1)
  - Use higher values (30-60) for faster processing and smaller libraries
  - Use lower values (1-10) for more variety in your tile library
- `--image-size SIZE` - Size of output square tiles in pixels (default: 256)
- `--source-dir DIR` - Directory with source videos (default: ./source)

Examples:
```bash
# Extract every 60th frame as 128x128 tiles
python mosaic_generator.py extract --skip-frames 60 --image-size 128

# Use a different source directory
python mosaic_generator.py extract --source-dir ~/Videos/MyClips
```

#### Analyze Frames

```bash
python mosaic_generator.py analyze [OPTIONS]
```

Analyzes all extracted frames and computes their average colors. Useful for pre-computing color data to speed up multiple mosaic generations.

Options:
- `--output FILE` - Save analysis to CSV file (default: ./frame_colors.csv)

Example:
```bash
python mosaic_generator.py analyze --output colors.csv
```

#### Generate Mosaic

```bash
python mosaic_generator.py generate TARGET [OPTIONS]
```

Arguments:
- `TARGET` - Path to the target image to recreate

Options:
- `--single-use` - Each frame can only be used once (creates more varied mosaics)
- `--output FILE` - Output filename (default: ./output.png)
- `--colors FILE` - Use pre-computed color data from CSV

Examples:
```bash
# Basic generation
python mosaic_generator.py generate photo.jpg

# Single-use frames for more variety
python mosaic_generator.py generate portrait.jpg --single-use

# Custom output name with pre-computed colors
python mosaic_generator.py generate sunset.jpg --colors colors.csv --output sunset_mosaic.png
```

### Programmatic Usage

You can also import and use MosaicGenerator functions in your Python code:

```python
from mosaic_generator import extract_frames, process_images, generate_image

# Step 1: Extract frames from videos
extract_frames(skip_frames=30, image_size=256, source_dir='./my_videos')

# Step 2: Analyze frame colors
frame_colors = process_images()

# Step 3: Generate mosaic
output = generate_image(
    target_filename='my_photo.jpg',
    df=frame_colors,
    single_use=False,
    output_filename='my_mosaic.png'
)
```

## Example Workflow

Here's a complete example of creating a mosaic:

```bash
# 1. Set up your video library
mkdir source
cp ~/Videos/vacation2024.mp4 source/
cp ~/Videos/birthday_party.mov source/

# 2. Extract frames (every 30th frame, takes a few minutes)
python mosaic_generator.py extract --skip-frames 30

# Example output:
# Extracting frames from vacation2024.mp4...
# Extracting frames from birthday_party.mov...
# Extraction complete! Saved 1247 frames to ./frames/

# 3. Generate your mosaic
python mosaic_generator.py generate portrait.jpg --output portrait_mosaic.png

# Example output:
# Analyzing 1247 frames...
# Analysis complete! Processed 1247 frames.
# Target image size: 400x300
# Using 1247 frames for mosaic generation
# Output will contain 120000 frame tiles
# Building mosaic...
# Saving mosaic to portrait_mosaic.png... Done!
# Output size: 102400x76800 pixels
```

**Note:** The output image will be much larger than your input! If your target is 400x300 pixels and each frame is 256x256, the output will be 102,400x76,800 pixels. Consider using a small target image or reducing `--image-size` during extraction.

## Tips & Best Practices

### For Best Results

1. **Frame Library Size**: Extract 500-2000 frames for good variety
   - More frames = better color matching = better-looking mosaics
   - Fewer frames = faster generation + smaller disk usage

2. **Skip Frames Wisely**:
   - Videos at 30 fps → `--skip-frames 30` gives you 1 frame/second
   - 5-minute video at 30 fps with skip 30 = 300 frames

3. **Target Image Size**: Keep your target small (100x100 to 500x500 pixels)
   - Larger targets = exponentially larger outputs
   - Remember: 100x100 target = 10,000 frame tiles in output

4. **Single-Use Mode**: Use `--single-use` for artistic variety
   - Prevents color banding from frame reuse
   - Requires more frames in your library than pixels in target

5. **Video Selection**: Use colorful, varied videos
   - Diverse scenes = diverse colors = better matching
   - Monochrome videos = limited color palette

### Output Size Calculator

```
Output Width = Target Width × Frame Tile Size
Output Height = Target Height × Frame Tile Size

Example: 200×150 target with 256px tiles = 51,200×38,400 px output (≈2GB)
```

## Troubleshooting

### "source directory does not exist"
Create the `source/` directory and add video files:
```bash
mkdir source
```

### "frames directory not found"
Run the extract command first:
```bash
python mosaic_generator.py extract --skip-frames 30
```

### "Output image is too large"
Your target image is too big. Try:
1. Resize your target image to something smaller (100x100 to 300x300)
2. Use a smaller `--image-size` during extraction (e.g., 128 or 64)

### "Not enough frames for single-use mode"
You need at least as many frames as pixels in your target. Either:
1. Extract more frames (lower `--skip-frames` value)
2. Add more source videos
3. Don't use `--single-use` flag

### "Could not open video file"
Make sure your video files are valid formats (MP4, AVI, MOV) and not corrupted. Try opening them in a video player first.

## Technical Details

- **Color Matching**: Uses Manhattan distance (sum of absolute BGR differences)
- **Cropping**: Center-crops videos to square before resizing
- **Caching**: Frames are cached in memory during generation to avoid repeated disk I/O
- **Format**: Frames saved as JPEG, output as PNG

## Project Structure

```
MosaicGenerator/
├── mosaic_generator.py   # Main script with all functionality
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── LICENSE              # MIT License
├── .gitignore           # Git ignore rules
├── source/              # Place your video files here (you create this)
├── frames/              # Extracted frames (auto-created)
└── output.png           # Generated mosaic (auto-created)
```

## License

MIT License - Copyright (c) 2025

See [LICENSE](LICENSE) file for details.

## Contributing

This is a personal project, but suggestions and improvements are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share your awesome mosaic creations

## Credits

Created by Adriaan Engelbrecht

---

**Have fun creating mosaics!** Share your creations and tag them with #MosaicGenerator
