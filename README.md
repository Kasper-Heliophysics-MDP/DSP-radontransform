# DSP-radontransform

# Radon Transform Burst Detection

A Python tool for detecting and analyzing linear burst patterns in spectrogram data using the Radon Transform.

## What This Does

The Radon Transform detects straight lines in an image by measuring how much the image aligns with lines at every possible angle and position. It creates a new image where bright vertical streaks indicate angles where strong linear patterns exist, and the brightness shows how strong those patterns are. This lets us automatically find the orientation and location of bursts in noisy spectrogram data.

## Features

- **Adaptive Thresholding**: Removes background noise by isolating only strong signals
- **Contrast Enhancement**: Makes bursts more visible in the data
- **Multiple Visualizations**: 
  - Preprocessing steps showing data cleaning
  - Four different views of the Radon transform (log-scaled, normalized, high-pass filtered, intensity profile)
  - Detailed burst location analysis with zoomed views
- **Automatic Peak Detection**: Identifies dominant burst orientations
- **1D Profile Analysis**: Shows exact burst positions along the strongest angle

## Requirements

- Python 3.7 or higher
- Required packages (see `requirements.txt`)

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd radon-transform-analysis
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data:
   - Your spectrogram data should be in CSV format
   - The CSV should contain only numeric values (frequency vs time intensity data)
   - Update the `csv_file` variable in the script to point to your data file

2. Run the analysis:
```bash
python3 radon_analysis_improved.py
```

3. Output files generated:
   - `preprocessing_steps.png` - Shows the data cleaning process
   - `radon_analysis_enhanced.png` - Main analysis with 4 visualization panels
   - `burst_location_detail.png` - Detailed view of the strongest burst with 1D profile

## Understanding the Output

### Graph 1: Preprocessing Steps
- **Left**: Original spectrogram
- **Middle**: After thresholding (noise removed)
- **Right**: Enhanced and ready for analysis

### Graph 2: Radon Analysis Enhanced
- **Top-Left**: Log-scaled Radon transform
- **Top-Right**: Normalized view (best for seeing all angles)
- **Bottom-Left**: High-pass filtered (shows burst locations)
- **Bottom-Right**: Intensity vs angle with detected peaks (red stars)

### Graph 3: Burst Location Detail
- **Top-Left**: Full Radon transform with strongest angle marked
- **Top-Right**: Zoomed view around strongest burst
- **Bottom**: 1D intensity profile showing exact burst positions

## Interpreting Angles

- **0° or 180°**: Horizontal bursts (short time duration across frequencies)
- **90°**: Vertical bursts (frequency sweeps over time)
- **45° or 135°**: Diagonal bursts (frequency changing linearly with time)

## Customization

You can adjust these parameters in the code:

- `threshold = np.percentile(spectrogram, 75)` - Change 75 to adjust noise filtering (higher = more aggressive)
- `downsample_factor = 4` - Change for speed vs accuracy tradeoff
- `window = 15` - Adjust zoom window size in burst detail view
- `peak_threshold = 0.3 * np.max(...)` - Adjust peak detection sensitivity

## Example

```python
# Update the input and output directory path
input_dir = "./data"
output_dir = "./output"

# Run the script
python3 radon_analysis_improved.py
```

The console output will show:
- Detected burst orientations with interpretations
- Number of distinct bursts found
- Processing progress

## Troubleshooting

**Issue**: Script runs slowly
- **Solution**: Increase `downsample_factor` (e.g., from 4 to 8)

**Issue**: No bursts detected
- **Solution**: Lower the threshold percentile (e.g., from 75 to 60)

**Issue**: Too many false peaks
- **Solution**: Increase `peak_threshold` or `distance` parameter in `find_peaks()`

## License

MIT License - feel free to use and modify for your projects.

## Citation

If you use this code in your research, please cite:
```
[Your Name/Organization], Radon Transform Burst Detection Tool, [Year]
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your contact info].
