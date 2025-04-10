# audio_gen
Prototyping with the stable audio open model. Follow these instructions to get the script running on your Mac


## Prerequisites
- Python 3.11 
- `pip` (Python package installer)
- created / tested on MBP M3

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/arm-michael/audio_gen
   cd audio_gen
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install torch
   python -c "import torch; print(torch.__version__)"                                           #verify
   pip install numpy                
   python -c "import numpy; print('numpy installed successfully')"                              #verify
   pip install torchaudio                 
   python -c "import torchaudio; print('torchaudio installed successfully')"                    #verify
   pip install einops
   python -c "import einops; print('einops installed successfully')"                            #verify
   pip install stable_audio_tools
   python -c "import stable_audio_tools; print('stable_audio_tools installed successfully')"    #verify
   ```

4. **Run the Program**:
   ```bash
   python gen.py
   ```

## Usage
- When prompted, enter a text prompt for audio generation.
- Specify the tempo (in beats per minute).
- The generated audio will be saved as a `.wav` file in the current directory.

## Notes
- Ensure your system supports the `mps` device (Apple Silicon) or modify the `device` variable in `gen.py` to `cuda` (for NVIDIA GPUs) or `cpu` if necessary.
- If you encounter issues with dependencies, ensure all packages in `requirements.txt` are installed correctly.

## Example
```bash
Enter a prompt for generating audio:
Ambient music
Enter a tempo for the audio:
120
Generated audio saved to: Ambient music.wav
Would you like to generate more audio? (yes/no)
no
Exiting audio generation.
```
