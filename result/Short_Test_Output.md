# Potential Errors When Running USR 2.0

This file records errors encountered or likely to occur when running `demo.py` on Windows, along with their causes and suggested fixes.

## 1. Python version is not installed

**Message:** `No suitable Python runtime found` when running `py -3.10`.

**Cause:** Python 3.10 is not installed. Check available versions with `py --list` and create the virtual environment with an installed version, for example:

```powershell
py -3.12 -m venv .venv
```

## 2. A Python package is missing

**Messages:** `ModuleNotFoundError: No module named 'cv2'` or `No module named 'torchaudio'`.

**Cause:** The package is missing from the Python environment used to run the script. Install packages in the project virtual environment, or invoke its Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install opencv-python
.\.venv\Scripts\python.exe -m pip install torch torchaudio torchvision
```

The `(.venv)` prompt alone does not guarantee that `python` resolves to the virtual environment. Check with `python -c "import sys; print(sys.executable)"`.

## 3. Unicode decoding error on Windows

**Message:** `UnicodeDecodeError: 'charmap' codec can't decode byte ...` while loading `utils/labels/unigram1000_units.txt`.

**Cause:** Python is using the Windows `cp1252` default encoding to read a UTF-8 label file.

**Fix:** Enable UTF-8 for the current PowerShell session and rerun the script:

```powershell
$env:PYTHONUTF8 = "1"
```

## 4. PyTorch and TorchVision are incompatible

**Message:** `AttributeError: module 'torchvision.io' has no attribute 'read_video'`.

**Cause:** The installed TorchVision version does not provide the video-reading API used by this project, or it does not match the installed PyTorch version.

**Fix:** Install a mutually compatible set. One tested version combination for the project is PyTorch 2.5.1, TorchVision 0.20.1, and TorchAudio 2.5.1. Choose the CPU or CUDA wheel index that matches the runtime.

## 5. PyAV version is incompatible with TorchVision

**Messages:** `open() got an unexpected keyword argument 'metadata_errors'` or `module 'av' has no attribute 'AVError'`.

**Cause:** The installed PyAV (`av`) API differs from what the installed TorchVision video reader expects.

**Fix:** Use a compatible PyAV release, for example:

```powershell
python -m pip install --force-reinstall "av==13.1.0"
```

## 6. Video decoding runs out of memory

**Message:** `av.error.MemoryError: Cannot allocate memory`.

**Cause:** A long or high-resolution video is decoded into memory. The 1920x1080 IELTS sample is about 17.5 minutes long and may exceed available RAM.

**Fix:** Trim the video to a short segment and reduce its resolution before running inference. For example, with FFmpeg:

```powershell
ffmpeg -ss 00:00:00 -i "video\IELTS_Speaking_Test.mp4" -t 30 -vf "scale=640:-2" -c:v libx264 -preset veryfast -crf 23 -c:a aac "video\IELTS_test_30s.mp4"
```

## 7. Face or mouth landmarks cannot be detected

**Message:** `Could not detect a face in enough frames. Make sure the video contains a clearly visible face.`

**Cause:** The face is small, turned away, blurred, poorly lit, occluded, or missing from too many frames.

**Fix:** Use a segment where the speaker's face and mouth are clearly visible for most of the clip. Try another segment if needed.

## 8. Video or checkpoint path is wrong

**Likely messages:** `FileNotFoundError` or an error while opening/loading the video or checkpoint.

**Cause:** The specified video or `.pth` checkpoint does not exist at that path, or the PowerShell command is using a different working directory.

**Fix:** Confirm both files exist and use quoted absolute paths:

```powershell
Test-Path "D:\HK252\NCKH\usr2-main\video\Short_Test.mp4"
Test-Path "D:\HK252\NCKH\usr2-main\path\lrs3vox2_baseplus.pth"
```

## 9. Checkpoint and model configuration do not match

**Likely messages:** Missing or unexpected state-dictionary keys, tensor size mismatch, or a model-loading error.

**Cause:** The checkpoint was trained for a different model size or architecture than the selected configuration.

**Fix:** Use the matching backbone configuration for the checkpoint (Base, Base+, Large, or Huge) and verify that the checkpoint file downloaded completely.

## 10. No NVIDIA GPU is available

**Symptom:** PyTorch reports CUDA is unavailable, or the computer only lists Intel/AMD graphics.

**Cause:** Local inference is running on CPU. This is not necessarily an error, but processing can be slow and long videos can use substantial memory.

**Fix:** Use shorter, lower-resolution clips, or run the project in a Colab runtime with an available NVIDIA GPU. A GPU is not required for CPU inference.

## Current run command

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe demo.py video="D:\HK252\NCKH\usr2-main\video\Short_Test.mp4" model.pretrained_model_path="D:\HK252\NCKH\usr2-main\path\lrs3vox2_baseplus.pth"
```
