# StillMovieGen

A small Python tool to generate a "still movie" from a static image. Inspired by zyzsdy/static-movie-generator

It also supports generating single image with cover style ~~due to poor I/O performance~~.

Dependencies

- Python 3.8+
- opencv-python
- numpy (always installed with OpenCV-Python)

Quick install

```bash
pip install opencv-python
```

Usage examples

Generate a cover-mode Y4M from `cover.jpg` (5 seconds at 30 fps), with resolution 1280x720, and pipe to an encoder (example):

```bash
python main.py -i cover.jpg -d 5000 -f 30 -c -w 1280 -t 720 -o - | QSVEncC64.exe --y4m -i - -o output.mp4 --la 2400
```

Generate a Y4M file (Not recommended, as y4m file can be very large):

```bash
python main.py -i cover.jpg -d 5000 -f 30 -w 1280 -t 720 -o output.y4m
```

Export the static cover frame as an image (time-saving compared to generating video, and also easy for preview):

```bash
python main.py -i cover.jpg -c --format img -w 1280 -t 720 -o cover_frame.png
```
