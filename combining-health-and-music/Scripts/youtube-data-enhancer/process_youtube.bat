@echo off
echo Starting YouTube Analysis with GPU acceleration

REM Using fewer processes with GPU to avoid memory issues with GTX 1650 Super (4GB VRAM)

REM python youtube_content_analyzer.py --file youtube-gaurav.csv --processes 8000 --max-runs 5000
REM echo Batch 1 complete (8000-13000)

REM python youtube_content_analyzer.py --file youtube-gaurav.csv --processes 13000 --max-runs 5000
REM echo Batch 2 complete (13000-18000)

python youtube_content_analyzer.py --file youtube-gaurav.csv --processes 2 --resume 18000 --max-runs 5000
echo Batch 3 complete (18000-23000)

python youtube_content_analyzer.py --file youtube-gaurav.csv --processes 2 --resume 23000 --max-runs 5000
echo Batch 4 complete (23000-28000)

python youtube_content_analyzer.py --file youtube-gaurav.csv --processes 2 --resume 28000
echo Final batch complete (28000-end)

echo All processing complete!
