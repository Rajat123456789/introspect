@echo off
echo Starting YouTube Analysis on high-RAM system

REM Process in larger 5000-video batches (good for 32GB RAM)
python youtube_content_analyzer.py --file youtube-gaurav.csv --processes 12 --resume 8000 --max-runs 5000
echo Batch 1 complete (8000-13000)

python youtube_content_analyzer.py --file youtube-gaurav.csv --processes 12 --resume 13000 --max-runs 5000
echo Batch 2 complete (13000-18000)

python youtube_content_analyzer.py --file youtube-gaurav.csv --processes 12 --resume 18000 --max-runs 5000
echo Batch 3 complete (18000-23000)

python youtube_content_analyzer.py --file youtube-gaurav.csv --processes 12 --resume 23000 --max-runs 5000
echo Batch 4 complete (23000-28000)

python youtube_content_analyzer.py --file youtube-gaurav.csv --processes 12 --resume 28000
echo Final batch complete (28000-end)

echo All processing complete!
