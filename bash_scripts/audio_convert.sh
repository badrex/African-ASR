#!/bin/bash
# convert all audio files to 16kHz wav format
# usage: ./convert_audio.sh [input_dir] [output_dir]

INPUT_DIR="${1:-.}"
OUTPUT_DIR="${2:-.}"
INPUT_DIR="${INPUT_DIR%/}"
OUTPUT_DIR="${OUTPUT_DIR%/}"

LOG_FILE="conversion.log"
PROGRESS_FILE="conversion_progress.txt"

# check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "error: ffmpeg not found. please install ffmpeg first."
    exit 1
fi

audio_dir="$INPUT_DIR"
output_audio_dir="$OUTPUT_DIR"

if [[ ! -d "$audio_dir" ]]; then
    echo "error: audio directory not found at $audio_dir"
    exit 1
fi

# create output directory
mkdir -p "$output_audio_dir"

# count total files
total_files=$(ls "$audio_dir" | wc -l)
processed=0

echo "found $total_files audio files to convert"
echo "starting conversion..."
echo "log file: $LOG_FILE"
echo "progress file: $PROGRESS_FILE"

# initialize progress file
echo "0/$total_files (0%)" > "$PROGRESS_FILE"

# convert all files in audio directory
for webm_file in "$audio_dir"/*; do
    filename=$(basename "$webm_file")
    wav_file="$output_audio_dir/${filename}.wav"
    
    # convert with progress logging
    if ffmpeg -i "$webm_file" -ar 16000 -ac 1 -y "$wav_file" 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - converted: $filename -> $filename.wav" >> "$LOG_FILE"
        status="success"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - failed: $filename" >> "$LOG_FILE"
        status="failed"
    fi
    
    # update progress
    ((processed++))
    progress=$((processed * 100 / total_files))
    echo "$processed/$total_files ($progress%) - last: $filename ($status)" > "$PROGRESS_FILE"
    
    # print progress to stdout
    printf "\rProgress: %d/%d (%d%%) - %s" "$processed" "$total_files" "$progress" "$filename"
done

echo -e "\nconversion complete. check $LOG_FILE for details."
