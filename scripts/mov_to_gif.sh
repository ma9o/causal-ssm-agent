#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") <input.mov> [output.gif] [--width N] [--fps N] [--colors N]

Converts a video file to a palette-optimized GIF for README embedding.

Defaults: width=800, fps=12, colors=128
If output is omitted, replaces the input extension with .gif.
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

INPUT="$1"
shift

if [[ ! -f "$INPUT" ]]; then
  echo "error: input file not found: $INPUT" >&2
  exit 1
fi

OUTPUT="${INPUT%.*}.gif"
WIDTH=800
FPS=12
COLORS=128

if [[ $# -gt 0 && "${1:0:2}" != "--" ]]; then
  OUTPUT="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --width)  WIDTH="$2";  shift 2 ;;
    --fps)    FPS="$2";    shift 2 ;;
    --colors) COLORS="$2"; shift 2 ;;
    *) echo "error: unknown flag: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg not found on PATH" >&2
  exit 1
fi

ffmpeg -y -i "$INPUT" \
  -vf "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=${COLORS}[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
  "$OUTPUT"

ls -lh "$OUTPUT"
