#!/usr/bin/env bash
set -euo pipefail

# Download/verify external HAR/HAQER tracks.
# Default source is a validated external copy. Override the connection
# variables when running from a host with a different SSH configuration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT/tables/har_haqer_tracks}"
TARGET_DIR="$BUNDLE_DIR/external_tracks/ucsc"
REMOTE_USER="${REMOTE_USER:-<remote-user>}"
REMOTE_HOST="${REMOTE_HOST:-<remote-host>}"
REMOTE_PORT="${REMOTE_PORT:-<remote-port>}"
SOURCE_MODE="${SOURCE_MODE:-remote}"
SOURCE_ROOT="${HAR_HAQER_SOURCE_ROOT:-<source-project-root>}"

# Set HAR_HAQER_SOURCE_ROOT to the validated external source before downloading.
REMOTE_H3K27AC="$SOURCE_ROOT/external_tracks/ucsc/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig"
REMOTE_PHASTCONS="$SOURCE_ROOT/external_tracks/ucsc/hg38.phastCons100way.bw"
REMOTE_CCRE="$SOURCE_ROOT/external_tracks/ucsc/encodeCcreCombined.hg38.bb"
REMOTE_REFSEQ="$SOURCE_ROOT/reference/hg38_refGene.txt.gz"
REMOTE_ATAC_HUMAN_PEAKS="$SOURCE_ROOT/results/final_har_haqer_panel/formal_main_figure_split/external_track_cache/GSE221830_Human_idr.optimal_peak.narrowPeak.gz"
REMOTE_ATAC_CHIMP_PEAKS="$SOURCE_ROOT/results/final_har_haqer_panel/formal_main_figure_split/external_track_cache/GSE221830_Chimp_idr.optimal_peak.narrowPeak.gz"

usage() {
  echo "Usage: $0 [--list | --download]"
  echo "  --list      print all known source addresses and eligibility notes"
  echo "  --download  download H3K27ac and phastCons100way, then verify SHA256"
}

list_sources() {
  cat <<EOF
H3K27ac:
  source=$REMOTE_H3K27AC
  target=$TARGET_DIR/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig
  status=validated external track; downloaded by default
phastCons100way:
  source=$REMOTE_PHASTCONS
  target=$TARGET_DIR/hg38.phastCons100way.bw
  status=validated external track; downloaded by default
cCRE/CAGE:
  source=$REMOTE_CCRE
  status=cCRE file exists upstream, but no matched CAGE component; not used by current figure
RefSeq:
  source=$REMOTE_REFSEQ
  target=$BUNDLE_DIR/reference/hg38_refGene.txt.gz
  status=already tracked in the bundle; no download by default
ATAC-seq:
  human_candidate_peaks=$REMOTE_ATAC_HUMAN_PEAKS
  chimp_candidate_peaks=$REMOTE_ATAC_CHIMP_PEAKS
  status=candidate-only QC files; no quantitative CPM/delta track, not used
Human-ape delta:
  status=no validated locus-matched source address available
Human-branch phyloP:
  status=no validated human-branch source address available; ordinary phyloP100way is not a substitute
EOF
}

download_one() {
  local source_path="$1"
  local target_path="$2"
  local expected_sha="$3"
  mkdir -p "$(dirname "$target_path")"
  if [[ "$SOURCE_MODE" != "remote" ]]; then
    echo "Unsupported SOURCE_MODE=$SOURCE_MODE; use SOURCE_MODE=remote or download manually." >&2
    exit 2
  fi
  scp -P "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST:$source_path" "$target_path"
  local observed_sha
  observed_sha="$(sha256sum "$target_path" | awk '{print $1}')"
  if [[ "$observed_sha" != "$expected_sha" ]]; then
    echo "SHA256 mismatch for $target_path: observed=$observed_sha expected=$expected_sha" >&2
    exit 1
  fi
  echo "$target_path"
  echo "  sha256=$observed_sha"
}

case "${1:---download}" in
  --list)
    list_sources
    ;;
  --download)
    mkdir -p "$TARGET_DIR"
    download_one "$REMOTE_H3K27AC" "$TARGET_DIR/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig" 67dc786495f630899ed9078656cfe95915a3d75fd239ef2aa493aa1274d7c71e
    download_one "$REMOTE_PHASTCONS" "$TARGET_DIR/hg38.phastCons100way.bw" 920977066d9ef80b73d9d973f777ac50b6ed38bee53e3431451e40fbd824e568
    ;;
  *)
    usage
    exit 2
    ;;
esac
