#!/usr/bin/env bash
# Streams the results of a running simulation to the GitHub release $RELEASE (gh must be authenticated):
#   grids.tar.gz               SpSIC.dat, Incident.dat and the grid files of res/ (segment 01 only)
#   segNN_Zaaaa-Zbbbb.tar.gz   res/{IS,Fl,Or,Ne,IO} of the snapshots completed since the last pass
#   latest_segNN.tar.gz        res/Imax.dat, Nemax.dat, Energy.dat, snapshots of this segment, log_segNN.txt
# kw_THz is not streamed. A snapshot is complete once its number is listed in res/snapshots.txt.
# Runs a pass every <interval> seconds; when <out>/.stream_stop appears it makes a final pass and exits.
# Usage: stream_results.sh <out dir> <segment, two digits> [interval s]
set -u

out=$1
seg=$2
interval=${3:-600}
: "${RELEASE:?RELEASE must name the target release}"

state="$out/.streamed"
touch "$state"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

upload() {
  local attempt
  for attempt in 1 2 3 4 5; do
    gh release upload "$RELEASE" "$1" --clobber && return 0
    echo "upload of $(basename "$1") failed (attempt $attempt)" >&2
    sleep 30
  done
  return 1
}

stream_grids() {
  [ "$seg" = "01" ] || return 0
  [ -f "$out/.grids_streamed" ] && return 0
  # Every grid file is written before the first snapshot is completed.
  [ -s "$out/res/snapshots.txt" ] || return 0
  local files=(SpSIC.dat Incident.dat) f b
  for f in "$out"/res/*.dat; do
    b=$(basename "$f")
    case "$b" in
      Imax.dat | Nemax.dat | Energy.dat | Den.dat) ;;
      *) files+=("res/$b") ;;
    esac
  done
  tar -czf "$work/grids.tar.gz" -C "$out" "${files[@]}" && upload "$work/grids.tar.gz" && touch "$out/.grids_streamed"
  rm -f "$work/grids.tar.gz"
}

stream_snapshots() {
  [ -f "$out/res/snapshots.txt" ] || return 0
  local new=() z line files=() d n name
  while read -r line; do
    z=${line//[[:space:]]/}
    [ -n "$z" ] || continue
    grep -qx "$z" "$state" || new+=("$z")
  done <"$out/res/snapshots.txt"
  [ ${#new[@]} -gt 0 ] || return 0
  for z in "${new[@]}"; do
    n=$(printf 'Z%04d.dat' "$((10#$z))")
    for d in IS Fl Or Ne IO; do files+=("res/$d/$n"); done
  done
  name=$(printf 'seg%s_Z%04d-Z%04d.tar.gz' "$seg" "$((10#${new[0]}))" "$((10#${new[-1]}))")
  if tar -czf "$work/$name" -C "$out" "${files[@]}" && upload "$work/$name"; then
    printf '%s\n' "${new[@]}" >>"$state"
    echo "streamed $name (${#new[@]} snapshots)"
  fi
  rm -f "$work/$name"
}

stream_latest() {
  local stage="$work/latest" f
  rm -rf "$stage"
  mkdir -p "$stage/res"
  for f in Imax.dat Nemax.dat Energy.dat; do
    [ -f "$out/res/$f" ] && cp "$out/res/$f" "$stage/res/"
  done
  [ -f "$out/res/snapshots.txt" ] && cp "$out/res/snapshots.txt" "$stage/res/snapshots_seg$seg.txt"
  [ -f "$out/log_seg$seg.txt" ] && cp "$out/log_seg$seg.txt" "$stage/"
  tar -czf "$work/latest_seg$seg.tar.gz" -C "$stage" . && upload "$work/latest_seg$seg.tar.gz"
  rm -rf "$stage" "$work/latest_seg$seg.tar.gz"
}

pass() {
  stream_grids
  stream_snapshots
  stream_latest
}

while :; do
  for ((t = 0; t < interval; t += 5)); do
    [ -f "$out/.stream_stop" ] && break
    sleep 5
  done
  pass
  [ -f "$out/.stream_stop" ] && break
done
echo "streamer finished"
