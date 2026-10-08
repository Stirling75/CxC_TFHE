#!/bin/sh
# Compare out/ with expected/; exits non-zero on any difference.
cd "$(dirname "$0")"
fail=0
for f in expected/*; do
  n=$(basename "$f"); g=out/data/$n; case "$n" in *.tex) g=out/$n;; esac
  if [ ! -f "$g" ]; then echo "missing  $n"; fail=1
  elif cmp -s "$f" "$g"; then echo "same     $n"
  else echo "DIFFERS  $n"; fail=1; fi
done
exit $fail
