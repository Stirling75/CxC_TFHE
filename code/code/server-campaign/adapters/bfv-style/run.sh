#!/bin/sh
# Standalone diagnostic entry point for this adapter (not the campaign runner).
# Cargo picks up the package-root .cargo/config.toml (offline, vendored crates).
# Inside a built server package, write --output outside code/: files under
# code/ change the campaign source identity and invalidate existing builds.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec cargo run --release --locked --offline --manifest-path "$HERE/Cargo.toml" -- "$@"
