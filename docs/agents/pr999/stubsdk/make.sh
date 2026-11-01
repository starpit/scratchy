#!/bin/bash
# Empty stand-ins for the Spyre SDK's dylibs, so a card (spyre-hw) build LINKS on a host with no
# SDK — for type-checking the card worker locally only. Nothing here runs on a card.
set -eu
d=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$d/senlib/lib" "$d/boost"
printf 'void scratchy_stub_sdk(void) {}\n' > "$d/empty.c"
for l in senlib flightlog; do
  cc -dynamiclib -o "$d/senlib/lib/lib$l.dylib" "$d/empty.c"
done
for l in boost_log boost_log_setup boost_thread boost_filesystem boost_atomic boost_regex \
         boost_json boost_container boost_locale boost_chrono; do
  cc -dynamiclib -o "$d/boost/lib$l.dylib" "$d/empty.c"
done
find "$d" -name '*.dylib' | wc -l
