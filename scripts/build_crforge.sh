#!/bin/sh
# Compile the CRForge bridge class against an installed CRForge env distribution.
#   scripts/build_crforge.sh /path/to/crforge/env/build/install/env
# The classes go to build/crforge; python/cfbot/crforge.py puts that folder and the
# distribution's jars on the JVM's class path.
set -e
install="${1:?give the CRForge env distribution (crforge/env/build/install/env)}"
here="$(cd "$(dirname "$0")/.." && pwd)"
javac="${JAVA_HOME:+$JAVA_HOME/bin/}javac"
mkdir -p "$here/build/crforge"
"$javac" --release 21 -Xlint:-options -cp "$install/lib/*" -d "$here/build/crforge" "$here"/java/cfbot/bridge/*.java
echo "built $here/build/crforge"
