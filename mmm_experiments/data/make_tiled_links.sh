#!/usr/bin/env bash

# Create soft links from profiles/file.py to ./file.py.
# The HTTP/TCP server will use ./file.py. The ASGI codepath will use profiles/file.py.

for path in $(find *.py); do
    # ln -sf /absolute/path/to/<file.py> profiles/<file.py>
    # We get the absolute path using Python because it works on all platforms.
    # https://stackoverflow.com/a/3373298
    ln -svf $(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" $path) profiles/$path;
done