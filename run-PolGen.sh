#!/bin/bash

set -e

if [ ! -d "env" ]; then
    echo "Please run './PolGen-Installer.sh' first to set up the environment."
    exit 1
fi

check_internet_connection() {
    echo "Checking internet connection..."
    if ping -c 1 google.com &> /dev/null; then
        echo "Internet connection is available."
        INTERNET_AVAILABLE=1
    else
        echo "No internet connection detected."
        INTERNET_AVAILABLE=0
    fi
    echo
}

apply_patches() {
    echo "Applying Gradio 4.x compatibility patches..."
    "${PYTHON}" patch_gradio4.py
    echo
}

running_interface() {
    echo "Running Interface..."
    if [ "$INTERNET_AVAILABLE" -eq 1 ]; then
        echo "Running app.py in ONLINE mode..."
        "./env/bin/python" app.py --open
    else
        echo "Running app.py in OFFLINE mode..."
        "./env/bin/python" app.py --offline --open
    fi
}

PYTHON="${PYTHON:-./env/bin/python}"
apply_patches
check_internet_connection
running_interface
