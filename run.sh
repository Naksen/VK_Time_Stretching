#!/bin/bash
python3 -m venv ./.venv && source ./.venv/bin/activate && pip3 install -r requirements.txt
python3 time_stretching.py --input-path $1 --output-path $2 --r $3