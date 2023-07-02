#!/bin/bash
/opt/app/ai.py
/opt/app/cleaner.py

export PYTHONPATH="${PYTHONPATH}:/opt/app"
/usr/bin/env python3 -m api
