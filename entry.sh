#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/opt/app"
/usr/bin/env python3 -m daemon -d queuecleaner
/opt/app/ai.py
/usr/bin/env python3 -m api
