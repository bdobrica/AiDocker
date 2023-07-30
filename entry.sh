#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/opt/app"
/usr/bin/env python3 -m daemon -d queuecleaner
/usr/bin/env python3 -m ai
/usr/bin/env python3 -m api
