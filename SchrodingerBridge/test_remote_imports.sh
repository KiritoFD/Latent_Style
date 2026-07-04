#!/bin/bash
cd /home/xy/Latent_Style/SchrodingerBridge
echo "=== Testing Python Imports ===" > /home/xy/import_test.txt 2>&1
python3 -c "from src.config_schema import *; print('CONFIG_SCHEMA OK')" >> /home/xy/import_test.txt 2>&1
python3 -c "from src.model620 import *; print('MODEL OK')" >> /home/xy/import_test.txt 2>&1
python3 -c "from src.losses620 import *; print('LOSSES OK')" >> /home/xy/import_test.txt 2>&1
python3 -c "import run; print('RUN OK')" >> /home/xy/import_test.txt 2>&1
echo "=== Import Test Complete ===" >> /home/xy/import_test.txt 2>&1