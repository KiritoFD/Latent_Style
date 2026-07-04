#!/bin/bash
cd /home/xy/Latent_Style/SchrodingerBridge

echo "=== Python Import Test ===" > /home/xy/import_test_result.txt
echo "Date: $(date)" >> /home/xy/import_test_result.txt
echo "" >> /home/xy/import_test_result.txt

echo "Testing config_schema..." >> /home/xy/import_test_result.txt
python3 -c "from src.config_schema import *; print('CONFIG_SCHEMA OK')" >> /home/xy/import_test_result.txt 2>&1

echo "Testing model620..." >> /home/xy/import_test_result.txt
python3 -c "from src.model620 import *; print('MODEL OK')" >> /home/xy/import_test_result.txt 2>&1

echo "Testing losses620..." >> /home/xy/import_test_result.txt
python3 -c "from src.losses620 import *; print('LOSSES OK')" >> /home/xy/import_test_result.txt 2>&1

echo "Testing run module..." >> /home/xy/import_test_result.txt
python3 -c "import run; print('RUN OK')" >> /home/xy/import_test_result.txt 2>&1

echo "" >> /home/xy/import_test_result.txt
echo "=== Import Test Complete ===" >> /home/xy/import_test_result.txt

cat /home/xy/import_test_result.txt