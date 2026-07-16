import os
import sys

print("Searching for TensorRT libraries on all drives...")
drives = ['C:\\', 'D:\\', 'E:\\', 'F:\\', 'G:\\', 'H:\\', 'I:\\']
found = False

for d in drives:
    if not os.path.exists(d):
        continue
    print(f"Scanning drive {d}...")
    for root, dirs, files in os.walk(d):
        # Skip system or irrelevant directories to speed up search
        if any(x in root.lower() for x in ('$recycle.bin', 'windows\\system32\\driverstore', 'programdata\\microsoft')):
            dirs.clear() # don't recurse
            continue
        
        for f in files:
            if f.lower() in ('nvinfer.dll', 'nvinfer_10.dll', 'nvinfer_builder.dll'):
                print(f"FOUND: {os.path.join(root, f)}")
                found = True
                
if not found:
    print("No TensorRT libraries found on this system.")
else:
    print("Search complete.")
