#!/usr/bin/env python3
"""Quick update script: scan remote + local, regenerate CSV and HTML dashboard.

Usage:
  python docs/update_dashboard.py              # local scan only (fast)
  python docs/update_dashboard.py --remote      # also scan remote I: drive (slower)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from scan_and_dashboard import main

if __name__ == '__main__':
    main()
