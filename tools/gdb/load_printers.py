#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors  
# SPDX-License-Identifier: GPL-3.0-or-later

"""Loader script for GDB pretty printers"""

import sys
import os

# Add the directory containing this script to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import and register tensor printers
import tensor_printer

print("LichtFeld Studio GDB pretty printers loaded successfully")
