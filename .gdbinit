# LichtFeld Studio GDB initialization file
# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Enable Python
set python print-stack full

# Load pretty printers
python
import sys
import os

# Add tools/gdb to Python path
project_root = os.path.dirname(os.path.abspath(gdb.current_objfile().filename if gdb.current_objfile() else '.'))
gdb_tools_path = os.path.join(project_root, 'tools', 'gdb')
if os.path.exists(gdb_tools_path):
    sys.path.insert(0, gdb_tools_path)
    exec(open(os.path.join(gdb_tools_path, 'load_printers.py')).read())
end

# Useful settings for debugging
set print pretty on
set print array on
set print array-indexes on
set pagination off
set confirm off

# CUDA debugging settings (if cuda-gdb is available)
set cuda memcheck on
set cuda api_failures stop

# Custom aliases for tensor debugging
define pt
    print-tensor $arg0 20
end
document pt
Print tensor with first 20 values
Usage: pt tensor_variable
end

define pts
    tensor-stats $arg0
end
document pts
Compute and print tensor statistics
Usage: pts tensor_variable
end

define ptv
    print-tensor $arg0 100
end
document ptv
Print tensor with first 100 values (verbose)
Usage: ptv tensor_variable
end

define tv
    tensor-view $arg0 10 10
end
document tv
Visualize 2D tensor (10x10 window)
Usage: tv tensor_variable
end

define tvl
    tensor-view $arg0 20 20
end
document tvl
Visualize 2D tensor (20x20 window)
Usage: tvl tensor_variable
end

define td
    tensor-diff $arg0 $arg1
end
document td
Compare two tensors
Usage: td tensor1 tensor2
end

define tb
    tensor-broadcast $arg0 $arg1
end
document tb
Check broadcast compatibility
Usage: tb tensor1 tensor2
end

# Breakpoint helpers
define tensor_break_on_nan
    break $arg0
    commands
        silent
        python
import gdb
import math

def check_tensor_for_nan():
    frame = gdb.selected_frame()
    for var in frame.block():
        if str(var.type).startswith('gs::Tensor'):
            tensor = frame.read_var(var)
            # Check if tensor contains NaN
            # This is simplified - would need full implementation
            print(f"Checking tensor {var.name} for NaN...")

check_tensor_for_nan()
        end
        continue
    end
end
document tensor_break_on_nan
Set a breakpoint that checks tensors for NaN values
Usage: tensor_break_on_nan function_name
end

# Print startup message
echo \n
echo ===== LichtFeld Studio GDB Configuration Loaded =====\n
echo Available tensor commands:\n
echo   pt <tensor>             - Print tensor (20 values)\n
echo   ptv <tensor>            - Print tensor verbose (100 values)\n
echo   pts <tensor>            - Print tensor statistics\n
echo   tv <tensor>             - Visualize 2D tensor (10x10)\n
echo   tvl <tensor>            - Visualize 2D tensor (20x20)\n
echo   td <t1> <t2>            - Compare two tensors\n
echo   tb <t1> <t2>            - Check broadcast compatibility\n
echo \n
echo Full commands:\n
echo   print-tensor            - Full tensor print command\n
echo   tensor-stats            - Full statistics command\n
echo   tensor-view             - Full visualization command\n
echo   tensor-diff             - Full comparison command\n
echo   tensor-broadcast        - Full broadcast check command\n
echo =====================================================\n
echo \n