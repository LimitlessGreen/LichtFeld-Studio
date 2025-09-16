#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GDB Pretty Printers for gs::Tensor and related classes"""

import gdb
import gdb.printing
import re

class TensorShapePrinter:
    """Pretty printer for gs::TensorShape"""
    
    def __init__(self, val):
        self.val = val
        
    def to_string(self):
        dims_vec = self.val['dims_']
        dims = self._extract_vector(dims_vec)
        total_elements = int(self.val['total_elements_'])
        
        return f"TensorShape{dims} ({total_elements} elements)"
    
    def _extract_vector(self, vec):
        """Extract std::vector contents"""
        impl = vec['_M_impl']
        start = impl['_M_start']
        finish = impl['_M_finish']
        
        dims = []
        if start and finish:
            size = int((finish - start) / start.type.target().sizeof)
            for i in range(min(size, 10)):  # Limit to 10 dims
                dims.append(int((start + i).dereference()))
        
        return dims
        
    def display_hint(self):
        return 'array'


class TensorPrinter:
    """Pretty printer for gs::Tensor"""
    
    def __init__(self, val):
        self.val = val
        
    def to_string(self):
        # Get basic info
        id_val = int(self.val['id_'])
        device = int(self.val['device_'])
        dtype = int(self.val['dtype_'])
        data_ptr = self.val['data_']
        owns = bool(self.val['owns_memory_'])
        
        # Extract shape
        shape = self.val['shape_']
        shape_str = self._format_shape(shape)
        
        # Get device and dtype names
        device_str = "cuda" if device == 1 else "cpu"
        dtype_map = {
            0: "float32",
            1: "float16", 
            2: "int32",
            3: "int64",
            4: "uint8"
        }
        dtype_str = dtype_map.get(dtype, "unknown")
        
        # Build ownership string
        own_str = "owned" if owns else "view"
        
        # Build base string
        result = f"Tensor #{id_val} ({own_str})\n"
        result += f"  shape: {shape_str}\n"
        result += f"  dtype: {dtype_str}, device: {device_str}\n"
        result += f"  data: {data_ptr}"
        
        # Try to get values if CPU and float32
        if device == 0 and dtype == 0 and data_ptr:
            values = self._get_values(data_ptr, shape)
            if values:
                result += f"\n  values: {values}"
                
        return result
    
    def _format_shape(self, shape_val):
        """Format TensorShape"""
        dims_vec = shape_val['dims_']
        dims = self._extract_vector(dims_vec)
        total = int(shape_val['total_elements_'])
        return f"{dims}"
    
    def _extract_vector(self, vec):
        """Extract std::vector contents"""
        impl = vec['_M_impl']
        start = impl['_M_start']
        finish = impl['_M_finish']
        
        dims = []
        if start and finish:
            size = int((finish - start) / start.type.target().sizeof)
            for i in range(min(size, 10)):
                dims.append(int((start + i).dereference()))
        
        return dims
    
    def _get_values(self, data_ptr, shape_val):
        """Get first few values from tensor"""
        try:
            # Get total elements
            total = int(shape_val['total_elements_'])
            if total == 0:
                return "[]"
            
            # Cast to float pointer
            float_ptr = data_ptr.cast(gdb.lookup_type("float").pointer())
            
            # Get first few values
            n = min(10, total)
            values = []
            for i in range(n):
                val = float((float_ptr + i).dereference())
                values.append(f"{val:.4f}")
            
            result = "[" + ", ".join(values)
            if total > n:
                result += f", ... ({total - n} more)"
            result += "]"
            
            return result
        except:
            return None
    
    def display_hint(self):
        return 'map'


class TensorErrorPrinter:
    """Pretty printer for gs::TensorError"""
    
    def __init__(self, val):
        self.val = val
        
    def to_string(self):
        # Get the base runtime_error message
        msg = self._get_string(self.val.cast(gdb.lookup_type("std::runtime_error")))
        tensor_info = self._get_string(self.val['tensor_info_'])
        
        return f'TensorError("{msg}", tensor_info="{tensor_info}")'
    
    def _get_string(self, std_string_val):
        """Extract std::string contents"""
        try:
            # For C++11 and later std::string
            ptr = std_string_val['_M_dataplus']['_M_p']
            return ptr.string()
        except:
            return "<unable to read string>"


class DeviceEnumPrinter:
    """Pretty printer for gs::Device enum"""
    
    def __init__(self, val):
        self.val = val
        
    def to_string(self):
        val = int(self.val)
        if val == 0:
            return "Device::CPU"
        elif val == 1:
            return "Device::CUDA"
        else:
            return f"Device::Unknown({val})"


class DataTypeEnumPrinter:
    """Pretty printer for gs::DataType enum"""
    
    def __init__(self, val):
        self.val = val
        
    def to_string(self):
        val = int(self.val)
        dtype_map = {
            0: "DataType::Float32",
            1: "DataType::Float16",
            2: "DataType::Int32",
            3: "DataType::Int64",
            4: "DataType::UInt8"
        }
        return dtype_map.get(val, f"DataType::Unknown({val})")


def build_pretty_printers():
    """Build and return the pretty printer collection"""
    pp = gdb.printing.RegexpCollectionPrettyPrinter("gs_tensor")
    
    # Add tensor classes
    pp.add_printer('TensorShape', '^gs::TensorShape$', TensorShapePrinter)
    pp.add_printer('Tensor', '^gs::Tensor$', TensorPrinter)
    pp.add_printer('TensorError', '^gs::TensorError$', TensorErrorPrinter)
    
    # Add enums
    pp.add_printer('Device', '^gs::Device$', DeviceEnumPrinter)
    pp.add_printer('DataType', '^gs::DataType$', DataTypeEnumPrinter)
    
    return pp


def register_printers():
    """Register all pretty printers with GDB"""
    gdb.printing.register_pretty_printer(
        gdb.current_objfile(),
        build_pretty_printers())


# Auto-register when loaded
register_printers()


# ============= Convenience Commands =============

class PrintTensorCommand(gdb.Command):
    """Print tensor details with values"""
    
    def __init__(self):
        super(PrintTensorCommand, self).__init__("print-tensor", gdb.COMMAND_USER)
    
    def invoke(self, arg, from_tty):
        args = gdb.string_to_argv(arg)
        if len(args) < 1:
            print("Usage: print-tensor <tensor_variable> [num_values]")
            return
        
        tensor_name = args[0]
        num_values = int(args[1]) if len(args) > 1 else 20
        
        try:
            tensor = gdb.parse_and_eval(tensor_name)
            self._print_tensor_details(tensor, num_values)
        except gdb.error as e:
            print(f"Error: {e}")
    
    def _print_tensor_details(self, tensor, num_values):
        """Print detailed tensor information"""
        # Get tensor fields
        id_val = int(tensor['id_'])
        device = int(tensor['device_'])
        dtype = int(tensor['dtype_'])
        data_ptr = tensor['data_']
        shape = tensor['shape_']
        
        print(f"=== Tensor #{id_val} ===")
        
        # Print shape
        dims = self._get_shape_dims(shape)
        total = int(shape['total_elements_'])
        print(f"Shape: {dims} ({total} elements)")
        
        # Print device and dtype
        device_str = "CUDA" if device == 1 else "CPU"
        dtype_map = {0: "float32", 1: "float16", 2: "int32", 3: "int64", 4: "uint8"}
        dtype_str = dtype_map.get(dtype, "unknown")
        print(f"Device: {device_str}, Dtype: {dtype_str}")
        
        # Print memory info
        owns = bool(tensor['owns_memory_'])
        print(f"Memory: {data_ptr} ({'owned' if owns else 'view'})")
        
        # Try to print values
        if device == 0 and dtype == 0 and data_ptr:  # CPU float32
            self._print_values(data_ptr, min(num_values, total))
    
    def _get_shape_dims(self, shape_val):
        """Extract shape dimensions"""
        dims_vec = shape_val['dims_']
        impl = dims_vec['_M_impl']
        start = impl['_M_start']
        finish = impl['_M_finish']
        
        dims = []
        if start and finish:
            size = int((finish - start) / start.type.target().sizeof)
            for i in range(size):
                dims.append(int((start + i).dereference()))
        return dims
    
    def _print_values(self, data_ptr, count):
        """Print tensor values"""
        try:
            float_ptr = data_ptr.cast(gdb.lookup_type("float").pointer())
            print(f"\nValues (first {count}):")
            
            for i in range(count):
                if i % 10 == 0:
                    print(f"  [{i:4d}]:", end="")
                
                val = float((float_ptr + i).dereference())
                print(f" {val:8.4f}", end="")
                
                if (i + 1) % 10 == 0 or i == count - 1:
                    print()  # Newline
        except Exception as e:
            print(f"Could not read values: {e}")


class TensorStatsCommand(gdb.Command):
    """Compute statistics for a tensor"""
    
    def __init__(self):
        super(TensorStatsCommand, self).__init__("tensor-stats", gdb.COMMAND_USER)
    
    def invoke(self, arg, from_tty):
        args = gdb.string_to_argv(arg)
        if len(args) < 1:
            print("Usage: tensor-stats <tensor_variable>")
            return
        
        try:
            tensor = gdb.parse_and_eval(args[0])
            self._compute_stats(tensor)
        except gdb.error as e:
            print(f"Error: {e}")
    
    def _compute_stats(self, tensor):
        """Compute and print tensor statistics"""
        device = int(tensor['device_'])
        dtype = int(tensor['dtype_'])
        
        if device != 0:
            print("Statistics only available for CPU tensors")
            return
        
        if dtype != 0:
            print("Statistics only available for float32 tensors")
            return
        
        data_ptr = tensor['data_']
        total = int(tensor['shape_']['total_elements_'])
        
        if not data_ptr or total == 0:
            print("Tensor is empty")
            return
        
        try:
            float_ptr = data_ptr.cast(gdb.lookup_type("float").pointer())
            
            # Collect values
            values = []
            for i in range(min(total, 10000)):  # Limit to 10k for performance
                val = float((float_ptr + i).dereference())
                values.append(val)
            
            # Compute stats
            import math
            min_val = min(values)
            max_val = max(values)
            mean_val = sum(values) / len(values)
            
            # Compute std dev
            squared_diff = [(x - mean_val) ** 2 for x in values]
            std_val = math.sqrt(sum(squared_diff) / len(values))
            
            # Count special values
            nan_count = sum(1 for x in values if math.isnan(x))
            inf_count = sum(1 for x in values if math.isinf(x))
            zero_count = sum(1 for x in values if x == 0.0)
            
            print(f"=== Tensor Statistics ===")
            print(f"Elements analyzed: {len(values)}/{total}")
            print(f"Min:  {min_val:.6f}")
            print(f"Max:  {max_val:.6f}")
            print(f"Mean: {mean_val:.6f}")
            print(f"Std:  {std_val:.6f}")
            print(f"Zeros: {zero_count}")
            print(f"NaNs:  {nan_count}")
            print(f"Infs:  {inf_count}")
            
        except Exception as e:
            print(f"Could not compute statistics: {e}")


# Register commands
PrintTensorCommand()
TensorStatsCommand()

print("GS Tensor GDB extensions loaded. Available commands:")
print("  print-tensor <var> [n]  - Print tensor with first n values")
print("  tensor-stats <var>      - Compute tensor statistics")
