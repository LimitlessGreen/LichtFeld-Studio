#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GDB Pretty Printers for gs::Tensor and related classes"""

import gdb
import gdb.printing
import re
import math

# ============= Improvement 2: Enhanced Shape Display =============
class TensorShapePrinter:
    """Pretty printer for gs::TensorShape"""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        dims_vec = self.val['dims_']
        dims = self._extract_vector(dims_vec)
        total_elements = int(self.val['total_elements_'])
        initialized = bool(self.val['initialized_'])

        # Improvement 8: Better error handling
        if not initialized:
            return "TensorShape[uninitialized]"

        # Show broadcasting-friendly format
        if len(dims) == 0:
            return "TensorShape(scalar)"

        # Highlight broadcast dimensions (size 1)
        dims_str = []
        for d in dims:
            if d == 1:
                dims_str.append(f"*{d}")  # Mark broadcast dims
            else:
                dims_str.append(str(d))

        return f"TensorShape[{', '.join(dims_str)}] ({total_elements} elements)"

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


# ============= Improvements 1, 3, 8: Bool support, all dtypes, error handling =============
class TensorPrinter:
    """Pretty printer for gs::Tensor"""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        # Improvement 8: Check if tensor is valid
        initialized = bool(self.val['initialized_'])
        if not initialized:
            return "Tensor[uninitialized]"

        # Get basic info
        id_val = int(self.val['id_'])
        device = int(self.val['device_'])
        dtype = int(self.val['dtype_'])
        data_ptr = self.val['data_']
        owns = bool(self.val['owns_memory_'])

        # Extract shape
        shape = self.val['shape_']
        shape_str = self._format_shape(shape)
        total_elements = int(shape['total_elements_'])

        # Improvement 8: Check for null data pointer
        if not data_ptr and total_elements > 0:
            return f"Tensor #{id_val} [INVALID: null data, shape={shape_str}]"

        # Get device and dtype names
        device_str = "cuda" if device == 1 else "cpu"
        # Improvement 1: Added bool dtype
        dtype_map = {
            0: "float32",
            1: "float16",
            2: "int32",
            3: "int64",
            4: "uint8",
            5: "bool"
        }
        dtype_str = dtype_map.get(dtype, "unknown")

        # Build ownership string
        own_str = "owned" if owns else "view"

        # Build base string
        result = f"Tensor #{id_val} ({own_str})\n"
        result += f"  shape: {shape_str}\n"
        result += f"  dtype: {dtype_str}, device: {device_str}\n"
        result += f"  data: {data_ptr}"

        # Improvement 3: Support all data types
        values = self._get_values(data_ptr, shape, dtype, device)
        if values:
            result += f"\n  values: {values}"

        return result

    def _format_shape(self, shape_val):
        """Format TensorShape with broadcasting hints"""
        dims_vec = shape_val['dims_']
        dims = self._extract_vector(dims_vec)
        total = int(shape_val['total_elements_'])

        # Highlight broadcast dimensions
        dims_str = []
        for d in dims:
            if d == 1:
                dims_str.append(f"*{d}")
            else:
                dims_str.append(str(d))

        return f"[{', '.join(dims_str)}]"

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

    def _get_values(self, data_ptr, shape_val, dtype, device):
        """Get first few values from tensor - supports all dtypes"""
        try:
            # Get total elements
            total = int(shape_val['total_elements_'])
            if total == 0:
                return "[]"

            # Only CPU for now
            if device != 0:
                return "[GPU tensor - values on device]"

            n = min(10, total)
            values = []

            if dtype == 0:  # float32
                ptr = data_ptr.cast(gdb.lookup_type("float").pointer())
                for i in range(n):
                    val = float((ptr + i).dereference())
                    values.append(f"{val:.4f}")

            elif dtype == 1:  # float16
                return "[float16 - display not implemented]"

            elif dtype == 2:  # int32
                ptr = data_ptr.cast(gdb.lookup_type("int").pointer())
                for i in range(n):
                    val = int((ptr + i).dereference())
                    values.append(str(val))

            elif dtype == 3:  # int64
                ptr = data_ptr.cast(gdb.lookup_type("long").pointer())
                for i in range(n):
                    val = int((ptr + i).dereference())
                    values.append(str(val))

            elif dtype == 4:  # uint8
                ptr = data_ptr.cast(gdb.lookup_type("unsigned char").pointer())
                for i in range(n):
                    val = int((ptr + i).dereference())
                    values.append(str(val))

            elif dtype == 5:  # bool
                ptr = data_ptr.cast(gdb.lookup_type("unsigned char").pointer())
                for i in range(n):
                    val = int((ptr + i).dereference())
                    values.append("T" if val else "F")
            else:
                return f"[unknown dtype {dtype}]"

            result = "[" + ", ".join(values)
            if total > n:
                result += f", ... ({total - n} more)"
            result += "]"

            return result
        except Exception as e:
            return f"[error reading values: {e}]"

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


# ============= Improvement 1: Added bool to DataType enum =============
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
            4: "DataType::UInt8",
            5: "DataType::Bool"
        }
        return dtype_map.get(val, f"DataType::Unknown({val})")


# ============= Improvement 4: Broadcasting and Indexing Printers =============
class BroadcastIteratorPrinter:
    """Pretty printer for gs::BroadcastIterator"""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        shape = self._format_shape(self.val['shape_'])
        broadcast_shape = self._format_shape(self.val['broadcast_shape_'])
        current_idx = int(self.val['current_index_'])
        element_count = int(self.val['element_count_'])
        total_elements = int(self.val['total_elements_'])
        done = bool(self.val['done_'])

        status = "done" if done else f"at {element_count}/{total_elements} (idx={current_idx})"
        return f"BroadcastIterator({shape} -> {broadcast_shape}, {status})"

    def _format_shape(self, shape_val):
        """Format TensorShape"""
        dims_vec = shape_val['dims_']
        dims = self._extract_vector(dims_vec)
        return str(dims)

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


class TensorIndexerPrinter:
    """Pretty printer for gs::TensorIndexer"""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        tensor_ptr = self.val['tensor_']
        indices_vec = self.val['indices_']

        # Extract indices info
        num_indices = self._get_vector_size(indices_vec)

        return f"TensorIndexer[{num_indices} index tensors]"

    def _get_vector_size(self, vec):
        """Get std::vector size"""
        impl = vec['_M_impl']
        start = impl['_M_start']
        finish = impl['_M_finish']

        if start and finish:
            return int((finish - start) / start.type.target().sizeof)
        return 0


class MaskedTensorProxyPrinter:
    """Pretty printer for gs::MaskedTensorProxy"""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        tensor_ptr = self.val['tensor_']
        mask = self.val['mask_']

        # Get mask info
        mask_shape = self._format_shape(mask['shape_'])

        return f"MaskedTensorProxy[mask shape: {mask_shape}]"

    def _format_shape(self, shape_val):
        """Format TensorShape"""
        dims_vec = shape_val['dims_']
        dims = self._extract_vector(dims_vec)
        return str(dims)

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


# ============= Improvement 5: Memory Information Printer =============
class MemoryInfoPrinter:
    """Pretty printer for gs::MemoryInfo"""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        free_bytes = int(self.val['free_bytes'])
        total_bytes = int(self.val['total_bytes'])
        allocated_bytes = int(self.val['allocated_bytes'])
        device_id = int(self.val['device_id'])

        free_mb = free_bytes / (1024*1024)
        total_mb = total_bytes / (1024*1024)
        allocated_mb = allocated_bytes / (1024*1024)

        device_str = "CPU" if device_id == -1 else f"GPU:{device_id}"

        percent_used = (allocated_mb / total_mb * 100) if total_mb > 0 else 0

        return (f"MemoryInfo({device_str}: "
                f"{allocated_mb:.1f}/{total_mb:.1f} MB used ({percent_used:.1f}%), "
                f"{free_mb:.1f} MB free)")


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

    # Improvement 4: Add broadcasting and indexing printers
    pp.add_printer('BroadcastIterator', '^gs::BroadcastIterator$', BroadcastIteratorPrinter)
    pp.add_printer('TensorIndexer', '^gs::TensorIndexer$', TensorIndexerPrinter)
    pp.add_printer('MaskedTensorProxy', '^gs::MaskedTensorProxy$', MaskedTensorProxyPrinter)

    # Improvement 5: Add memory info printer
    pp.add_printer('MemoryInfo', '^gs::MemoryInfo$', MemoryInfoPrinter)

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

        # Print shape with broadcast hints
        dims = self._get_shape_dims(shape)
        total = int(shape['total_elements_'])

        # Improvement 2: Show broadcast dimensions
        dims_str = []
        for d in dims:
            if d == 1:
                dims_str.append(f"*{d}")
            else:
                dims_str.append(str(d))

        print(f"Shape: [{', '.join(dims_str)}] ({total} elements)")

        # Print device and dtype
        device_str = "CUDA" if device == 1 else "CPU"
        # Improvement 1: Added bool
        dtype_map = {0: "float32", 1: "float16", 2: "int32", 3: "int64", 4: "uint8", 5: "bool"}
        dtype_str = dtype_map.get(dtype, "unknown")
        print(f"Device: {device_str}, Dtype: {dtype_str}")

        # Print memory info
        owns = bool(tensor['owns_memory_'])
        print(f"Memory: {data_ptr} ({'owned' if owns else 'view'})")

        # Try to print values
        if device == 0 and data_ptr:  # CPU only
            self._print_values(data_ptr, min(num_values, total), dtype)

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

    def _print_values(self, data_ptr, count, dtype):
        """Print tensor values - supports all dtypes"""
        try:
            print(f"\nValues (first {count}):")

            # Improvement 3: Support all dtypes
            if dtype == 0:  # float32
                ptr = data_ptr.cast(gdb.lookup_type("float").pointer())
                for i in range(count):
                    if i % 10 == 0:
                        print(f"  [{i:4d}]:", end="")
                    val = float((ptr + i).dereference())
                    print(f" {val:8.4f}", end="")
                    if (i + 1) % 10 == 0 or i == count - 1:
                        print()

            elif dtype == 2:  # int32
                ptr = data_ptr.cast(gdb.lookup_type("int").pointer())
                for i in range(count):
                    if i % 10 == 0:
                        print(f"  [{i:4d}]:", end="")
                    val = int((ptr + i).dereference())
                    print(f" {val:8d}", end="")
                    if (i + 1) % 10 == 0 or i == count - 1:
                        print()

            elif dtype == 5:  # bool
                ptr = data_ptr.cast(gdb.lookup_type("unsigned char").pointer())
                for i in range(count):
                    if i % 20 == 0:
                        print(f"  [{i:4d}]:", end="")
                    val = int((ptr + i).dereference())
                    print(f" {'T' if val else 'F'}", end="")
                    if (i + 1) % 20 == 0 or i == count - 1:
                        print()
            else:
                print(f"  Dtype {dtype} display not implemented")

        except Exception as e:
            print(f"Could not read values: {e}")


# ============= Improvement 6: Enhanced stats for bool tensors =============
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

        # Improvement 6: Handle bool tensors
        if dtype == 5:  # bool tensor
            self._compute_bool_stats(tensor)
        elif dtype in [0, 2, 3]:  # numeric types
            self._compute_numeric_stats(tensor, dtype)
        else:
            print(f"Statistics not implemented for dtype {dtype}")

    def _compute_bool_stats(self, tensor):
        """Stats for boolean/mask tensors"""
        data_ptr = tensor['data_']
        total = int(tensor['shape_']['total_elements_'])

        if not data_ptr or total == 0:
            print("Tensor is empty")
            return

        try:
            ptr = data_ptr.cast(gdb.lookup_type("unsigned char").pointer())

            # Count true/false
            true_count = 0
            false_count = 0

            for i in range(min(total, 100000)):  # Limit for performance
                val = int((ptr + i).dereference())
                if val:
                    true_count += 1
                else:
                    false_count += 1

            analyzed = min(total, 100000)
            sparsity = (false_count / analyzed * 100) if analyzed > 0 else 0
            coverage = (true_count / analyzed * 100) if analyzed > 0 else 0

            print(f"=== Boolean Tensor Statistics ===")
            print(f"Elements analyzed: {analyzed}/{total}")
            print(f"True values:  {true_count} ({coverage:.2f}%)")
            print(f"False values: {false_count} ({sparsity:.2f}%)")
            print(f"Sparsity: {sparsity:.2f}%")
            print(f"Coverage: {coverage:.2f}%")

        except Exception as e:
            print(f"Could not compute boolean statistics: {e}")

    def _compute_numeric_stats(self, tensor, dtype):
        """Compute numeric statistics"""
        data_ptr = tensor['data_']
        total = int(tensor['shape_']['total_elements_'])

        if not data_ptr or total == 0:
            print("Tensor is empty")
            return

        try:
            values = []

            if dtype == 0:  # float32
                float_ptr = data_ptr.cast(gdb.lookup_type("float").pointer())
                for i in range(min(total, 10000)):
                    val = float((float_ptr + i).dereference())
                    values.append(val)
            elif dtype == 2:  # int32
                int_ptr = data_ptr.cast(gdb.lookup_type("int").pointer())
                for i in range(min(total, 10000)):
                    val = int((int_ptr + i).dereference())
                    values.append(float(val))

            if not values:
                print("No values to analyze")
                return

            # Compute stats
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


# ============= Improvement 7: Visual Tensor Display =============
class TensorViewCommand(gdb.Command):
    """Visualize 2D tensor data"""

    def __init__(self):
        super(TensorViewCommand, self).__init__("tensor-view", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        args = gdb.string_to_argv(arg)
        if len(args) < 1:
            print("Usage: tensor-view <tensor_variable> [max_rows] [max_cols]")
            return

        tensor_name = args[0]
        max_rows = int(args[1]) if len(args) > 1 else 10
        max_cols = int(args[2]) if len(args) > 2 else 10

        try:
            tensor = gdb.parse_and_eval(tensor_name)
            self._view_tensor(tensor, max_rows, max_cols)
        except gdb.error as e:
            print(f"Error: {e}")

    def _view_tensor(self, tensor, max_rows, max_cols):
        """Visualize 2D tensor data"""
        device = int(tensor['device_'])
        dtype = int(tensor['dtype_'])
        shape = tensor['shape_']
        data_ptr = tensor['data_']

        if device != 0:
            print("Visualization only available for CPU tensors")
            return

        dims = self._get_shape_dims(shape)

        if len(dims) != 2:
            print(f"Visualization requires 2D tensor, got {len(dims)}D")
            return

        rows, cols = dims[0], dims[1]
        rows_to_show = min(rows, max_rows)
        cols_to_show = min(cols, max_cols)

        print(f"=== Tensor View [{rows}x{cols}] ===")
        print(f"Showing [{rows_to_show}x{cols_to_show}]")

        try:
            if dtype == 0:  # float32
                self._view_float_matrix(data_ptr, rows, cols, rows_to_show, cols_to_show)
            elif dtype == 2:  # int32
                self._view_int_matrix(data_ptr, rows, cols, rows_to_show, cols_to_show)
            elif dtype == 5:  # bool
                self._view_bool_matrix(data_ptr, rows, cols, rows_to_show, cols_to_show)
            else:
                print(f"Visualization not implemented for dtype {dtype}")

        except Exception as e:
            print(f"Could not visualize: {e}")

    def _view_float_matrix(self, data_ptr, rows, cols, max_r, max_c):
        """Visualize float matrix with color coding"""
        ptr = data_ptr.cast(gdb.lookup_type("float").pointer())

        # Find min/max for color coding
        values = []
        for i in range(min(rows * cols, 1000)):
            val = float((ptr + i).dereference())
            if not math.isnan(val) and not math.isinf(val):
                values.append(val)

        if values:
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val if max_val != min_val else 1.0
        else:
            min_val = max_val = range_val = 0.0

        # Print header
        print("     ", end="")
        for j in range(max_c):
            print(f" [{j:4d}]", end="")
        print()

        # Print rows
        for i in range(max_r):
            print(f"[{i:3d}]", end="")
            for j in range(max_c):
                idx = i * cols + j
                val = float((ptr + idx).dereference())

                # Color code by magnitude
                if math.isnan(val):
                    print("   NaN ", end="")
                elif math.isinf(val):
                    print("   Inf ", end="")
                else:
                    # Normalize to [0, 1]
                    norm = (val - min_val) / range_val if range_val > 0 else 0.5

                    # Simple ASCII intensity
                    if norm < 0.2:
                        marker = '.'
                    elif norm < 0.4:
                        marker = 'o'
                    elif norm < 0.6:
                        marker = 'O'
                    elif norm < 0.8:
                        marker = '0'
                    else:
                        marker = '@'

                    print(f" {val:6.2f}", end="")

            print()

        if rows > max_r or cols > max_c:
            print(f"... ({rows - max_r} more rows, {cols - max_c} more cols)")

    def _view_int_matrix(self, data_ptr, rows, cols, max_r, max_c):
        """Visualize integer matrix"""
        ptr = data_ptr.cast(gdb.lookup_type("int").pointer())

        # Print header
        print("     ", end="")
        for j in range(max_c):
            print(f" [{j:3d}]", end="")
        print()

        # Print rows
        for i in range(max_r):
            print(f"[{i:3d}]", end="")
            for j in range(max_c):
                idx = i * cols + j
                val = int((ptr + idx).dereference())
                print(f" {val:5d}", end="")
            print()

    def _view_bool_matrix(self, data_ptr, rows, cols, max_r, max_c):
        """Visualize boolean matrix as a bitmap"""
        ptr = data_ptr.cast(gdb.lookup_type("unsigned char").pointer())

        print("Legend: ■ = True, □ = False")

        # Print header with column indices every 10
        print("     ", end="")
        for j in range(max_c):
            if j % 10 == 0:
                print(f"{j:2d}", end="")
            else:
                print("  ", end="")
        print()

        # Print rows
        for i in range(max_r):
            print(f"[{i:3d}] ", end="")
            for j in range(max_c):
                idx = i * cols + j
                val = int((ptr + idx).dereference())
                print("■ " if val else "□ ", end="")
            print()

        if rows > max_r or cols > max_c:
            print(f"... ({rows - max_r} more rows, {cols - max_c} more cols)")

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


# ============= Additional Debugging Commands =============
class TensorDiffCommand(gdb.Command):
    """Compare two tensors"""

    def __init__(self):
        super(TensorDiffCommand, self).__init__("tensor-diff", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        args = gdb.string_to_argv(arg)
        if len(args) < 2:
            print("Usage: tensor-diff <tensor1> <tensor2>")
            return

        try:
            t1 = gdb.parse_and_eval(args[0])
            t2 = gdb.parse_and_eval(args[1])
            self._diff_tensors(t1, t2)
        except gdb.error as e:
            print(f"Error: {e}")

    def _diff_tensors(self, t1, t2):
        """Compare two tensors"""
        # Get tensor properties
        shape1 = self._get_shape_dims(t1['shape_'])
        shape2 = self._get_shape_dims(t2['shape_'])
        dtype1 = int(t1['dtype_'])
        dtype2 = int(t2['dtype_'])
        device1 = int(t1['device_'])
        device2 = int(t2['device_'])

        print("=== Tensor Comparison ===")

        # Compare shapes
        if shape1 == shape2:
            print(f"✓ Shape: {shape1}")
        else:
            print(f"✗ Shape: {shape1} vs {shape2}")

        # Compare dtypes
        dtype_map = {0: "float32", 1: "float16", 2: "int32", 3: "int64", 4: "uint8", 5: "bool"}
        if dtype1 == dtype2:
            print(f"✓ Dtype: {dtype_map.get(dtype1, 'unknown')}")
        else:
            print(f"✗ Dtype: {dtype_map.get(dtype1)} vs {dtype_map.get(dtype2)}")

        # Compare devices
        device_map = {0: "CPU", 1: "CUDA"}
        if device1 == device2:
            print(f"✓ Device: {device_map.get(device1, 'unknown')}")
        else:
            print(f"✗ Device: {device_map.get(device1)} vs {device_map.get(device2)}")

        # Compare values if possible
        if shape1 == shape2 and dtype1 == dtype2 and device1 == 0 and device2 == 0:
            self._compare_values(t1, t2)

    def _compare_values(self, t1, t2):
        """Compare tensor values"""
        dtype = int(t1['dtype_'])
        total = int(t1['shape_']['total_elements_'])

        if dtype != 0:  # Only float32 for now
            print("Value comparison only implemented for float32")
            return

        ptr1 = t1['data_'].cast(gdb.lookup_type("float").pointer())
        ptr2 = t2['data_'].cast(gdb.lookup_type("float").pointer())

        max_diff = 0.0
        diff_count = 0

        for i in range(min(total, 10000)):
            v1 = float((ptr1 + i).dereference())
            v2 = float((ptr2 + i).dereference())

            diff = abs(v1 - v2)
            if diff > 1e-6:
                diff_count += 1
                max_diff = max(max_diff, diff)

        print(f"\nValue comparison (first {min(total, 10000)} elements):")
        print(f"  Different values: {diff_count}")
        print(f"  Max difference: {max_diff:.6e}")

        if diff_count > 0:
            print(f"  First differences:")
            count = 0
            for i in range(min(total, 1000)):
                if count >= 5:
                    break
                v1 = float((ptr1 + i).dereference())
                v2 = float((ptr2 + i).dereference())
                if abs(v1 - v2) > 1e-6:
                    print(f"    [{i}]: {v1:.6f} vs {v2:.6f} (diff: {v1-v2:.6e})")
                    count += 1

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


class TensorBroadcastCommand(gdb.Command):
    """Check broadcast compatibility between tensors"""

    def __init__(self):
        super(TensorBroadcastCommand, self).__init__("tensor-broadcast", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        args = gdb.string_to_argv(arg)
        if len(args) < 2:
            print("Usage: tensor-broadcast <tensor1> <tensor2>")
            return

        try:
            t1 = gdb.parse_and_eval(args[0])
            t2 = gdb.parse_and_eval(args[1])
            self._check_broadcast(t1, t2)
        except gdb.error as e:
            print(f"Error: {e}")

    def _check_broadcast(self, t1, t2):
        """Check broadcast compatibility"""
        shape1 = self._get_shape_dims(t1['shape_'])
        shape2 = self._get_shape_dims(t2['shape_'])

        print("=== Broadcast Analysis ===")
        print(f"Tensor 1: {shape1}")
        print(f"Tensor 2: {shape2}")

        # Check if broadcastable
        max_rank = max(len(shape1), len(shape2))

        # Pad shapes with 1s on the left
        padded1 = [1] * (max_rank - len(shape1)) + shape1
        padded2 = [1] * (max_rank - len(shape2)) + shape2

        can_broadcast = True
        result_shape = []

        for i in range(max_rank):
            d1 = padded1[i]
            d2 = padded2[i]

            if d1 == d2:
                result_shape.append(d1)
            elif d1 == 1:
                result_shape.append(d2)
                print(f"  Dim {i}: {d1} broadcasts to {d2}")
            elif d2 == 1:
                result_shape.append(d1)
                print(f"  Dim {i}: {d2} broadcasts to {d1}")
            else:
                can_broadcast = False
                print(f"  Dim {i}: INCOMPATIBLE ({d1} vs {d2})")
                break

        if can_broadcast:
            print(f"\n✓ Can broadcast to: {result_shape}")
        else:
            print("\n✗ Cannot broadcast these shapes")

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


# Register commands
PrintTensorCommand()
TensorStatsCommand()
TensorViewCommand()
TensorDiffCommand()
TensorBroadcastCommand()

print("GS Tensor GDB extensions loaded. Available commands:")
print("  print-tensor <var> [n]     - Print tensor with first n values")
print("  tensor-stats <var>          - Compute tensor statistics")
print("  tensor-view <var> [r] [c]   - Visualize 2D tensor")
print("  tensor-diff <t1> <t2>       - Compare two tensors")
print("  tensor-broadcast <t1> <t2>  - Check broadcast compatibility")