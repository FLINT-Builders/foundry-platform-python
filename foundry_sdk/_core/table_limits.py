#  Copyright 2024 Palantir Technologies, Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


"""Memory limits for decoding Arrow IPC and Parquet table responses.

PyArrow sizes many allocations from values that the payload declares (decompressed
buffer lengths, page sizes, value and row counts, Thrift container lengths) before it
checks them against the bytes that are actually present. A payload of a few bytes can
therefore make it allocate gigabytes. The functions here read those declared values
themselves, the same way PyArrow will, and refuse the payload before PyArrow decodes it
when the memory it would need exceeds the limit. After decoding, ``check_dense_size``
bounds the size of the table once dictionary, run-end and view encodings are expanded,
which is what pandas and other consumers materialize.
"""

import base64
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple

from foundry_sdk._errors.table_error import MalformedTableError, TableTooLargeError

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore

DEFAULT_TABLE_MAX_BYTES = 8 * 1024 * 1024

# Approximate in-memory cost of the objects PyArrow builds for each metadata entry.
_FIELD_COST = 512
_KEY_VALUE_COST = 64
_FIELD_NODE_COST = 256
_BUFFER_COST = 64
_THRIFT_STRUCT_COST = 256
_THRIFT_LARGE_STRUCT_COST = 1024
_THRIFT_SCALAR_COST = 8
_THRIFT_STRING_COST = 32
_NULL_VALUE_COST = 8

# Limits that Arrow and Thrift apply themselves.
_MAX_FLATBUFFER_DEPTH = 128
_MAX_THRIFT_DEPTH = 64
_PARQUET_816_PADDING = 100


def _compute() -> Any:
    import pyarrow.compute as pc

    return pc


def resolve_max_bytes(max_bytes: Optional[int]) -> int:
    if max_bytes is None:
        return DEFAULT_TABLE_MAX_BYTES
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError(f"table_max_bytes must be a positive integer, not {max_bytes!r}.")
    return max_bytes


class _Budget:
    def __init__(self, max_bytes: int, what: str):
        self.max_bytes = max_bytes
        self.what = what
        self.used = 0

    def charge(self, nbytes: float) -> None:
        self.used += nbytes
        if self.used > self.max_bytes:
            raise _too_large(self.what, self.max_bytes)


def _too_large(what: str, max_bytes: int) -> TableTooLargeError:
    return TableTooLargeError(
        f"The {what} table response needs more than {max_bytes} bytes to decode. "
        "Pass a larger Config(table_max_bytes=...) to read larger tables."
    )


def _malformed(what: str, reason: str) -> MalformedTableError:
    return MalformedTableError(f"Malformed {what} table response: {reason}.")


def check_body_size(data: bytes, what: str, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise _too_large(what, max_bytes)


# ---------------------------------------------------------------------------
# Flatbuffers (Arrow IPC metadata)
# ---------------------------------------------------------------------------


class _Flat:
    """Read-only view of a flatbuffer table. Every read is bounds checked."""

    __slots__ = ("buf", "pos", "vtable", "vtable_len")

    def __init__(self, buf: memoryview, pos: int):
        self.buf = buf
        self.pos = pos
        self.vtable = pos - _unpack(buf, "<i", pos)
        self.vtable_len = _unpack(buf, "<H", self.vtable)

    def _field(self, index: int) -> int:
        slot = 4 + 2 * index
        if slot + 2 > self.vtable_len:
            return 0
        offset = _unpack(self.buf, "<H", self.vtable + slot)
        return self.pos + offset if offset else 0

    def scalar(self, index: int, fmt: str, default: int = 0) -> int:
        at = self._field(index)
        return _unpack(self.buf, fmt, at) if at else default

    def table(self, index: int) -> Optional["_Flat"]:
        at = self._field(index)
        return _Flat(self.buf, at + _unpack(self.buf, "<I", at)) if at else None

    def vector(self, index: int, element_size: int) -> Tuple[int, int]:
        at = self._field(index)
        if not at:
            return 0, 0
        start = at + _unpack(self.buf, "<I", at)
        count = _unpack(self.buf, "<I", start)
        if start + 4 + count * element_size > len(self.buf):
            raise _malformed("Arrow", "flatbuffer vector out of bounds")
        return start + 4, count

    def tables(self, index: int) -> Iterator["_Flat"]:
        start, count = self.vector(index, 4)
        for i in range(count):
            at = start + 4 * i
            yield _Flat(self.buf, at + _unpack(self.buf, "<I", at))

    def string_length(self, index: int) -> int:
        return self.vector(index, 1)[1]


def _unpack(buf: Any, fmt: str, pos: int) -> Any:
    size = struct.calcsize(fmt)
    if pos < 0 or pos + size > len(buf):
        raise _malformed("Arrow", "read out of bounds")
    return struct.unpack_from(fmt, buf, pos)[0]


def _charge_key_values(table: _Flat, index: int, budget: _Budget) -> None:
    for key_value in table.tables(index):
        budget.charge(_KEY_VALUE_COST + key_value.string_length(0) + key_value.string_length(1))


def _charge_fields(parent: _Flat, index: int, budget: _Budget, depth: int) -> None:
    # Arrow builds a Field for every reference, so a table shared by many vector slots is
    # charged once per slot.
    if depth > _MAX_FLATBUFFER_DEPTH:
        raise _malformed("Arrow", "schema is nested too deeply")
    for field in parent.tables(index):
        budget.charge(_FIELD_COST + field.string_length(0))
        _charge_key_values(field, 6, budget)
        _charge_fields(field, 5, budget, depth + 1)


def _charge_schema(schema: _Flat, budget: _Budget) -> bool:
    """Charge for the Fields Arrow builds from a Schema table. Returns True if big endian."""
    _charge_fields(schema, 1, budget, 0)
    _charge_key_values(schema, 2, budget)
    return schema.scalar(0, "<h") == 1


def _charge_record_batch(
    batch: _Flat, body: memoryview, budget: _Budget, big_endian: bool
) -> Tuple[int, int]:
    """Return the bytes Arrow allocates to load the batch and its logical value count."""
    length = batch.scalar(0, "<q")
    if length < 0:
        raise _malformed("Arrow", "negative record batch length")
    values = length

    nodes_start, node_count = batch.vector(1, 16)
    for i in range(node_count):
        node_length = _unpack(batch.buf, "<q", nodes_start + 16 * i)
        if node_length < 0:
            raise _malformed("Arrow", "negative field node length")
        values += node_length

    buffers_start, buffer_count = batch.vector(2, 16)
    compressed = batch.table(3) is not None
    decoded = 0
    for i in range(buffer_count):
        offset = _unpack(batch.buf, "<q", buffers_start + 16 * i)
        length = _unpack(batch.buf, "<q", buffers_start + 16 * i + 8)
        if offset < 0 or length < 0 or offset + length > len(body):
            raise _malformed("Arrow", "buffer out of bounds")
        if not compressed or length == 0:
            decoded += length
            continue
        # Compressed buffers start with the uncompressed length, and Arrow allocates
        # that many bytes before it decompresses anything.
        if length < 8:
            raise _malformed("Arrow", "compressed buffer is too short")
        uncompressed = _unpack(body, "<q", offset)
        if uncompressed == -1:
            decoded += length - 8
        elif uncompressed < 0:
            raise _malformed("Arrow", "negative uncompressed buffer length")
        else:
            decoded += uncompressed

    variadic_start, variadic_count = batch.vector(4, 8)
    variadic_total = 0
    for i in range(variadic_count):
        count = _unpack(batch.buf, "<q", variadic_start + 8 * i)
        if count < 0:
            raise _malformed("Arrow", "negative variadic buffer count")
        variadic_total += count
    if variadic_total > buffer_count:
        raise _malformed("Arrow", "variadic buffer count exceeds the number of buffers")

    if big_endian:
        decoded *= 2
    budget.charge(decoded + node_count * _FIELD_NODE_COST + buffer_count * _BUFFER_COST)
    return decoded, values


def check_arrow_stream(data: bytes, max_bytes: int) -> None:
    """Refuse an Arrow IPC stream whose decoding would allocate more than ``max_bytes``."""
    import pyarrow as pa

    budget = _Budget(max_bytes, "Arrow")
    values = 0
    big_endian = False
    dictionary_bytes: Dict[int, int] = {}

    # MessageReader frames messages exactly as the stream reader used to decode the table
    # does. Reading from an in-memory buffer slices it, so no message is copied here.
    reader = pa.ipc.MessageReader.open_stream(pa.py_buffer(data))
    while True:
        try:
            message = reader.read_next_message()
        except StopIteration:
            break
        metadata = memoryview(message.metadata)
        body = memoryview(message.body) if message.body is not None else memoryview(b"")
        root = _Flat(metadata, _unpack(metadata, "<I", 0))
        _charge_key_values(root, 4, budget)
        header_type = root.scalar(1, "<B")
        header = root.table(2)
        if header is None:
            continue
        if header_type == 1:
            big_endian = _charge_schema(header, budget) or big_endian
        elif header_type == 2:
            batch = header.table(1)
            if batch is None:
                continue
            decoded, batch_values = _charge_record_batch(batch, body, budget, big_endian)
            values += batch_values
            dictionary_id = header.scalar(0, "<q")
            if header.scalar(2, "<B"):
                # Arrow concatenates a delta onto the current dictionary into a new
                # allocation, and batches that use the previous version keep it alive.
                dictionary_bytes[dictionary_id] = dictionary_bytes.get(dictionary_id, 0) + decoded
                budget.charge(dictionary_bytes[dictionary_id])
            else:
                dictionary_bytes[dictionary_id] = decoded
        elif header_type == 3:
            values += _charge_record_batch(header, body, budget, big_endian)[1]
        if values > max_bytes:
            raise _too_large("Arrow", max_bytes)


# ---------------------------------------------------------------------------
# Thrift compact protocol (Parquet metadata)
# ---------------------------------------------------------------------------

_T_STOP = 0
_T_TRUE = 1
_T_FALSE = 2
_T_BYTE = 3
_T_I16 = 4
_T_I32 = 5
_T_I64 = 6
_T_DOUBLE = 7
_T_BINARY = 8
_T_LIST = 9
_T_SET = 10
_T_MAP = 11
_T_STRUCT = 12

# The struct, list and string fields of the parquet.thrift structs Arrow deserializes,
# plus the scalar fields this module reads. Generated Thrift readers decode a field this
# way only when its wire type matches, skip it otherwise, and decode list elements by the
# declared element type below whatever type the list header claims.
_I32 = "i32"
_I64 = "i64"
_BOOL = "bool"
_BIN = "binary"


def _st(name: str) -> Tuple[str, str]:
    return ("struct", name)


def _ls(element: Any) -> Tuple[str, Any]:
    return ("list", element)


_THRIFT_SCHEMA: Dict[str, Dict[int, Any]] = {
    "FileMetaData": {
        2: _ls(_st("SchemaElement")),
        4: _ls(_st("RowGroup")),
        5: _ls(_st("KeyValue")),
        6: _BIN,
        7: _ls(_st("ColumnOrder")),
        8: _st("EncryptionAlgorithm"),
        9: _BIN,
    },
    "SchemaElement": {4: _BIN, 10: _st("LogicalType")},
    "LogicalType": {
        **{i: _st("Empty") for i in (1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 19)},
        7: _st("TimeType"),
        8: _st("TimeType"),
        17: _st("GeometryType"),
        18: _st("GeometryType"),
    },
    "TimeType": {2: _st("TimeUnit")},
    "TimeUnit": {1: _st("Empty"), 2: _st("Empty"), 3: _st("Empty")},
    "GeometryType": {1: _BIN},
    "RowGroup": {1: _ls(_st("ColumnChunk")), 4: _ls(_st("Empty"))},
    "ColumnChunk": {1: _BIN, 3: _st("ColumnMetaData"), 8: _st("ColumnCryptoMetaData"), 9: _BIN},
    "ColumnMetaData": {
        2: _ls(_I32),
        3: _ls(_BIN),
        8: _ls(_st("KeyValue")),
        12: _st("Statistics"),
        13: _ls(_st("Empty")),
        16: _st("SizeStatistics"),
        17: _st("GeospatialStatistics"),
    },
    "Statistics": {1: _BIN, 2: _BIN, 5: _BIN, 6: _BIN},
    "SizeStatistics": {2: _ls(_I64), 3: _ls(_I64)},
    "GeospatialStatistics": {1: _st("Empty"), 2: _ls(_I32)},
    "KeyValue": {1: _BIN, 2: _BIN},
    "ColumnOrder": {1: _st("Empty")},
    "ColumnCryptoMetaData": {1: _st("Empty"), 2: _st("EncryptionWithColumnKey")},
    "EncryptionWithColumnKey": {1: _ls(_BIN), 2: _BIN},
    "EncryptionAlgorithm": {1: _st("AesGcm"), 2: _st("AesGcm")},
    "AesGcm": {1: _BIN, 2: _BIN},
    "Empty": {},
    "PageHeader": {
        1: _I32,
        2: _I32,
        3: _I32,
        5: _st("DataPageHeader"),
        7: _st("DictionaryPageHeader"),
        8: _st("DataPageHeaderV2"),
    },
    "DataPageHeader": {1: _I32, 2: _I32, 3: _I32, 4: _I32, 5: _st("Statistics")},
    "DictionaryPageHeader": {1: _I32, 2: _I32},
    "DataPageHeaderV2": {
        1: _I32,
        2: _I32,
        3: _I32,
        4: _I32,
        5: _I32,
        6: _I32,
        7: _BOOL,
        8: _st("Statistics"),
    },
}

_LARGE_THRIFT_STRUCTS = {"SchemaElement", "ColumnChunk"}
_WIRE_TYPES = {_I32: (_T_I32,), _I64: (_T_I64,), _BOOL: (_T_TRUE, _T_FALSE), _BIN: (_T_BINARY,)}


class _Thrift:
    """Thrift compact protocol reader that mirrors the generated C++ readers Arrow uses.

    When ``budget`` is set, every list element and string is charged at roughly the
    memory the C++ reader allocates for it.
    """

    def __init__(self, data: bytes, pos: int, end: int, budget: Optional[_Budget] = None):
        self.data = data
        self.pos = pos
        self.end = end
        self.budget = budget
        self.key_values: List[Tuple[bytes, bytes]] = []

    def _byte(self) -> int:
        if self.pos >= self.end:
            raise _malformed("Parquet", "unexpected end of Thrift data")
        value = self.data[self.pos]
        self.pos += 1
        return value

    def _varint(self, max_bytes: int = 10) -> int:
        result = 0
        for i in range(max_bytes):
            byte = self._byte()
            result |= (byte & 0x7F) << (7 * i)
            if byte < 0x80:
                return result
        raise _malformed("Parquet", "Thrift varint is too long")

    def _i32(self) -> int:
        value = self._varint() & 0xFFFFFFFF
        value = (value >> 1) ^ -(value & 1)
        return value

    def _i64(self) -> int:
        value = self._varint() & 0xFFFFFFFFFFFFFFFF
        return (value >> 1) ^ -(value & 1)

    def _size(self) -> int:
        size = self._varint() & 0xFFFFFFFF
        if size >= 0x80000000:
            raise _malformed("Parquet", "negative Thrift size")
        return size

    def _binary(self) -> bytes:
        size = self._size()
        if self.pos + size > self.end:
            raise _malformed("Parquet", "Thrift string out of bounds")
        value = self.data[self.pos : self.pos + size]
        self.pos += size
        if self.budget is not None:
            self.budget.charge(size)
        return value

    def _list_header(self) -> Tuple[int, int]:
        header = self._byte()
        size = header >> 4
        element_type = header & 0x0F
        if size == 15:
            size = self._size()
        if not _T_TRUE <= element_type <= _T_STRUCT:
            raise _malformed("Parquet", "unknown Thrift element type")
        return size, element_type

    def _skip(self, wire_type: int, depth: int, in_container: bool = False) -> None:
        if depth > _MAX_THRIFT_DEPTH:
            raise _malformed("Parquet", "Thrift data is nested too deeply")
        if wire_type in (_T_TRUE, _T_FALSE):
            if in_container:
                self._byte()
        elif wire_type == _T_BYTE:
            self._byte()
        elif wire_type in (_T_I16, _T_I32, _T_I64):
            self._varint()
        elif wire_type == _T_DOUBLE:
            if self.pos + 8 > self.end:
                raise _malformed("Parquet", "unexpected end of Thrift data")
            self.pos += 8
        elif wire_type == _T_BINARY:
            size = self._size()
            if self.pos + size > self.end:
                raise _malformed("Parquet", "Thrift string out of bounds")
            self.pos += size
        elif wire_type in (_T_LIST, _T_SET):
            size, element_type = self._list_header()
            for _ in range(size):
                self._skip(element_type, depth + 1, True)
        elif wire_type == _T_MAP:
            size = self._size()
            if size:
                types = self._byte()
                for _ in range(size):
                    self._skip(types >> 4, depth + 1, True)
                    self._skip(types & 0x0F, depth + 1, True)
        elif wire_type == _T_STRUCT:
            self.read_struct("Empty", depth + 1)
        else:
            raise _malformed("Parquet", "unknown Thrift type")

    def _read_value(self, kind: Any, depth: int, path: str) -> Any:
        if kind == _I32:
            return self._i32()
        if kind == _I64:
            return self._i64()
        if kind == _BIN:
            return self._binary()
        if kind[0] == "struct":
            return self.read_struct(kind[1], depth + 1)
        size, _ = self._list_header()
        element = kind[1]
        if self.budget is not None:
            if element[0] == "struct":
                cost = (
                    _THRIFT_LARGE_STRUCT_COST
                    if element[1] in _LARGE_THRIFT_STRUCTS
                    else _THRIFT_STRUCT_COST
                )
            else:
                cost = _THRIFT_STRING_COST if element == _BIN else _THRIFT_SCALAR_COST
            self.budget.charge(size * cost)
        for _ in range(size):
            value = self._read_value(element, depth + 1, path)
            if path == "FileMetaData.5":
                self.key_values.append((value.get(1, b""), value.get(2, b"")))
        return None

    def read_struct(self, name: str, depth: int = 0) -> Dict[int, Any]:
        if depth > _MAX_THRIFT_DEPTH:
            raise _malformed("Parquet", "Thrift data is nested too deeply")
        schema = _THRIFT_SCHEMA[name]
        values: Dict[int, Any] = {}
        last_id = 0
        while True:
            header = self._byte()
            wire_type = header & 0x0F
            if wire_type == _T_STOP:
                return values
            delta = header >> 4
            if delta:
                field_id = ((last_id + delta + 0x8000) & 0xFFFF) - 0x8000
            else:
                field_id = ((self._i32() + 0x8000) & 0xFFFF) - 0x8000
            last_id = field_id
            kind = schema.get(field_id)
            if kind is None:
                self._skip(wire_type, depth + 1)
                continue
            if isinstance(kind, tuple):
                expected = (_T_STRUCT,) if kind[0] == "struct" else (_T_LIST,)
            else:
                expected = _WIRE_TYPES[kind]
            if wire_type not in expected:
                self._skip(wire_type, depth + 1)
                continue
            if kind == _BOOL:
                values[field_id] = wire_type == _T_TRUE
                continue
            value = self._read_value(kind, depth, f"{name}.{field_id}")
            if isinstance(value, dict) and isinstance(values.get(field_id), dict):
                # A repeated struct field is read into the same C++ object.
                values[field_id].update(value)
            else:
                values[field_id] = value


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

_PAGE_DATA = 0
_PAGE_DICTIONARY = 2
_PAGE_DATA_V2 = 3

_ENC_PLAIN_DICTIONARY = 2
_ENC_RLE = 3
_ENC_BIT_PACKED = 4
_ENC_DELTA_BINARY_PACKED = 5
_ENC_DELTA_LENGTH_BYTE_ARRAY = 6
_ENC_DELTA_BYTE_ARRAY = 7
_ENC_RLE_DICTIONARY = 8
_ENC_BYTE_STREAM_SPLIT = 9
_DICTIONARY_ENCODINGS = (_ENC_PLAIN_DICTIONARY, _ENC_RLE_DICTIONARY)
_DELTA_BYTE_ENCODINGS = (_ENC_DELTA_LENGTH_BYTE_ARRAY, _ENC_DELTA_BYTE_ARRAY)
_DELTA_ENCODINGS = (_ENC_DELTA_BINARY_PACKED,) + _DELTA_BYTE_ENCODINGS

_PHYSICAL_WIDTHS = {
    "BOOLEAN": 1,
    "INT32": 4,
    "INT64": 8,
    "INT96": 12,
    "FLOAT": 4,
    "DOUBLE": 8,
    "BYTE_ARRAY": 8,
}

_CODECS = {
    "SNAPPY": "snappy",
    "GZIP": "gzip",
    "BROTLI": "brotli",
    "ZSTD": "zstd",
    "LZ4_RAW": "lz4_raw",
}


@dataclass
class ParquetReadPlan:
    read_dictionary: List[int]
    thrift_limit: int
    schema: "pa.Schema"


@dataclass
class _Page:
    page_type: int
    uncompressed: int
    num_values: int
    encoding: int
    header: Dict[int, Any]
    data: memoryview


def _read_footer(data: bytes, budget: _Budget) -> List[Tuple[bytes, bytes]]:
    size = len(data)
    if size < 12 or data[-4:] != b"PAR1":
        raise _malformed("Parquet", "missing Parquet footer")
    footer_length = struct.unpack_from("<I", data, size - 8)[0]
    footer_start = size - 8 - footer_length
    if footer_start < 0:
        raise _malformed("Parquet", "footer length exceeds the file size")
    thrift = _Thrift(data, footer_start, size - 8, budget)
    thrift.read_struct("FileMetaData")
    return thrift.key_values


def _check_stored_arrow_schema(key_values: List[Tuple[bytes, bytes]], budget: _Budget) -> None:
    import pyarrow as pa

    for key, value in key_values:
        if key != b"ARROW:schema":
            continue
        try:
            serialized = base64.b64decode(value, validate=True)
            message = pa.ipc.read_message(pa.py_buffer(serialized))
        except Exception:
            raise _malformed("Parquet", "the stored Arrow schema cannot be read")
        metadata = memoryview(message.metadata)
        root = _Flat(metadata, _unpack(metadata, "<I", 0))
        _charge_key_values(root, 4, budget)
        schema = root.table(2)
        if root.scalar(1, "<B") == 1 and schema is not None:
            _charge_schema(schema, budget)


def _iter_pages(data: bytes, column: Any) -> Iterator[_Page]:
    """Yield the pages Arrow's SerializedPageReader reads for one column chunk."""
    start = column.data_page_offset
    dictionary_offset = column.dictionary_page_offset
    if column.has_dictionary_page and dictionary_offset and 0 < dictionary_offset < start:
        start = dictionary_offset
    length = column.total_compressed_size
    end = start + length
    if start < 0 or length < 0 or end > len(data):
        raise _malformed("Parquet", "column chunk out of bounds")
    # Arrow pads the range of files from old parquet-mr writers. Scanning the padding for
    # every file can only add pages to the estimate.
    stream_end = min(len(data), end + _PARQUET_816_PADDING)

    pos = start
    seen = 0
    while seen < column.num_values and pos < stream_end:
        thrift = _Thrift(data, pos, stream_end)
        try:
            header = thrift.read_struct("PageHeader")
            for field_id in (1, 2, 3):
                if field_id not in header:
                    raise _malformed("Parquet", "page header is missing a required field")
        except MalformedTableError:
            if pos >= end:
                return
            raise
        compressed = header[3]
        uncompressed = header[2]
        if compressed < 0 or uncompressed < 0:
            raise _malformed("Parquet", "negative page size")
        data_start = thrift.pos
        if data_start + compressed > stream_end:
            if pos >= end:
                return
            raise _malformed("Parquet", "page out of bounds")
        pos = data_start + compressed
        page_type = header[1]
        if page_type == _PAGE_DATA:
            sub = header.get(5, {})
        elif page_type == _PAGE_DATA_V2:
            sub = header.get(8, {})
        elif page_type == _PAGE_DICTIONARY:
            sub = header.get(7, {})
        else:
            continue
        num_values = sub.get(1, 0)
        if num_values < 0:
            raise _malformed("Parquet", "negative page value count")
        if page_type != _PAGE_DICTIONARY:
            seen += num_values
        encoding = sub.get(4 if page_type == _PAGE_DATA_V2 else 2, 0)
        yield _Page(
            page_type,
            uncompressed,
            num_values,
            encoding,
            sub,
            memoryview(data)[data_start : data_start + compressed],
        )


def _decompress(data: memoryview, size: int, codec: str) -> bytes:
    import pyarrow as pa

    if codec == "UNCOMPRESSED":
        return bytes(data)
    if size == 0:
        return b""
    try:
        if codec == "LZ4":
            return _decompress_lz4_hadoop(data, size)
        name = _CODECS.get(codec)
        if name is None:
            raise ValueError(codec)
        out = pa.decompress(data, decompressed_size=size, codec=name, asbytes=True)
    except Exception:
        raise _malformed("Parquet", "a page cannot be decompressed")
    if len(out) != size:
        raise _malformed("Parquet", "a page did not decompress to its declared size")
    return out


def _decompress_lz4_hadoop(data: memoryview, size: int) -> bytes:
    import pyarrow as pa

    chunks = []
    pos = 0
    total = 0
    while len(data) - pos >= 8:
        expected, compressed = struct.unpack_from(">II", data, pos)
        pos += 8
        if compressed > len(data) - pos or total + expected > size:
            break
        chunks.append(
            pa.decompress(
                data[pos : pos + compressed],
                decompressed_size=expected,
                codec="lz4_raw",
                asbytes=True,
            )
        )
        total += expected
        pos += compressed
    else:
        if pos == len(data):
            return b"".join(chunks)
    return pa.decompress(data, decompressed_size=size, codec="lz4_raw", asbytes=True)


def _page_values(page: _Page, codec: str, max_rep: int, max_def: int) -> List[bytes]:
    """Return every way Arrow may interpret the encoded values section of a data page."""
    if page.page_type == _PAGE_DATA:
        body = _decompress(page.data, page.uncompressed, codec)
        pos = 0
        for max_level, encoding in (
            (max_rep, page.header.get(4, 0)),
            (max_def, page.header.get(3, 0)),
        ):
            if max_level <= 0:
                continue
            if encoding == _ENC_RLE:
                if len(body) - pos < 4:
                    raise _malformed("Parquet", "invalid levels")
                length = struct.unpack_from("<i", body, pos)[0]
                if length < 0 or length > len(body) - pos - 4:
                    raise _malformed("Parquet", "invalid levels")
                pos += 4 + length
            elif encoding == _ENC_BIT_PACKED:
                pos += (page.num_values * max_level.bit_length() + 7) // 8
                if pos > len(body):
                    raise _malformed("Parquet", "invalid levels")
            else:
                raise _malformed("Parquet", "unknown level encoding")
        return [body[pos:]]

    levels = page.header.get(5, 0) + page.header.get(6, 0)
    if levels < 0 or levels > len(page.data) or levels > page.uncompressed:
        raise _malformed("Parquet", "invalid levels")
    raw = bytes(page.data[levels:])
    if codec == "UNCOMPRESSED":
        return [raw]
    decompressed: Optional[bytes] = None
    try:
        decompressed = _decompress(page.data[levels:], page.uncompressed - levels, codec)
    except MalformedTableError:
        if page.header.get(7, False):
            raise
    if page.header.get(7, False):
        return [decompressed] if decompressed is not None else []
    # Arrow treats an unflagged page as compressed for files from old writers.
    return [raw] + ([decompressed] if decompressed is not None else [])


class _BitStream:
    """Mirrors Arrow's BitReader for DELTA_BINARY_PACKED streams."""

    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def vlq(self, bits: int) -> int:
        max_bytes = (bits + 6) // 7
        result = 0
        for i in range(max_bytes):
            if self.pos >= len(self.data):
                raise _malformed("Parquet", "unexpected end of delta data")
            byte = self.data[self.pos]
            self.pos += 1
            result |= (byte & 0x7F) << (7 * i)
            if byte < 0x80:
                if result >> bits:
                    raise _malformed("Parquet", "delta varint overflows")
                return result
        raise _malformed("Parquet", "delta varint is too long")

    def zigzag(self, bits: int) -> int:
        value = self.vlq(bits)
        return (value >> 1) ^ -(value & 1)


@dataclass
class _DeltaHeader:
    mini_blocks: int
    total: int


def _delta_header(stream: _BitStream, value_bits: int = 32) -> Tuple[int, _DeltaHeader, int]:
    values_per_block = stream.vlq(32)
    mini_blocks = stream.vlq(32)
    total = stream.vlq(32)
    first = stream.zigzag(value_bits)
    if values_per_block == 0 or values_per_block % 128 or mini_blocks == 0:
        raise _malformed("Parquet", "invalid delta header")
    per_mini_block = values_per_block // mini_blocks
    if per_mini_block == 0 or per_mini_block % 32:
        raise _malformed("Parquet", "invalid delta header")
    return per_mini_block, _DeltaHeader(mini_blocks, total), first


def _delta_values(stream: _BitStream, budget: _Budget) -> Tuple[_DeltaHeader, int]:
    """Decode a 32-bit DELTA_BINARY_PACKED stream and return the sum of its positive values.

    Arrow allocates the header's value count times four bytes before it decodes, so that is
    charged first, which also bounds the work done here.
    """
    per_mini_block, header, value = _delta_header(stream)
    budget.charge(header.mini_blocks + 4 * header.total)
    remaining = header.total
    if remaining == 0:
        return header, 0
    positive_sum = max(_to_int32(value), 0)
    remaining -= 1
    if remaining == 0:
        return header, positive_sum
    while remaining > 0:
        min_delta = stream.zigzag(32)
        widths_start = stream.pos
        stream.pos += header.mini_blocks
        if stream.pos > len(stream.data):
            raise _malformed("Parquet", "unexpected end of delta data")
        for block_index in range(header.mini_blocks):
            if remaining == 0:
                break
            width = stream.data[widths_start + block_index]
            if width > 32:
                raise _malformed("Parquet", "delta bit width is too large")
            count = min(per_mini_block, remaining)
            mask = (1 << width) - 1
            base = stream.pos
            if base + (per_mini_block * width) // 8 > len(stream.data):
                raise _malformed("Parquet", "unexpected end of delta data")
            for i in range(count):
                if width:
                    bit = i * width
                    chunk = int.from_bytes(
                        stream.data[base + (bit >> 3) : base + (bit >> 3) + 5], "little"
                    )
                    delta = (chunk >> (bit & 7)) & mask
                else:
                    delta = 0
                value = _to_int32(value + min_delta + delta)
                if value > 0:
                    positive_sum += value
            remaining -= count
            # Arrow skips the padding of the last mini block, then stops.
            stream.pos = base + (per_mini_block * width) // 8
    return header, positive_sum


def _to_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def _charge_delta_page(page: _Page, codec: str, descriptor: Any, budget: _Budget) -> int:
    """Charge what Arrow allocates while setting up a DELTA_* decoder for a data page.

    Returns how many bytes the page's values can expand to beyond the page itself.
    """
    interpretations = _page_values(
        page, codec, descriptor.max_repetition_level, descriptor.max_definition_level
    )
    value_bits = 64 if descriptor.physical_type == "INT64" else 32
    expansion: Optional[int] = None
    for values in interpretations:
        stream = _BitStream(values)
        try:
            if page.encoding == _ENC_DELTA_BINARY_PACKED:
                budget.charge(_delta_header(stream, value_bits)[1].mini_blocks)
                expansion = expansion or 0
                continue
            prefix_bytes = 0
            if page.encoding == _ENC_DELTA_BYTE_ARRAY:
                # Each value is a prefix of the previous value plus a suffix. Arrow sizes
                # the output from the prefix lengths before it checks them.
                prefix_bytes = _delta_values(stream, budget)[1]
            header = _delta_header(stream)[1]
            budget.charge(header.mini_blocks + 4 * header.total)
        except MalformedTableError:
            continue
        expansion = max(expansion or 0, prefix_bytes)
    if expansion is None:
        raise _malformed("Parquet", "a page cannot be decoded")
    return expansion


def _max_dictionary_entry(page: _Page, codec: str) -> int:
    body = _decompress(page.data, page.uncompressed, codec)
    longest = 0
    pos = 0
    for _ in range(page.num_values):
        if pos + 4 > len(body):
            raise _malformed("Parquet", "dictionary page is too short")
        length = struct.unpack_from("<i", body, pos)[0]
        if length < 0 or pos + 4 + length > len(body):
            raise _malformed("Parquet", "invalid dictionary entry")
        longest = max(longest, length)
        pos += 4 + length
    return longest


def _arrow_leaf_types(schema: "pa.Schema") -> List[Tuple["pa.DataType", bool]]:
    """Return the Arrow type of each Parquet leaf column and whether it is a top-level field."""
    import pyarrow as pa

    leaves: List[Tuple[pa.DataType, bool]] = []

    def visit(data_type: "pa.DataType", top_level: bool) -> None:
        if isinstance(data_type, pa.BaseExtensionType):
            data_type = data_type.storage_type
        if pa.types.is_struct(data_type):
            for i in range(data_type.num_fields):
                visit(data_type.field(i).type, False)
        elif pa.types.is_map(data_type):
            visit(data_type.key_type, False)
            visit(data_type.item_type, False)
        elif _is_list_like(data_type):
            visit(data_type.value_type, False)
        else:
            leaves.append((data_type, top_level))

    for field in schema:
        visit(field.type, True)
    return leaves


def _physical_width(descriptor: Any) -> int:
    if descriptor.physical_type == "FIXED_LEN_BYTE_ARRAY":
        return max(descriptor.length, 0)
    return _PHYSICAL_WIDTHS.get(descriptor.physical_type, 16)


def _value_width(descriptor: Any, leaf_type: Optional["pa.DataType"]) -> int:
    """Upper bound on the bytes per value Arrow allocates for a leaf column's values.

    Arrow decodes into buffers of the physical type and then, unless the Arrow type has
    the same layout, converts them into a second array.
    """
    import pyarrow as pa

    physical = descriptor.physical_type
    width = _physical_width(descriptor)
    if leaf_type is None or pa.types.is_dictionary(leaf_type):
        output = None
    else:
        output = _fixed_width(leaf_type)
    if physical == "BYTE_ARRAY":
        return width + (output or 0) if leaf_type is not None else width + 32
    if physical == "FIXED_LEN_BYTE_ARRAY":
        if leaf_type is not None and pa.types.is_fixed_size_binary(leaf_type):
            return width
        return width + (output if output is not None else 32)
    if output is None or output != width or physical == "INT96":
        return width + (output if output is not None else max(width, 8))
    logical = str(descriptor.logical_type.type) if descriptor.logical_type is not None else ""
    if logical in ("TIMESTAMP", "TIME") or str(descriptor.converted_type).startswith(
        ("TIMESTAMP", "TIME")
    ):
        # A stored Arrow schema can ask for a different unit, which Arrow casts to.
        return 2 * width
    return width


def _dictionary_page_bytes(descriptor: Any, page: _Page) -> int:
    """Bytes Arrow's dictionary decoder allocates for a dictionary page."""
    if descriptor.physical_type == "BYTE_ARRAY":
        return 16 * page.num_values + page.uncompressed
    if descriptor.physical_type == "FIXED_LEN_BYTE_ARRAY":
        return (8 + _physical_width(descriptor)) * page.num_values
    return _physical_width(descriptor) * page.num_values


def check_parquet_file(data: bytes, max_bytes: int) -> ParquetReadPlan:
    """Refuse a Parquet file whose decoding would allocate more than ``max_bytes``.

    Every column is charged for the Arrow arrays it decodes to. While a column is decoded,
    Arrow also holds its levels and the largest of the buffers it sizes from declared
    values (decompression buffer, dictionary, per-page scratch space); those are freed when
    the column is done, and at most one column per thread is decoded at a time. Returns the
    options the file must then be read with.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    budget = _Budget(max_bytes, "Parquet")
    key_values = _read_footer(data, budget)
    _check_stored_arrow_schema(key_values, budget)

    # Every Thrift string and container of a well-formed file fits in the file.
    thrift_limit = max(len(data), 1)
    parquet_file = pq.ParquetFile(
        pa.BufferReader(data),
        thrift_string_size_limit=thrift_limit,
        thrift_container_size_limit=thrift_limit,
    )
    metadata = parquet_file.metadata
    schema = parquet_file.schema
    leaves: List[Any] = _arrow_leaf_types(parquet_file.schema_arrow)
    if len(leaves) != metadata.num_columns:
        leaves = [(None, False)] * metadata.num_columns

    total_rows = 0
    for rg in range(metadata.num_row_groups):
        rows = metadata.row_group(rg).num_rows
        if rows < 0:
            raise _malformed("Parquet", "negative row count")
        total_rows += rows
    if total_rows > max_bytes:
        raise _too_large("Parquet", max_bytes)

    read_dictionary: List[int] = []
    transients: List[float] = []
    for column_index in range(metadata.num_columns):
        descriptor = schema.column(column_index)
        max_rep = descriptor.max_repetition_level
        max_def = descriptor.max_definition_level
        leaf_type, top_level = leaves[column_index]
        width = _value_width(descriptor, leaf_type)
        # Values, list offsets and validity bitmaps, then definition and repetition levels.
        per_value = width + 4 * max_rep + max_def / 8
        level_bytes = 2 * ((max_def > 0) + (max_rep > 0))
        is_byte_array = descriptor.physical_type == "BYTE_ARRAY"

        values = 0
        string_bytes = 0
        transient = 0
        has_delta_strings = False
        dictionaries: List[Tuple[_Page, str, int]] = []
        for rg in range(metadata.num_row_groups):
            row_group = metadata.row_group(rg)
            column = row_group.column(column_index)
            codec = str(column.compression)
            page_values = 0
            dictionary_page: Optional[_Page] = None
            dictionary_values = 0
            for page in _iter_pages(data, column):
                if codec != "UNCOMPRESSED":
                    transient = max(transient, page.uncompressed)
                if page.page_type == _PAGE_DICTIONARY:
                    transient = max(transient, _dictionary_page_bytes(descriptor, page))
                    dictionary_page = page
                    continue
                page_values += page.num_values
                scratch = 0
                if page.encoding in _DICTIONARY_ENCODINGS:
                    scratch = 4 * page.num_values
                    dictionary_values += page.num_values
                elif is_byte_array:
                    string_bytes += page.uncompressed
                if page.encoding == _ENC_BYTE_STREAM_SPLIT:
                    scratch = 2 * _physical_width(descriptor) * page.num_values
                if page.encoding in _DELTA_ENCODINGS:
                    if page.encoding in _DELTA_BYTE_ENCODINGS:
                        has_delta_strings = True
                        scratch = 16 * page.num_values
                    string_bytes += _charge_delta_page(page, codec, descriptor, budget)
                transient = max(transient, scratch)
            values += max(page_values, row_group.num_rows)
            if is_byte_array and dictionary_page is not None and dictionary_values:
                dictionaries.append((dictionary_page, codec, dictionary_values))
            if values > max_bytes:
                raise _too_large("Parquet", max_bytes)

        if is_byte_array and dictionaries:
            if leaf_type is not None and pa.types.is_dictionary(leaf_type):
                pass
            elif (
                top_level
                and leaf_type is not None
                and _is_plain_binary(leaf_type)
                and not has_delta_strings
            ):
                # Read as a dictionary so that its expansion is measured after decoding,
                # by check_dense_size, rather than allocated while decoding.
                read_dictionary.append(column_index)
            else:
                for page, codec, count in dictionaries:
                    string_bytes += count * _max_dictionary_entry(page, codec)
        transient += values * level_bytes
        if transient > max_bytes:
            raise _too_large("Parquet", max_bytes)
        transients.append(transient)
        budget.charge(values * per_value + string_bytes)

    transients.sort(reverse=True)
    budget.charge(sum(transients[: max(pa.cpu_count(), 1)]))
    return ParquetReadPlan(
        read_dictionary=read_dictionary,
        thrift_limit=thrift_limit,
        schema=parquet_file.schema_arrow,
    )


# ---------------------------------------------------------------------------
# Expanded size of a decoded table
# ---------------------------------------------------------------------------


def _is_type(name: str, data_type: "pa.DataType") -> bool:
    import pyarrow as pa

    check = getattr(pa.types, name, None)
    return bool(check is not None and check(data_type))


def _is_plain_binary(data_type: "pa.DataType") -> bool:
    import pyarrow as pa

    return (
        pa.types.is_binary(data_type)
        or pa.types.is_string(data_type)
        or pa.types.is_large_binary(data_type)
        or pa.types.is_large_string(data_type)
    )


def _is_view_binary(data_type: "pa.DataType") -> bool:
    return _is_type("is_binary_view", data_type) or _is_type("is_string_view", data_type)


def _is_list_view(data_type: "pa.DataType") -> bool:
    return _is_type("is_list_view", data_type) or _is_type("is_large_list_view", data_type)


def _is_list_like(data_type: "pa.DataType") -> bool:
    import pyarrow as pa

    return (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
        or _is_list_view(data_type)
    )


def _offset_width(data_type: "pa.DataType") -> int:
    import pyarrow as pa

    large = (
        pa.types.is_large_binary(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_large_list(data_type)
        or _is_type("is_large_list_view", data_type)
    )
    return 8 if large else 4


def _fixed_width(data_type: "pa.DataType") -> Optional[int]:
    try:
        bits = data_type.bit_width
    except ValueError:
        return None
    return max((bits + 7) // 8, 1)


def _has_shared_layout(data_type: "pa.DataType") -> bool:
    """Whether arrays of this type can reference the same values from many slots."""
    import pyarrow as pa

    if isinstance(data_type, pa.BaseExtensionType):
        return _has_shared_layout(data_type.storage_type)
    if (
        pa.types.is_dictionary(data_type)
        or pa.types.is_union(data_type)
        or _is_type("is_run_end_encoded", data_type)
        or _is_list_view(data_type)
        or _is_view_binary(data_type)
    ):
        return True
    return any(_has_shared_layout(data_type.field(i).type) for i in range(data_type.num_fields))


def _sum(values: Any) -> float:
    import pyarrow as pa

    pc = _compute()

    return float(pc.sum(pc.cast(values, pa.float64())).as_py() or 0)


def _max(values: Any) -> int:
    pc = _compute()

    return int(pc.max(values).as_py() or 0)


def _int64(values: Any) -> Any:
    import pyarrow as pa

    pc = _compute()

    return pc.cast(values, pa.int64())


def _diff(offsets: Any) -> Any:
    pc = _compute()

    offsets = _int64(offsets)
    return pc.subtract_checked(offsets.slice(1), offsets.slice(0, len(offsets) - 1))


def _binary_offsets(array: Any) -> Any:
    import pyarrow as pa

    offset_type = pa.int64() if _offset_width(array.type) == 8 else pa.int32()
    return _int64(
        pa.Array.from_buffers(
            offset_type, len(array) + 1, [None, array.buffers()[1]], offset=array.offset
        )
    )


def _view_lengths(array: Any) -> Any:
    import pyarrow as pa

    pc = _compute()

    count = len(array)
    words = pa.Array.from_buffers(
        pa.int32(), 4 * (array.offset + count), [None, array.buffers()[1]]
    )
    views = pa.FixedSizeListArray.from_arrays(words, 4).slice(array.offset, count)
    lengths = pc.abs(_int64(pc.list_element(views, 0)))
    return pc.if_else(array.is_valid(), lengths, pa.scalar(0, pa.int64()))


def _list_view_ranges(array: Any) -> Tuple[Any, Any]:
    import pyarrow as pa

    pc = _compute()

    valid = array.is_valid()
    zero = pa.scalar(0, pa.int64())
    offsets = pc.if_else(valid, _int64(array.offsets), zero)
    sizes = pc.if_else(valid, _int64(array.sizes), zero)
    return offsets, sizes


def _run_lengths(array: Any) -> Any:
    import pyarrow as pa

    pc = _compute()

    run_ends = _int64(array.run_ends)
    start = array.offset
    stop = array.offset + len(array)
    previous = pa.concat_arrays(
        [pa.array([0], pa.int64()), run_ends.slice(0, max(len(run_ends) - 1, 0))]
    )
    ends = pc.min_element_wise(run_ends, pa.scalar(stop, pa.int64()))
    begins = pc.max_element_wise(previous, pa.scalar(start, pa.int64()))
    return pc.max_element_wise(pc.subtract(ends, begins), pa.scalar(0, pa.int64()))


def _prefix_sums(sizes: Any, limit: int, what: str) -> Any:
    import pyarrow as pa

    pc = _compute()

    sizes = pc.fill_null(sizes, 0)
    if _sum(sizes) > limit:
        raise _too_large(what, limit)
    return pa.concat_arrays([pa.array([0], pa.int64()), pc.cumulative_sum_checked(sizes)])


def _segment_sums(prefix: Any, begins: Any, ends: Any) -> Any:
    pc = _compute()

    return pc.subtract_checked(pc.take(prefix, ends), pc.take(prefix, begins))


def _add(left: Any, right: Any) -> Any:
    pc = _compute()

    if isinstance(left, int) and isinstance(right, int):
        return left + right
    return pc.add_checked(left, right)


def _element_nbytes(array: Any, limit: int, what: str) -> Any:
    """Expanded size of each element: an int when it is the same for all, else an array."""
    import pyarrow as pa

    pc = _compute()

    data_type = array.type
    count = len(array)
    if isinstance(data_type, pa.BaseExtensionType):
        return _element_nbytes(array.storage, limit, what)
    if pa.types.is_null(data_type):
        return _NULL_VALUE_COST
    if pa.types.is_boolean(data_type):
        return 1
    if pa.types.is_dictionary(data_type):
        values = _element_nbytes(array.dictionary, limit, what)
        if isinstance(values, int):
            return values
        return pc.fill_null(pc.take(values, array.indices), 0)
    if _is_type("is_run_end_encoded", data_type):
        values = _element_nbytes(array.values, limit, what)
        return values if isinstance(values, int) else _max(values)
    if pa.types.is_union(data_type):
        widest = 0
        for i in range(data_type.num_fields):
            child = _element_nbytes(array.field(i), limit, what)
            widest = max(widest, child if isinstance(child, int) else _max(child))
        return widest
    width = _fixed_width(data_type)
    if width is not None:
        return width
    if _is_plain_binary(data_type):
        return pc.add_checked(_diff(_binary_offsets(array)), _offset_width(data_type))
    if _is_view_binary(data_type):
        return pc.add_checked(_view_lengths(array), 16)
    if (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_map(data_type)
    ):
        offsets = _int64(array.offsets)
        child = _element_nbytes(array.values, limit, what)
        if isinstance(child, int):
            sizes = pc.multiply_checked(_diff(offsets), child)
        else:
            prefix = _prefix_sums(child, limit, what)
            sizes = _segment_sums(prefix, offsets.slice(0, count), offsets.slice(1))
        return pc.add_checked(sizes, _offset_width(data_type))
    if pa.types.is_fixed_size_list(data_type):
        size = data_type.list_size
        child = _element_nbytes(array.values.slice(array.offset * size, count * size), limit, what)
        if isinstance(child, int):
            return child * size
        prefix = _prefix_sums(child, limit, what)
        steps = pa.nulls(count + 1, pa.int64()).fill_null(size)
        bounds = pc.subtract(pc.cumulative_sum(steps), size)
        return _segment_sums(prefix, bounds.slice(0, count), bounds.slice(1))
    if _is_list_view(data_type):
        offsets, sizes = _list_view_ranges(array)
        child = _element_nbytes(array.values, limit, what)
        if isinstance(child, int):
            sizes = pc.multiply_checked(sizes, child)
        else:
            prefix = _prefix_sums(child, limit, what)
            sizes = _segment_sums(prefix, offsets, pc.add_checked(offsets, sizes))
        return pc.add_checked(sizes, 2 * _offset_width(data_type))
    if pa.types.is_struct(data_type):
        total: Any = 0
        for i in range(data_type.num_fields):
            total = _add(total, _element_nbytes(array.field(i), limit, what))
        return total
    return max(array.get_total_buffer_size(), 1)


def _dense_nbytes(array: Any, seen: Set[Tuple[int, ...]], limit: int, what: str) -> float:
    import pyarrow as pa

    pc = _compute()

    data_type = array.type
    count = len(array)
    if isinstance(data_type, pa.BaseExtensionType):
        return _dense_nbytes(array.storage, seen, limit, what)
    if count == 0:
        return 0
    validity = (count + 7) // 8
    if pa.types.is_null(data_type):
        return count * _NULL_VALUE_COST
    if pa.types.is_boolean(data_type):
        return 2 * validity
    if pa.types.is_dictionary(data_type):
        total = _dense_nbytes(array.indices, seen, limit, what)
        dictionary = array.dictionary
        key = tuple(b.address if b is not None else 0 for b in dictionary.buffers())
        key += (dictionary.offset, len(dictionary))
        if key not in seen:
            seen.add(key)
            total += _dense_nbytes(dictionary, seen, limit, what)
        values = _element_nbytes(dictionary, limit, what)
        if isinstance(values, int):
            return total + values * (count - array.null_count)
        counts = pc.value_counts(array.indices)
        keys = counts.field("values")
        valid = pc.is_valid(keys)
        used = pc.take(values, pc.filter(keys, valid))
        return total + _sum(pc.multiply_checked(used, pc.filter(counts.field("counts"), valid)))
    if _is_type("is_run_end_encoded", data_type):
        values = _element_nbytes(array.values, limit, what)
        if isinstance(values, int):
            return count * values + validity
        return _sum(pc.multiply_checked(_run_lengths(array), values)) + validity
    if pa.types.is_union(data_type):
        if data_type.mode == "sparse":
            return count + sum(
                _dense_nbytes(array.field(i), seen, limit, what)
                for i in range(data_type.num_fields)
            )
        return 5 * count + count * _element_nbytes(array, limit, what)
    width = _fixed_width(data_type)
    if width is not None:
        return count * width + validity
    if _is_plain_binary(data_type):
        offsets = _binary_offsets(array)
        data = offsets[count].as_py() - offsets[0].as_py()
        return (count + 1) * _offset_width(data_type) + data + validity
    if _is_view_binary(data_type):
        return 16 * count + _sum(_view_lengths(array)) + validity
    if (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_map(data_type)
    ):
        offsets = array.offsets
        first = offsets[0].as_py()
        last = offsets[count].as_py()
        child = _dense_nbytes(array.values.slice(first, last - first), seen, limit, what)
        return (count + 1) * _offset_width(data_type) + validity + child
    if pa.types.is_fixed_size_list(data_type):
        size = data_type.list_size
        child = array.values.slice(array.offset * size, count * size)
        return validity + _dense_nbytes(child, seen, limit, what)
    if _is_list_view(data_type):
        offsets, sizes = _list_view_ranges(array)
        child = _element_nbytes(array.values, limit, what)
        if isinstance(child, int):
            referenced = _sum(sizes) * child
        else:
            prefix = _prefix_sums(child, limit, what)
            referenced = _sum(_segment_sums(prefix, offsets, pc.add_checked(offsets, sizes)))
        return 2 * count * _offset_width(data_type) + validity + referenced
    if pa.types.is_struct(data_type):
        return validity + sum(
            _dense_nbytes(array.field(i), seen, limit, what) for i in range(data_type.num_fields)
        )
    return array.get_total_buffer_size()


def check_dense_size(table: "pa.Table", what: str, max_bytes: int) -> None:
    """Refuse a decoded table whose fully expanded size exceeds ``max_bytes``.

    Dictionary, run-end encoded, view and list-view arrays can reference the same values
    from many slots, so a small Arrow table can expand to far more memory when it is
    converted to pandas or to Python objects.
    """
    seen: Set[Tuple[int, ...]] = set()
    total = 0.0
    for column in table.columns:
        for chunk in column.chunks:
            if _has_shared_layout(chunk.type):
                chunk.validate(full=True)
            total += _dense_nbytes(chunk, seen, max_bytes, what)
            if total > max_bytes:
                raise _too_large(what, max_bytes)
