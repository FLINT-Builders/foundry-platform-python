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


"""Spec for uncapped Arrow and Parquet table materialization.

Dataset.read_table, SqlQuery.get_results, ExperimentArtifactTable.parquet, and
ExperimentSeries.parquet (sync and async) return ArrowTableResponse or
ParquetTableResponse. to_pyarrow, to_pandas, to_polars, and to_duckdb then
materialize those bytes with PyArrow and no decoded-size cap.

These tests require the SDK to refuse a table whose declared or decoded size
is above DECODED_SIZE_CAP (8 MiB) without allocating that much in the PyArrow
memory pool. A byte limit on the HTTP body alone does not satisfy them: the
payloads below are far smaller than the tables they declare.

The cases fail while that cap is missing.
"""

import asyncio
import gc
import struct
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DECODED_SIZE_CAP = 8 * 1024 * 1024
DECLARED_SIZE = 32 * 1024 * 1024
# 2_000_000 int64 values decode to 16 MiB, above DECODED_SIZE_CAP.
VALID_INT64_ROWS = 2_000_000

DATASET_RID = "ri.foundry.main.dataset.test-table"
MODEL_RID = "ri.foundry.main.model.test-model"
EXPERIMENT_RID = "ri.foundry.main.experiment.test-exp"
SQL_QUERY_ID = "query-1"
ARTIFACT_NAME = "metrics"
SERIES_NAME = "loss"
# Obvious placeholder. The local server does not check it.
AUTH_TOKEN = "placeholder"

ARROW_CONVERTERS = ("to_pyarrow", "to_pandas", "to_polars", "to_duckdb")
PARQUET_CONVERTERS = ARROW_CONVERTERS
HTTP_CASES = (
    "read_table",
    "get_results",
    "artifact_parquet",
    "series_parquet",
)


def _zigzag32(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    number = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        number |= (byte & 0x7F) << shift
        if byte < 0x80:
            return number, offset
        shift += 7


def _write_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _arrow_metadata_bomb(declared: int = DECLARED_SIZE) -> bytes:
    """Small LZ4 Arrow IPC stream whose first buffer declares ``declared`` bytes."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = pa.table({"x": pa.array([1, 2, 3, 4], type=pa.int64())})
    sink = pa.BufferOutputStream()
    with ipc.new_stream(
        sink, table.schema, options=ipc.IpcWriteOptions(compression="lz4")
    ) as writer:
        writer.write_table(table)
    raw = sink.getvalue().to_pybytes()
    reader = ipc.MessageReader.open_stream(pa.py_buffer(raw))
    schema = reader.read_next_message()
    batch = reader.read_next_message()
    body = bytearray(batch.body.to_pybytes())
    if struct.unpack_from("<q", body, 0)[0] <= 0:
        raise RuntimeError("compressed Arrow body has no uncompressed-size prefix")
    struct.pack_into("<q", body, 0, declared)
    schema_bytes = schema.serialize().to_pybytes()
    if not raw.startswith(schema_bytes):
        raise RuntimeError("schema message is not a prefix of the IPC stream")
    rest = bytearray(raw[len(schema_bytes) :])
    original = batch.body.to_pybytes()
    index = rest.rfind(original)
    if index < 0:
        raise RuntimeError("record-batch body was not found in the IPC stream")
    rest[index : index + len(original)] = body
    return schema_bytes + bytes(rest)


def _parquet_metadata_bomb(declared: int = DECLARED_SIZE) -> bytes:
    """Small ZSTD Parquet file whose first page declares ``declared`` uncompressed bytes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"x": pa.array([1, 2, 3, 4], type=pa.int64())})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    data = bytearray(sink.getvalue().to_pybytes())
    if data[:4] != b"PAR1":
        raise RuntimeError("parquet magic missing")
    # PageHeader field 1 is the page type. Field 2 is uncompressed_page_size.
    offset = 4
    header = data[offset]
    if (header & 0x0F) != 5:
        raise RuntimeError("first parquet page field is not i32")
    offset += 1
    _, offset = _read_varint(data, offset)
    header = data[offset]
    if (header & 0x0F) != 5 or (header >> 4) == 0:
        raise RuntimeError("uncompressed_page_size field was not where the page header puts it")
    offset += 1
    _, value_end = _read_varint(data, offset)
    data[offset:value_end] = _write_varint(_zigzag32(declared))
    return bytes(data)


def _valid_arrow_table() -> bytes:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    column = pa.array([0] * VALID_INT64_ROWS, type=pa.int64())
    table = pa.table({"x": column})
    sink = pa.BufferOutputStream()
    with ipc.new_stream(
        sink, table.schema, options=ipc.IpcWriteOptions(compression="lz4")
    ) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _valid_parquet_table() -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    column = pa.array([0] * VALID_INT64_ROWS, type=pa.int64())
    table = pa.table({"x": column})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue().to_pybytes()


def _convert(response, method: str):
    return getattr(response, method)()


def _memory_refusal(action) -> int:
    """Return 0 when action raises without exceeding DECODED_SIZE_CAP in the PyArrow pool."""
    import pyarrow as pa

    pool = pa.default_memory_pool()
    gc.collect()
    before_live = pool.bytes_allocated()
    before_peak = pool.max_memory()
    try:
        action()
    except ImportError as exc:
        print(f"MISSING {type(exc).__name__}: {exc}")
        return 1
    except Exception as exc:
        live = pool.bytes_allocated() - before_live
        peak = pool.max_memory() - before_peak
        if live >= DECODED_SIZE_CAP or peak >= DECODED_SIZE_CAP:
            print(
                f"ALLOCATED live={live} peak={peak} "
                f"exc={type(exc).__name__}: {exc}"
            )
            return 1
        print(f"REFUSED live={live} peak={peak} exc={type(exc).__name__}: {exc}")
        return 0
    live = pool.bytes_allocated() - before_live
    peak = pool.max_memory() - before_peak
    print(f"MATERIALIZED live={live} peak={peak}")
    return 1


class _PayloadHandler(BaseHTTPRequestHandler):
    payload = b""
    requests = 0

    def do_GET(self):  # noqa: N802
        body = type(self).payload
        type(self).requests += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        return


def _serve(payload: bytes) -> tuple[ThreadingHTTPServer, int]:
    _PayloadHandler.payload = payload
    _PayloadHandler.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PayloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def _client(port: int, *, async_client: bool):
    from foundry_sdk import Config, UserTokenAuth
    from foundry_sdk.v2 import AsyncFoundryClient, FoundryClient

    auth = UserTokenAuth(token=AUTH_TOKEN)
    config = Config(scheme="http")
    hostname = f"127.0.0.1:{port}"
    if async_client:
        return AsyncFoundryClient(auth=auth, hostname=hostname, config=config, preview=True)
    return FoundryClient(auth=auth, hostname=hostname, config=config)


def _fetch_sync(port: int, case: str):
    client = _client(port, async_client=False)
    if case == "read_table":
        return client.datasets.Dataset.read_table(DATASET_RID, format="ARROW")
    if case == "get_results":
        return client.sql_queries.SqlQuery.get_results(SQL_QUERY_ID)
    if case == "artifact_parquet":
        return client.models.Model.Experiment.ArtifactTable.parquet(
            MODEL_RID, EXPERIMENT_RID, ARTIFACT_NAME
        )
    if case == "series_parquet":
        return client.models.Model.Experiment.Series.parquet(
            MODEL_RID, EXPERIMENT_RID, SERIES_NAME
        )
    raise AssertionError(case)


async def _fetch_async(port: int, case: str):
    client = _client(port, async_client=True)
    if case == "read_table":
        return await client.datasets.Dataset.read_table(DATASET_RID, format="ARROW")
    if case == "get_results":
        return await client.sql_queries.SqlQuery.get_results(SQL_QUERY_ID)
    if case == "artifact_parquet":
        return await client.models.Model.Experiment.ArtifactTable.parquet(
            MODEL_RID, EXPERIMENT_RID, ARTIFACT_NAME
        )
    if case == "series_parquet":
        return await client.models.Model.Experiment.Series.parquet(
            MODEL_RID, EXPERIMENT_RID, SERIES_NAME
        )
    raise AssertionError(case)


def _http_case(case: str, *, async_client: bool) -> int:
    if case in ("read_table", "get_results"):
        payload = _arrow_metadata_bomb()
    else:
        payload = _parquet_metadata_bomb()
    server, port = _serve(payload)
    try:
        def action():
            if async_client:
                response = asyncio.run(_fetch_async(port, case))
            else:
                response = _fetch_sync(port, case)
            if _PayloadHandler.requests < 1:
                raise AssertionError("table endpoint was not requested")
            response.to_pyarrow()

        return _memory_refusal(action)
    finally:
        server.shutdown()
        server.server_close()


def _run_child(case: str) -> int:
    from foundry_sdk._core.table import ArrowTableResponse, ParquetTableResponse

    if case.startswith("arrow-meta-"):
        method = case.removeprefix("arrow-meta-")
        payload = _arrow_metadata_bomb()
        return _memory_refusal(lambda: _convert(ArrowTableResponse(payload), method))
    if case.startswith("parquet-meta-"):
        method = case.removeprefix("parquet-meta-")
        payload = _parquet_metadata_bomb()
        return _memory_refusal(lambda: _convert(ParquetTableResponse(payload), method))
    if case == "arrow-valid":
        payload = sys.stdin.buffer.read()
        return _memory_refusal(lambda: ArrowTableResponse(payload).to_pyarrow())
    if case == "parquet-valid":
        payload = sys.stdin.buffer.read()
        return _memory_refusal(lambda: ParquetTableResponse(payload).to_pyarrow())
    if case.startswith("http-sync-"):
        return _http_case(case.removeprefix("http-sync-"), async_client=False)
    if case.startswith("http-async-"):
        return _http_case(case.removeprefix("http-async-"), async_client=True)
    print(f"unknown case {case}")
    return 1


def _assert_child(case: str, payload: bytes | None = None) -> None:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "child", case],
        input=payload,
        capture_output=True,
        check=False,
    )
    details = proc.stdout.decode() + proc.stderr.decode()
    assert proc.returncode == 0, details


@pytest.mark.parametrize("method", ARROW_CONVERTERS)
def test_arrow_metadata_bomb_is_refused(method: str) -> None:
    _assert_child(f"arrow-meta-{method}")


@pytest.mark.parametrize("method", PARQUET_CONVERTERS)
def test_parquet_metadata_bomb_is_refused(method: str) -> None:
    _assert_child(f"parquet-meta-{method}")


def test_valid_compressed_arrow_over_cap_is_refused() -> None:
    _assert_child("arrow-valid", _valid_arrow_table())


def test_valid_compressed_parquet_over_cap_is_refused() -> None:
    _assert_child("parquet-valid", _valid_parquet_table())


@pytest.mark.parametrize("case", HTTP_CASES)
def test_sync_table_endpoint_refuses_oversized_decode(case: str) -> None:
    _assert_child(f"http-sync-{case}")


@pytest.mark.parametrize("case", HTTP_CASES)
def test_async_table_endpoint_refuses_oversized_decode(case: str) -> None:
    _assert_child(f"http-async-{case}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "child":
        sys.exit(_run_child(sys.argv[2]))
