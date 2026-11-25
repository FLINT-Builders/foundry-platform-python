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


from foundry_sdk._errors.palantir_exception import PalantirException


class TableTooLargeError(PalantirException):
    """
    Decoding the table response would use more memory than the configured limit allows.
    Raise the limit with ``Config(table_max_bytes=...)`` to read larger tables.
    """


class MalformedTableError(PalantirException, ValueError):
    """The table response is not a well-formed Arrow IPC stream or Parquet file."""
