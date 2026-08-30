"""SQLite persistence for repository indexes."""

from codeintel.storage.database import (
    IndexDatabase,
    IndexDatabaseError,
    IndexStats,
    SchemaVersionError,
)
from codeintel.storage.schema import SCHEMA_VERSION, default_index_path

__all__ = [
    "SCHEMA_VERSION",
    "IndexDatabase",
    "IndexDatabaseError",
    "IndexStats",
    "SchemaVersionError",
    "default_index_path",
]
