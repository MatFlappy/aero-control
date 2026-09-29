"""SQL Server storage for detected safety violations."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=matvey\\sqlexpress;"
    "DATABASE=AeroControl;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)


SCHEMA_SQL = """
IF OBJECT_ID(N'dbo.Violations', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.Violations (
        Id INT IDENTITY(1,1) PRIMARY KEY,
        CreatedAtUtc DATETIME2 NOT NULL,
        Source NVARCHAR(500) NOT NULL,
        ViolationType NVARCHAR(100) NOT NULL,
        Confidence FLOAT NOT NULL,
        Latitude FLOAT NULL,
        Longitude FLOAT NULL,
        OriginalImagePath NVARCHAR(1000) NOT NULL,
        AnnotatedImagePath NVARCHAR(1000) NOT NULL,
        Status NVARCHAR(50) NOT NULL,
        MetadataJson NVARCHAR(MAX) NOT NULL
    );
END
"""


@dataclass(frozen=True)
class ViolationRecord:
    source: str
    violation_type: str
    confidence: float
    original_image_path: Path
    annotated_image_path: Path
    metadata: dict[str, Any]
    latitude: float | None = None
    longitude: float | None = None
    status: str = "requires_review"


def get_connection_string() -> str:
    return os.getenv("AEROCONTROL_SQLSERVER", DEFAULT_CONNECTION_STRING)


def connect():
    try:
        import pyodbc
    except ImportError as error:
        raise RuntimeError("Не установлен pyodbc. Установи зависимость: pip install pyodbc") from error

    return pyodbc.connect(get_connection_string())


def init_db() -> None:
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(SCHEMA_SQL)
        connection.commit()


def save_violation(record: ViolationRecord) -> int:
    created_at = datetime.now(timezone.utc).replace(tzinfo=None)
    metadata_json = json.dumps(record.metadata, ensure_ascii=False)

    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO dbo.Violations (
                CreatedAtUtc, Source, ViolationType, Confidence,
                Latitude, Longitude, OriginalImagePath, AnnotatedImagePath,
                Status, MetadataJson
            )
            OUTPUT INSERTED.Id
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            created_at,
            record.source,
            record.violation_type,
            float(record.confidence),
            record.latitude,
            record.longitude,
            str(record.original_image_path),
            str(record.annotated_image_path),
            record.status,
            metadata_json,
        )
        row = cursor.fetchone()
        connection.commit()
        return int(row[0])


def fetch_recent(limit: int = 100) -> list[dict[str, Any]]:
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(
            f"""
            SELECT TOP ({int(limit)})
                Id, CreatedAtUtc, Source, ViolationType, Confidence,
                Latitude, Longitude, OriginalImagePath, AnnotatedImagePath,
                Status, MetadataJson
            FROM dbo.Violations
            ORDER BY CreatedAtUtc DESC
            """
        )
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
