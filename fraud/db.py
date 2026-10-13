"""Minimal database interface with two backends.

    postgresql://user:pass@host:5432/dbname   production (needs sqlalchemy and psycopg2)
    sqlite:///path/to/file.db  or  sqlite://:memory:   local development and tests

All SQL in this project uses named parameters (:name) and avoids dialect-specific syntax,
so the same statements run on both. Rows come back as plain dicts.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


class SqliteDb:
    dialect = "sqlite"

    def __init__(self, path: str = ":memory:"):
        # autocommit mode; transactions are opened explicitly in transaction()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.RLock()

    def query(self, sql: str, params: Mapping[str, Any] | None = None) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(sql, dict(params or {}))
            rows = [dict(r) for r in cur.fetchall()] if cur.description else []
            return rows

    @contextmanager
    def transaction(self) -> Iterator["SqliteDb"]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def init_schema(self) -> None:
        with self._lock:
            self._conn.executescript((SQL_DIR / "schema.sqlite.sql").read_text())

    def close(self) -> None:
        self._conn.close()


class _PgTx:
    def __init__(self, conn, text):
        self._conn, self._text = conn, text

    def query(self, sql: str, params: Mapping[str, Any] | None = None) -> list[dict]:
        res = self._conn.execute(self._text(sql), dict(params or {}))
        return [dict(r._mapping) for r in res] if res.returns_rows else []


class PostgresDb:
    dialect = "postgresql"

    def __init__(self, url: str):
        from sqlalchemy import create_engine, text  # imported here so SQLite-only use needs neither

        self._text = text
        self.engine = create_engine(url, pool_pre_ping=True)

    def query(self, sql: str, params: Mapping[str, Any] | None = None) -> list[dict]:
        with self.engine.begin() as conn:
            return _PgTx(conn, self._text).query(sql, params)

    @contextmanager
    def transaction(self) -> Iterator[_PgTx]:
        with self.engine.begin() as conn:
            yield _PgTx(conn, self._text)

    def init_schema(self) -> None:
        lines = (SQL_DIR / "schema.postgres.sql").read_text().splitlines()
        raw = "\n".join(ln for ln in lines if not ln.strip().startswith("--"))
        with self.engine.begin() as conn:
            for stmt in (s.strip() for s in raw.split(";")):
                if stmt:
                    conn.execute(self._text(stmt))

    def close(self) -> None:
        self.engine.dispose()


def connect(url: str):
    if url.startswith("sqlite:///"):
        return SqliteDb(url[len("sqlite:///"):] or ":memory:")
    if url in ("sqlite://", "sqlite://:memory:"):
        return SqliteDb(":memory:")
    if url.startswith(("postgresql://", "postgresql+psycopg2://", "postgres://")):
        return PostgresDb(url.replace("postgres://", "postgresql://", 1))
    raise ValueError("DATABASE_URL must start with postgresql:// or sqlite:///")
