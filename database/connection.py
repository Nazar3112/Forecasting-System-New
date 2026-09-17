"""
database/connection.py
======================
Konfigurasi koneksi SQLAlchemy ke MySQL lokal (XAMPP / MySQL Service).
Menggunakan driver pymysql (100% Pure Python — tanpa kompilasi C++).

Fitur utama:
  - Auto-create database  → Tidak perlu buka phpMyAdmin secara manual
  - Auto-create tables    → Skema terbuat otomatis dari ORM models
  - pool_recycle=3600     → Mencegah error "MySQL server has gone away"
  - pool_pre_ping=True    → Validasi koneksi sebelum dipakai

Penggunaan:
    from database.connection import get_engine, SessionLocal, init_database
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Muat variabel dari file .env
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baca Parameter Koneksi dari .env
# ---------------------------------------------------------------------------
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
DB_USER: str = os.getenv("DB_USER", "root")
DB_PASS: str = os.getenv("DB_PASS", "")          # XAMPP default: password kosong
DB_NAME: str = os.getenv("DB_NAME", "db_forecasting")

# URL koneksi PyMySQL (format SQLAlchemy)
DATABASE_URL: str = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    "?charset=utf8mb4"
)

# ---------------------------------------------------------------------------
# Base Deklaratif ORM
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Engine & Session Factory (lazy initialization)
# ---------------------------------------------------------------------------
engine = None
SessionLocal = None


def _build_engine():
    """Buat SQLAlchemy engine dengan konfigurasi optimal untuk MySQL."""
    global engine, SessionLocal
    if engine is None:
        engine = create_engine(
            DATABASE_URL,
            pool_recycle=3600,    # Recycle koneksi setiap 1 jam
            pool_pre_ping=True,   # Ping DB sebelum menggunakan koneksi dari pool
            pool_size=5,
            max_overflow=10,
            echo=False,
        )
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine,
        )
        logger.info("MySQL engine dibuat: %s@%s:%s/%s", DB_USER, DB_HOST, DB_PORT, DB_NAME)
    return engine


def get_engine():
    """Kembalikan instance SQLAlchemy engine (otomatis build jika None)."""
    global engine
    if engine is None:
        return _build_engine()
    return engine


def get_session_factory():
    """Kembalikan SessionLocal factory (otomatis build jika None)."""
    global SessionLocal
    if SessionLocal is None:
        _build_engine()
    return SessionLocal


# ---------------------------------------------------------------------------
# Auto-Create Database & Tables
# ---------------------------------------------------------------------------
def init_database() -> None:
    """
    Inisialisasi database MySQL secara otomatis:
      1. Buat database `db_forecasting` jika belum ada (CREATE DATABASE IF NOT EXISTS)
      2. Buat semua tabel ORM (CREATE TABLE IF NOT EXISTS)
    """
    # ── Langkah 1: Buat database jika belum ada ─────────────────────────
    server_url = (
        f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}"
        "?charset=utf8mb4"
    )
    try:
        temp_engine = create_engine(server_url, pool_pre_ping=True)
        with temp_engine.connect() as conn:
            conn.execute(text(
                f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            ))
            conn.commit()
        temp_engine.dispose()
        logger.info("Database '%s' dipastikan ada.", DB_NAME)
    except Exception as exc:
        logger.error("Gagal membuat database: %s", exc)
        raise

    # ── Langkah 2: Buat engine & tabel ORM ──────────────────────────────
    eng = get_engine()
    from database import models  # noqa: F401
    Base.metadata.create_all(bind=eng)
    logger.info("Semua tabel ORM telah dibuat / sudah ada.")


# ---------------------------------------------------------------------------
# Context Manager Session
# ---------------------------------------------------------------------------
@contextmanager
def get_db_session() -> Generator:
    """
    Context manager aman untuk session database.
    Auto-commit pada sukses, auto-rollback pada exception.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Utilitas: Cek Status Koneksi
# ---------------------------------------------------------------------------
def check_connection() -> tuple[bool, str]:
    """
    Cek apakah koneksi ke MySQL berhasil.

    Returns
    -------
    (bool, str)  (True jika terhubung, pesan status)
    """
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, f"MySQL terhubung ✓ ({DB_HOST}:{DB_PORT}/{DB_NAME})"
    except Exception as exc:
        return False, f"Koneksi MySQL gagal: {exc}"
