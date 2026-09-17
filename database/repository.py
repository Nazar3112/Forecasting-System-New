"""
database/repository.py
======================
Query helpers — operasi database MySQL level tinggi.

Perbedaan utama vs versi PostgreSQL:
  - UPSERT menggunakan sintaks MySQL native: ON DUPLICATE KEY UPDATE
  - Agregasi mingguan menggunakan:
      DATE_SUB(transaction_date, INTERVAL WEEKDAY(transaction_date) DAY)
  - Agregasi bulanan menggunakan:
      DATE_FORMAT(transaction_date, '%Y-%m-01')
  - Bulk insert menggunakan pd.DataFrame.to_sql() dengan chunksize=25000

Fungsi publik:
  fast_bulk_insert_raw()     → Insert batch ke raw_transactions
  sync_aggregations()        → UPSERT inkremental ke sales_aggregations
  get_available_categories() → List dropdown untuk UI
  fetch_series()             → DataFrame ['ds', 'y'] untuk model
  fetch_series_data()        → Alias untuk fetch_series()
  get_database_summary()     → Statistik untuk sidebar
  batch_exists()             → Cek duplikasi batch_id
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List, Optional

import pandas as pd
from sqlalchemy import text

from database.connection import get_engine, get_db_session

logger = logging.getLogger(__name__)


# ===========================================================================
# INSERT: Transaksi Raw (Batch Chunked)
# ===========================================================================

def fast_bulk_insert_raw(
    df: pd.DataFrame,
    batch_id: str,
    chunksize: int = 25_000,
) -> int:
    """
    Insert DataFrame transaksi bersih ke tabel `raw_transactions`.

    Menggunakan pd.DataFrame.to_sql() dengan chunksize=25000 untuk
    mencegah error "MySQL server has gone away" pada dataset besar.

    Parameters
    ----------
    df        : pd.DataFrame  DataFrame bersih hasil DeltaPreprocessor.
    batch_id  : str           Identifier batch (misal 'INITIAL_SEED_3YEARS').
    chunksize : int           Baris per batch insert (default: 25.000).

    Returns
    -------
    int  Jumlah baris yang berhasil diinsert.
    """
    if df.empty:
        logger.warning("fast_bulk_insert_raw: DataFrame kosong.")
        return 0

    insert_df = pd.DataFrame({
        "transaction_date": pd.to_datetime(df["TANGGAL"], errors="coerce").dt.date,
        "division_code":    df["DIV"].astype(str),
        "category_code":    df["CAT_COD"].astype(str),
        "product_code":     df.get("PRDCD", pd.Series([""] * len(df))).astype(str),
        "quantity":         pd.to_numeric(df["QTY"], errors="coerce").fillna(0).astype(int),
        "price":            pd.to_numeric(
            df.get("PRICE", pd.Series([None] * len(df))), errors="coerce"
        ),
        "transaction_type": "J",
        "batch_id":         batch_id,
    })

    insert_df = insert_df.dropna(subset=["transaction_date"])
    if insert_df.empty:
        return 0

    eng = get_engine()
    insert_df.to_sql(
        "raw_transactions",
        con=eng,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=chunksize,
    )

    logger.info(
        "fast_bulk_insert_raw: %d baris diinsert ke MySQL (batch_id=%s, chunksize=%d)",
        len(insert_df), batch_id, chunksize,
    )
    return len(insert_df)


# ===========================================================================
# UPSERT: Agregasi Deret Waktu (MySQL ON DUPLICATE KEY UPDATE)
# ===========================================================================

def sync_aggregations(start_date: Optional[date] = None) -> None:
    """
    Hitung ulang agregasi dari `raw_transactions` dan UPSERT ke
    tabel `sales_aggregations` menggunakan MySQL native:
        ON DUPLICATE KEY UPDATE ...

    Parameters
    ----------
    start_date : date, optional
        Jika diberikan, hanya proses baris dengan transaction_date >= start_date.
        None = proses seluruh tabel (untuk seeding awal).
    """
    date_filter = ""
    params: dict = {}
    if start_date is not None:
        date_filter = "AND transaction_date >= :start_date"
        params["start_date"] = start_date

    # ── DIV Mingguan ──────────────────────────────────────────────────
    _run_upsert(
        group_col="division_code",
        group_type_val="DIV",
        frequency_val="W",
        period_expr="DATE_SUB(transaction_date, INTERVAL WEEKDAY(transaction_date) DAY)",
        date_filter=date_filter,
        params=params,
    )

    # ── DIV Bulanan ───────────────────────────────────────────────────
    _run_upsert(
        group_col="division_code",
        group_type_val="DIV",
        frequency_val="M",
        period_expr="DATE_FORMAT(transaction_date, '%Y-%m-01')",
        date_filter=date_filter,
        params=params,
    )

    # ── CAT_COD Mingguan ──────────────────────────────────────────────
    _run_upsert(
        group_col="category_code",
        group_type_val="CAT_COD",
        frequency_val="W",
        period_expr="DATE_SUB(transaction_date, INTERVAL WEEKDAY(transaction_date) DAY)",
        date_filter=date_filter,
        params=params,
    )

    # ── CAT_COD Bulanan ───────────────────────────────────────────────
    _run_upsert(
        group_col="category_code",
        group_type_val="CAT_COD",
        frequency_val="M",
        period_expr="DATE_FORMAT(transaction_date, '%Y-%m-01')",
        date_filter=date_filter,
        params=params,
    )

    logger.info("sync_aggregations selesai (start_date=%s).", start_date)


def _run_upsert(
    group_col:      str,
    group_type_val: str,
    frequency_val:  str,
    period_expr:    str,
    date_filter:    str,
    params:         dict,
) -> None:
    """Jalankan satu blok INSERT ... ON DUPLICATE KEY UPDATE ke MySQL."""
    upsert_sql = text(f"""
        INSERT INTO sales_aggregations
            (period_date, group_type, group_code, frequency, total_qty, trx_count)
        SELECT
            {period_expr}                AS period_date,
            :group_type                  AS group_type,
            {group_col}                  AS group_code,
            :frequency                   AS frequency,
            SUM(quantity)                AS total_qty,
            COUNT(*)                     AS trx_count
        FROM raw_transactions
        WHERE 1=1 {date_filter}
        GROUP BY {period_expr}, {group_col}
        ON DUPLICATE KEY UPDATE
            total_qty  = VALUES(total_qty),
            trx_count  = VALUES(trx_count),
            updated_at = CURRENT_TIMESTAMP
    """)

    with get_db_session() as session:
        result = session.execute(
            upsert_sql,
            {**params, "group_type": group_type_val, "frequency": frequency_val},
        )
        logger.debug(
            "UPSERT MySQL: group_type=%s freq=%s rows_affected=%s",
            group_type_val, frequency_val, result.rowcount,
        )


# ===========================================================================
# READ: Kategori Tersedia
# ===========================================================================

def get_available_categories(group_type: str) -> List[str]:
    """Ambil daftar unik group_code dari sales_aggregations."""
    sql = text("""
        SELECT DISTINCT group_code
        FROM sales_aggregations
        WHERE group_type = :group_type
        ORDER BY group_code ASC
    """)
    eng = get_engine()
    with eng.connect() as conn:
        result = conn.execute(sql, {"group_type": group_type})
        return [row[0] for row in result]


# ===========================================================================
# READ: Deret Waktu untuk Model Peramalan
# ===========================================================================

def fetch_series(
    group_type: str,
    group_code: str,
    freq:       str,
) -> pd.DataFrame:
    """
    Kembalikan deret waktu penjualan format ['ds', 'y'] untuk
    langsung dikonsumsi oleh HybridForecastingEngine.
    """
    sql = text("""
        SELECT period_date AS ds, total_qty AS y
        FROM sales_aggregations
        WHERE group_type = :gt
          AND group_code  = :gc
          AND frequency   = :freq
        ORDER BY period_date ASC
    """)
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(sql, conn, params={"gt": group_type, "gc": group_code, "freq": freq})

    if df.empty:
        logger.warning(
            "fetch_series: tidak ada data untuk %s=%s freq=%s",
            group_type, group_code, freq,
        )
        return pd.DataFrame(columns=["ds", "y"])

    df["ds"] = pd.to_datetime(df["ds"])
    df["y"]  = df["y"].astype(float)
    return df.reset_index(drop=True)


# Alias untuk kompatibilitas
fetch_series_data = fetch_series


# ===========================================================================
# READ: Ringkasan Database untuk Sidebar
# ===========================================================================

def get_database_summary() -> dict:
    """Statistik ringkasan database untuk widget sidebar Streamlit."""
    sql = text("""
        SELECT
            COUNT(*)                      AS total_raw_transactions,
            MIN(transaction_date)         AS min_date,
            MAX(transaction_date)         AS max_date,
            COUNT(DISTINCT division_code) AS total_divisions,
            COUNT(DISTINCT category_code) AS total_categories
        FROM raw_transactions
    """)
    try:
        eng = get_engine()
        with eng.connect() as conn:
            row = conn.execute(sql).fetchone()
        return {
            "total_raw_transactions": int(row[0]) if row[0] else 0,
            "min_date":               row[1],
            "max_date":               row[2],
            "total_divisions":        int(row[3]) if row[3] else 0,
            "total_categories":       int(row[4]) if row[4] else 0,
        }
    except Exception as exc:
        logger.error("get_database_summary gagal: %s", exc)
        return {
            "total_raw_transactions": 0,
            "min_date": None, "max_date": None,
            "total_divisions": 0, "total_categories": 0,
        }


# ===========================================================================
# UTIL: Cek Duplikasi batch_id
# ===========================================================================

def batch_exists(batch_id: str) -> bool:
    """Cek apakah batch_id sudah pernah diinsert (anti-duplikasi seeding)."""
    sql = text("""
        SELECT EXISTS (
            SELECT 1 FROM raw_transactions WHERE batch_id = :batch_id LIMIT 1
        ) AS ex
    """)
    try:
        eng = get_engine()
        with eng.connect() as conn:
            result = conn.execute(sql, {"batch_id": batch_id}).scalar()
        return bool(result)
    except Exception:
        return False
