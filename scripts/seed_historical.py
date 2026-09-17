"""
scripts/seed_historical.py
==========================
Skrip CLI untuk memuat dataset master historis (~686 MB) ke MySQL
secara efisien menggunakan chunked read dan batch insert.

Penggunaan:
    python -m scripts.seed_historical --file merged_MTRAN_with_desc.csv

Alur:
    1. init_database()        → Auto-create DB + tabel (tanpa phpMyAdmin)
    2. Baca CSV chunked        → chunksize=50.000 baris (hemat RAM)
    3. Filter & transform      → QTY > 0, parse tanggal DD-MM-YYYY
    4. Insert ke MySQL         → chunksize=25.000 per INSERT batch
    5. sync_aggregations()     → Bangun sales_aggregations pertama kali
    6. Laporan waktu           → Total baris, durasi eksekusi

Catatan:
    Skrip aman dijalankan ulang — duplikasi dicegah via batch_id check.
    Jika 'INITIAL_SEED_3YEARS' sudah ada, skrip berhenti dengan peringatan.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Tambahkan root project ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.connection import init_database, check_connection
from database import repository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seed_historical")

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
BATCH_ID         = "INITIAL_SEED_3YEARS"
CSV_CHUNK_SIZE   = 50_000    # Baris per chunk saat baca CSV (hemat RAM)
DB_CHUNK_SIZE    = 2_500     # Baris per batch INSERT MySQL (multi-insert safe for max_allowed_packet)
DATE_FORMAT      = "%d-%m-%Y"
REQUIRED_COLUMNS = {"TANGGAL", "DIV", "CAT_COD", "QTY"}


# ---------------------------------------------------------------------------
# Fungsi Utama
# ---------------------------------------------------------------------------

def seed(file_path: str) -> None:
    """
    Proses seeding data historis dari CSV ke MySQL.

    Parameters
    ----------
    file_path : str  Path berkas CSV historis (~686 MB).
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("Berkas tidak ditemukan: %s", path)
        sys.exit(1)

    # ── 1. Inisialisasi database & tabel ────────────────────────────────
    logger.info("Menginisialisasi database MySQL …")
    try:
        init_database()
    except Exception as exc:
        logger.error(
            "Gagal terhubung ke MySQL: %s\n"
            "Pastikan:\n"
            "  1. MySQL berjalan (XAMPP → Start MySQL)\n"
            "  2. File .env sudah dikonfigurasi dengan benar", exc
        )
        sys.exit(1)

    ok, msg = check_connection()
    logger.info("%s", msg)

    # ── 2. Cek duplikasi seeding ─────────────────────────────────────────
    if repository.batch_exists(BATCH_ID):
        logger.warning(
            "Batch '%s' sudah ada di database. Seeding dibatalkan.", BATCH_ID
        )
        logger.warning("Hapus baris dengan batch_id='%s' untuk re-seed.", BATCH_ID)
        sys.exit(0)

    # ── 3. Baca & insert CSV secara chunked ──────────────────────────────
    logger.info("Memulai seeding dari: %s (chunk=%d baris)", path, CSV_CHUNK_SIZE)
    start_time   = time.perf_counter()
    total_raw    = 0
    total_insert = 0
    chunk_num    = 0

    def _usecols(col: str) -> bool:
        return col.strip("'").strip() in {
            "TANGGAL", "RTYPE", "DIV", "CAT_COD", "QTY", "PRDCD", "PRICE", "HARGA"
        }

    reader = pd.read_csv(
        path,
        usecols=_usecols,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8",
    )

    for chunk_df in reader:
        chunk_num  += 1
        raw_count   = len(chunk_df)
        total_raw  += raw_count

        # Normalisasi nama kolom (strip kutip/spasi)
        chunk_df.columns = [c.strip("'").strip() for c in chunk_df.columns]

        # Sanitasi nilai string
        str_cols = chunk_df.select_dtypes(include="object").columns
        for col in str_cols:
            chunk_df[col] = chunk_df[col].astype(str).str.strip("'").str.strip()

        # Filter RTYPE == 'J'
        if "RTYPE" in chunk_df.columns:
            chunk_df = chunk_df[chunk_df["RTYPE"] == "J"].copy()

        # Validasi kolom wajib
        available = set(chunk_df.columns)
        missing   = REQUIRED_COLUMNS - available
        if missing:
            logger.warning("Chunk %d: kolom wajib tidak ada %s, dilanjutkan.", chunk_num, missing)
            continue

        # Rename HARGA → PRICE jika ada
        if "HARGA" in chunk_df.columns and "PRICE" not in chunk_df.columns:
            chunk_df = chunk_df.rename(columns={"HARGA": "PRICE"})

        # Filter QTY > 0
        chunk_df["QTY"] = pd.to_numeric(chunk_df["QTY"], errors="coerce")
        chunk_df = chunk_df[chunk_df["QTY"] > 0].copy()
        chunk_df = chunk_df.dropna(subset=["QTY", "DIV", "CAT_COD", "TANGGAL"]).copy()

        if chunk_df.empty:
            continue

        # Parse tanggal DD-MM-YYYY → datetime (MySQL butuh format ISO)
        chunk_df["TANGGAL"] = pd.to_datetime(
            chunk_df["TANGGAL"], format=DATE_FORMAT, errors="coerce"
        )
        chunk_df = chunk_df.dropna(subset=["TANGGAL"]).copy()

        if chunk_df.empty:
            continue

        # Insert ke MySQL (batch chunksize=25.000)
        n = repository.fast_bulk_insert_raw(
            chunk_df, batch_id=BATCH_ID, chunksize=DB_CHUNK_SIZE
        )
        total_insert += n

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Chunk %4d | CSV: %7d | Valid: %7d | Total insert: %8d | %.1fs",
            chunk_num, raw_count, len(chunk_df), total_insert, elapsed,
        )

    # ── 4. Sinkronisasi agregasi ─────────────────────────────────────────
    logger.info("Membangun tabel sales_aggregations …")
    sync_start = time.perf_counter()
    repository.sync_aggregations()
    logger.info("sync_aggregations selesai dalam %.1fs.", time.perf_counter() - sync_start)

    # ── 5. Laporan akhir ─────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - start_time
    summary = repository.get_database_summary()

    logger.info("=" * 60)
    logger.info("SEEDING SELESAI")
    logger.info("  Berkas         : %s", path.name)
    logger.info("  Total CSV raw  : %d baris", total_raw)
    logger.info("  Berhasil insert: %d baris", total_insert)
    logger.info("  Waktu total    : %.1f detik (%.1f menit)", total_elapsed, total_elapsed / 60)
    logger.info("  Rentang data   : %s s.d. %s", summary["min_date"], summary["max_date"])
    logger.info("  Divisi         : %d | Kategori: %d", summary["total_divisions"], summary["total_categories"])
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed data historis CSV (~686 MB) ke MySQL db_forecasting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prasyarat:
  1. MySQL berjalan (XAMPP → Start MySQL)
  2. File .env sudah dikonfigurasi (lihat .env.example)

Contoh:
  python -m scripts.seed_historical --file data/merged_MTRAN_with_desc.csv
  python -m scripts.seed_historical --file D:/data/MTRAN_2021_2024.csv
        """,
    )
    parser.add_argument(
        "--file", required=True, metavar="PATH",
        help="Path ke berkas CSV log transaksi kasir historis.",
    )
    args = parser.parse_args()
    seed(args.file)


if __name__ == "__main__":
    main()
