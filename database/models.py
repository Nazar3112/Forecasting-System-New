"""
database/models.py
==================
Definisi ORM SQLAlchemy untuk MySQL — dua tabel utama sistem forecasting.

DDL yang dihasilkan oleh ORM ini identik dengan skema pada spesifikasi:
  - raw_transactions   : Log transaksi kasir granular (InnoDB, utf8mb4)
  - sales_aggregations : Deret waktu teragregasi (PK komposit 4 kolom)

Driver: PyMySQL (mysql+pymysql) — tanpa kompilasi C++
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    BIGINT, DATE, DOUBLE, Index, INTEGER,
    NUMERIC, TIMESTAMP, VARCHAR, func, text,
)
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base


# ===========================================================================
# Tabel 1: raw_transactions
# Log transaksi kasir POS granular — diisi via seed_historical.py
# dan diperbarui inkremental via upload delta CSV bulanan.
# ===========================================================================
class RawTransaction(Base):
    """
    Model ORM tabel `raw_transactions` (MySQL InnoDB, ENGINE=utf8mb4).

    Indexes:
      idx_trx_date_div : mempercepat query per divisi + rentang tanggal
      idx_trx_date_cat : mempercepat query per kategori + rentang tanggal
    """

    __tablename__ = "raw_transactions"

    # Primary Key
    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)

    # Atribut Transaksi
    transaction_date: Mapped[date]         = mapped_column(DATE,          nullable=False)
    division_code:    Mapped[str]          = mapped_column(VARCHAR(20),   nullable=False)
    category_code:    Mapped[str]          = mapped_column(VARCHAR(20),   nullable=False)
    product_code:     Mapped[str | None]   = mapped_column(VARCHAR(50),   nullable=True)
    quantity:         Mapped[int]          = mapped_column(INTEGER,        nullable=False)
    price:            Mapped[float | None] = mapped_column(NUMERIC(12, 2), nullable=True)
    transaction_type: Mapped[str]          = mapped_column(VARCHAR(5),    server_default="J")

    # Metadata Ingesti
    batch_id:   Mapped[str]      = mapped_column(VARCHAR(100),  nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, server_default=func.current_timestamp()
    )

    # Index komposit untuk query efisien
    __table_args__ = (
        Index("idx_trx_date_div", "transaction_date", "division_code"),
        Index("idx_trx_date_cat", "transaction_date", "category_code"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    def __repr__(self) -> str:
        return (
            f"<RawTransaction id={self.id} "
            f"date={self.transaction_date} "
            f"div={self.division_code} qty={self.quantity}>"
        )


# ===========================================================================
# Tabel 2: sales_aggregations
# Deret waktu teragregasi per kategori — dibaca langsung oleh model.
# Data ~160 baris per kategori/freq → query < 0.1 detik.
#
# UPSERT menggunakan MySQL native: ON DUPLICATE KEY UPDATE
# ===========================================================================
class SalesAggregation(Base):
    """
    Model ORM tabel `sales_aggregations` (MySQL InnoDB).

    Primary Key Komposit (4 kolom):
      (period_date, group_type, group_code, frequency)
    Memastikan tidak ada duplikasi per periode + kategori + frekuensi.

    Kolom:
      period_date : Senin untuk mingguan, hari-1 untuk bulanan
      group_type  : 'DIV' atau 'CAT_COD'
      group_code  : Kode divisi/kategori (misal '04', '010414')
      frequency   : 'W' (mingguan) atau 'M' (bulanan)
      total_qty   : Total kuantitas penjualan pada periode tersebut
      trx_count   : Jumlah transaksi granular yang diringkas
      updated_at  : Diperbarui otomatis saat UPSERT (ON UPDATE CURRENT_TIMESTAMP)
    """

    __tablename__ = "sales_aggregations"

    # Primary Key Komposit (4 kolom)
    period_date: Mapped[date] = mapped_column(DATE,       primary_key=True)
    group_type:  Mapped[str]  = mapped_column(VARCHAR(10), primary_key=True)  # 'DIV' | 'CAT_COD'
    group_code:  Mapped[str]  = mapped_column(VARCHAR(20), primary_key=True)
    frequency:   Mapped[str]  = mapped_column(VARCHAR(2),  primary_key=True)  # 'W' | 'M'

    # Nilai Agregat
    total_qty:  Mapped[float] = mapped_column(DOUBLE,   nullable=False)
    trx_count:  Mapped[int]   = mapped_column(INTEGER,  nullable=False)

    # Timestamp otomatis diperbarui saat INSERT atau UPDATE
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
    )

    # Index untuk mempercepat lookup forecasting
    __table_args__ = (
        Index("idx_agg_lookup", "group_type", "group_code", "frequency", "period_date"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    def __repr__(self) -> str:
        return (
            f"<SalesAggregation {self.group_type}={self.group_code} "
            f"freq={self.frequency} period={self.period_date} qty={self.total_qty:.0f}>"
        )
