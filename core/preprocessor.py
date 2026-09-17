"""
core/preprocessor.py
====================
Kelas DeltaPreprocessor: Validasi dan pembersihan berkas CSV delta
(transaksi bulanan baru) yang diunggah via antarmuka Streamlit.

Fungsi ini TIDAK memuat file besar ke memori —
hanya memproses file delta upload kecil (~15–25 MB).
Output: DataFrame bersih siap dioper ke repository.fast_bulk_insert_raw().
"""

from __future__ import annotations

import io
import logging
from typing import Tuple

import pandas as pd

from core.config import REQUIRED_DELTA_COLUMNS, DATE_FORMAT, VALID_RTYPE

logger = logging.getLogger(__name__)


class DeltaPreprocessor:
    """
    Validasi dan bersihkan berkas CSV delta (upload bulanan baru).

    Alur:
        1. Baca CSV dari file-like object (Streamlit uploader)
        2. Validasi kolom wajib: TANGGAL, QTY, DIV, CAT_COD
        3. Sanitasi string (strip kutip)
        4. Filter RTYPE == 'J' (jika ada)
        5. Filter QTY > 0 (buang retur/batal kasir: QTY <= 0)
        6. Parse tanggal DD-MM-YYYY → datetime
        7. Return DataFrame bersih untuk dikirim ke database
    """

    def clean_delta(
        self,
        source: str | io.BytesIO | io.StringIO,
    ) -> pd.DataFrame:
        """
        Pipeline lengkap validasi + pembersihan berkas delta.

        Parameters
        ----------
        source : file-like dari st.file_uploader atau path string.

        Returns
        -------
        pd.DataFrame  Kolom: ['TANGGAL', 'DIV', 'CAT_COD', 'QTY', 'PRDCD'(opt), 'PRICE'(opt)]

        Raises
        ------
        ValueError  Jika kolom wajib tidak ada atau tidak ada data valid.
        """
        # Baca CSV
        df = self._read_csv(source)
        df.columns = [c.strip("'").strip() for c in df.columns]

        # Validasi skema
        valid, missing = self.validate_schema(df)
        if not valid:
            raise ValueError(
                f"Kolom wajib tidak ditemukan: {missing}. "
                "Pastikan CSV memuat: " + ", ".join(REQUIRED_DELTA_COLUMNS)
            )

        # Sanitasi string
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip("'").str.strip()

        # Filter RTYPE
        if "RTYPE" in df.columns:
            df = df[df["RTYPE"] == VALID_RTYPE].copy()

        # Filter QTY > 0 (buang retur: QTY <= 0)
        df["QTY"] = pd.to_numeric(df["QTY"], errors="coerce")
        df = df[df["QTY"] > 0].copy()
        df = df.dropna(subset=["QTY", "DIV", "CAT_COD"]).copy()
        df = df[df["DIV"].str.strip() != ""].copy()
        df = df[df["CAT_COD"].str.strip() != ""].copy()

        # Parse tanggal (format DD-MM-YYYY utama, fallback otomatis)
        df["TANGGAL"] = self._parse_dates_safe(df["TANGGAL"])
        n_failed = df["TANGGAL"].isna().sum()
        if n_failed > 0:
            logger.warning("%d baris gagal parse tanggal, dibuang.", n_failed)
            df = df.dropna(subset=["TANGGAL"])

        if df.empty:
            raise ValueError(
                "Tidak ada data valid setelah pembersihan. "
                "Periksa format tanggal (DD-MM-YYYY) dan nilai QTY."
            )

        logger.info("DeltaPreprocessor selesai: %d baris bersih.", len(df))
        return df.reset_index(drop=True)

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, list[str]]:
        """Periksa keberadaan kolom wajib."""
        existing = set(df.columns)
        missing  = [c for c in REQUIRED_DELTA_COLUMNS if c not in existing]
        return (len(missing) == 0, missing)

    def get_delta_summary(self, df: pd.DataFrame) -> dict:
        """Ringkasan singkat DataFrame delta bersih untuk konfirmasi ke pengguna."""
        return {
            "total_baris":    len(df),
            "tanggal_awal":   df["TANGGAL"].min(),
            "tanggal_akhir":  df["TANGGAL"].max(),
            "total_qty":      int(df["QTY"].sum()),
            "total_divisi":   df["DIV"].nunique(),
            "total_kategori": df["CAT_COD"].nunique(),
        }

    @staticmethod
    def _read_csv(source) -> pd.DataFrame:
        def _usecols(col: str) -> bool:
            return col.strip("'").strip() in {
                "TANGGAL", "RTYPE", "DIV", "CAT_COD", "QTY", "PRDCD", "PRICE", "HARGA"
            }
        try:
            df = pd.read_csv(source, usecols=_usecols, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, usecols=_usecols, low_memory=False, encoding="cp1252")
        logger.info("Delta CSV dibaca: %d baris, %d kolom", len(df), len(df.columns))
        return df

    @staticmethod
    def _parse_dates_safe(series: pd.Series) -> pd.Series:
        """Parse tanggal dengan format utama DD-MM-YYYY, fallback otomatis."""
        parsed = pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")
        mask_failed = parsed.isna()
        if mask_failed.any():
            parsed[mask_failed] = pd.to_datetime(
                series[mask_failed], errors="coerce", dayfirst=True
            )
        return parsed
