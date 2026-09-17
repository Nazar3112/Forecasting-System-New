"""
tests/test_database.py
======================
Unit tests untuk lapisan database MySQL (PyMySQL):
  - Konfigurasi koneksi & check_connection()
  - ORM Models (RawTransaction, SalesAggregation)
  - Repository helpers
"""

from __future__ import annotations

import os
from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Test: Database Connection & Base
# ---------------------------------------------------------------------------

class TestDatabaseConnection:
    def test_check_connection_signature(self):
        """check_connection harus mengembalikan tuple (bool, str)."""
        from database.connection import check_connection
        result = check_connection()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_base_exists(self):
        """Base ORM deklaratif harus tersedia."""
        from database.connection import Base
        assert Base is not None

    def test_init_database_callable(self):
        """init_database harus dapat dipanggil."""
        from database.connection import init_database
        assert callable(init_database)


# ---------------------------------------------------------------------------
# Test: ORM Models
# ---------------------------------------------------------------------------

class TestOrmModels:
    def test_raw_transaction_tablename(self):
        from database.models import RawTransaction
        assert RawTransaction.__tablename__ == "raw_transactions"

    def test_sales_aggregation_tablename(self):
        from database.models import SalesAggregation
        assert SalesAggregation.__tablename__ == "sales_aggregations"

    def test_raw_transaction_columns(self):
        from database.models import RawTransaction
        columns = {c.key for c in RawTransaction.__table__.columns}
        required = {"id", "transaction_date", "division_code", "category_code", "quantity", "batch_id"}
        assert required.issubset(columns)

    def test_sales_aggregation_composite_pk(self):
        from database.models import SalesAggregation
        pk_cols = {c.key for c in SalesAggregation.__table__.primary_key.columns}
        expected = {"period_date", "group_type", "group_code", "frequency"}
        assert pk_cols == expected

    def test_raw_transaction_repr(self):
        from database.models import RawTransaction
        rt = RawTransaction(
            id=1, transaction_date=date(2024, 1, 15),
            division_code="04", quantity=10, batch_id="TEST",
        )
        assert "RawTransaction" in repr(rt)

    def test_sales_aggregation_repr(self):
        from database.models import SalesAggregation
        sa = SalesAggregation(
            period_date=date(2024, 1, 1),
            group_type="DIV", group_code="04",
            frequency="W", total_qty=1500.0, trx_count=20,
        )
        assert "SalesAggregation" in repr(sa)


# ---------------------------------------------------------------------------
# Test: Repository Functions (Unit Mock)
# ---------------------------------------------------------------------------

class TestRepositoryUnit:
    def test_fast_bulk_insert_raw_empty_df(self):
        from database import repository
        assert repository.fast_bulk_insert_raw(pd.DataFrame(), batch_id="TEST") == 0

    def test_get_database_summary_on_error(self):
        from database import repository
        with patch("database.repository.get_engine") as mock_get_engine:
            mock_get_engine.side_effect = Exception("MySQL down")
            res = repository.get_database_summary()
        assert isinstance(res, dict)
        assert res["total_raw_transactions"] == 0
