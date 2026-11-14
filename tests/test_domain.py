import unittest
from gainspread.catalog import Catalog
from gainspread.domain import observed_spending


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog("data")

    def test_observed_spending_preserves_amounts(self):
        self.assertGreater(sum(float(value) for value in observed_spending(self.catalog).values()), 0)
        self.assertTrue(any(not record["observed"] for record in self.catalog.rows("panel_design")))


    def test_exact_spending_on_small_purchase(self):
        records = [{"merchant_id": "A", "quantity_kg": .1, "unit_price": "0.20"},
                   {"merchant_id": "A", "quantity_kg": .2, "unit_price": "0.30"}]
        self.assertEqual(observed_spending(FakeCatalog({"purchases": records}))["A"], "0.080")

    def test_empty_purchases_do_not_create_nonbuyers(self):
        self.assertEqual(observed_spending(FakeCatalog({"purchases": []})), {})

    def test_merchant_identity_is_retained(self):
        records = [{"merchant_id": identifier, "quantity_kg": 1, "unit_price": "1.00"} for identifier in ("A", "B")]
        self.assertEqual(set(observed_spending(FakeCatalog({"purchases": records}))), {"A", "B"})

    def test_design_has_cluster_and_stratum(self):
        records = self.catalog.rows("panel_design")
        self.assertGreater(len({record["cluster"] for record in records}), 1)
        self.assertGreater(len({record["stratum"] for record in records}), 1)


class FakeCatalog:
    def __init__(self, tables):
        self.tables = tables

    def rows(self, name):
        return self.tables[name]
