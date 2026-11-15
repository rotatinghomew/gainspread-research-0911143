import tempfile
import unittest
from pathlib import Path

from gainspread.panel import AnalysisError, build_panel_analysis
from gainspread.service import Application


def panel_catalog(tables):
    class FakeCatalog:
        fingerprint = "fake"

        def rows(self, name):
            return tables.get(name, [])

    return FakeCatalog()


def purchase(identifier, merchant, month, species, grade, quantity, price):
    return {"id": identifier, "merchant_id": merchant, "month": month,
            "species": species, "grade": grade, "quantity_kg": quantity,
            "unit_price": price}


def design(identifier, merchant, month, weight, observed, days=30, stratum="S0", cluster="K0"):
    return {"id": identifier, "merchant_id": merchant, "month": month,
            "weight": weight, "observed": observed, "days_open": days,
            "stratum": stratum, "cluster": cluster}


def merchants(identifier, group, entry="2025-01", exit_month=None):
    return {"id": identifier, "group": group, "entry_month": entry, "exit_month": exit_month}


def reference(month, price, species="带鱼", grade="中规格"):
    return {"id": f"{month}-{species}-{grade}", "month": month, "species": species,
            "grade": grade, "unit_price": price, "reference_kind": "区域同规格参考价"}


class PanelFormulaTests(unittest.TestCase):
    def setUp(self):
        base, current = "2026-01", "2026-02"
        self.tables = {
            "merchants": [
                merchants("A", "小商户"),
                merchants("B", "大商户", exit_month=current),
                merchants("C", "小商户", entry=current),
                merchants("D", "小商户"),
                merchants("E", "小商户"),
            ],
            "purchases": [
                purchase("A-1", "A", base, "带鱼", "中规格", 10, "8"),
                purchase("B-1", "B", base, "带鱼", "中规格", 100, "9"),
                purchase("A-2", "A", current, "带鱼", "中规格", 10, "12"),
                purchase("C-2", "C", current, "带鱼", "中规格", 5, "10"),
                purchase("E-2", "E", current, "带鱼", "中规格", 2, "9"),
            ],
            "price_reference": [reference(base, "10"), reference(current, "12")],
            "panel_design": [
                design("A-1", "A", base, 2, True),
                design("A-2", "A", current, 2, True),
                design("B-1", "B", base, 3, True),
                design("C-2", "C", current, 1, True),
                design("D-1", "D", base, 1, True),
                design("D-2", "D", current, 1, True),
                design("E-1", "E", base, 1, False),
                design("E-2", "E", current, 1, True),
            ],
        }
        self.catalog = panel_catalog(self.tables)

    def analyze(self, **options):
        return build_panel_analysis(self.catalog, base="2026-01", current="2026-02", **options)

    def test_merchant_savings_use_quantity_and_comparable_price(self):
        result = self.analyze()
        base = {row["merchant_id"]: row for row in result["base"]["merchants"]}
        self.assertAlmostEqual(base["A"]["savings"], 10 * (10 - 8))
        self.assertAlmostEqual(base["B"]["savings"], 100 * (10 - 9))
        self.assertEqual(result["base"]["aggregate"]["merchant_count"], 2)
        self.assertEqual(result["current"]["aggregate"]["merchant_count"], 3)

    def test_negative_savings_are_kept_not_deleted(self):
        result = self.analyze(reference_mode="frozen")
        current = [row for row in result["current"]["merchants"]]
        a_row = next(row for row in current if row["merchant_id"] == "A")
        self.assertLess(a_row["savings"], 0)
        summary = result["distribution_indicators"]["current"]["total_savings"]
        self.assertEqual(summary["negative_count"], 1)
        self.assertAlmostEqual(summary["min"], a_row["savings"])

    def test_nonresponse_month_is_not_a_zero_purchase_record(self):
        result = self.analyze()
        self.assertEqual(result["base"]["diagnostics"]["nonresponse_with_purchase_rows"], [])
        self.assertNotIn("E", {row["merchant_id"] for row in result["base"]["merchants"]})
        self.assertEqual(result["base"]["diagnostics"]["nonresponse_with_purchase_rows"], [])

    def test_observed_without_purchase_is_reported_not_zero(self):
        result = self.analyze()
        missing = result["base"]["diagnostics"]["observed_without_purchase_merchants"]
        self.assertIn("D", missing)
        self.assertNotIn("D", {row["merchant_id"] for row in result["base"]["merchants"]})

    def test_entrant_and_exit_are_classified_by_registration(self):
        result = self.analyze()
        pairs = result["comparison"]["matched_merchants"]
        # C entered between periods; B has a registered exit month at the start of current.
        self.assertIn("C", pairs["entered_with_registered_entry"])
        self.assertIn("B", pairs["exited_with_registered_exit"])
        # E appears only in current but entered long ago: incomplete, not an entrant.
        self.assertIn("E", pairs["missing_base_without_entry"])
        self.assertNotIn("E", pairs["entered_with_registered_entry"])
        # D is in the register but has no purchase rows in either period; it is
        # neither an entrant nor an exiter, and is not silently dropped.
        self.assertIn("D", pairs["registered_without_records"])
        self.assertNotIn("D", pairs["entered_with_registered_entry"])

    def test_common_merchant_comparison_excludes_flows(self):
        result = self.analyze()
        common = result["comparison"]["common_merchants"]
        self.assertEqual(common["merchant_count"], 1)
        self.assertAlmostEqual(common["base_total_savings"], 20)
        self.assertAlmostEqual(common["current_total_savings"], 0)
        self.assertAlmostEqual(common["weighted_mean_change"], -20)

    def test_decomposition_identity_holds_with_interaction(self):
        result = self.analyze()
        decomposition = result["comparison"]["decomposition"]
        total = decomposition["total_mean_change"]
        parts = (decomposition["composition"]["total"]
                 + decomposition["within"]["total"]
                 + decomposition["interaction"]["total"]
                 + decomposition["group_turnover"]["total"])
        self.assertAlmostEqual(parts, total, places=5)
        self.assertNotEqual(decomposition["interaction"]["total"], 0)
        self.assertAlmostEqual(decomposition["residual"], 0, places=5)
        # 大商户 only appears in the base period, so the whole group is turnover.
        self.assertAlmostEqual(decomposition["group_turnover"]["total"], -60)
        self.assertAlmostEqual(decomposition["composition"]["total"], 12)

    def test_between_group_gap_is_reported_separately(self):
        decomposition = self.analyze()["comparison"]["decomposition"]
        gap = decomposition["between_group_gap"]
        self.assertIn("base", gap)
        self.assertIn("current", gap)
        self.assertEqual(gap["base_pair"], gap["current_pair"])

    def test_reference_modes_answer_different_questions(self):
        matched = self.analyze(reference_mode="matched")
        frozen = self.analyze(reference_mode="frozen")
        pooled = self.analyze(reference_mode="pooled")
        current_matched = matched["current"]["aggregate"]["savings"]
        current_frozen = frozen["current"]["aggregate"]["savings"]
        self.assertNotAlmostEqual(current_matched, current_frozen)
        self.assertNotAlmostEqual(matched["base"]["aggregate"]["savings"],
                                  pooled["base"]["aggregate"]["savings"])

    def test_weight_choice_changes_weighted_totals_only(self):
        design_weighted = self.analyze(weight_field="weight_design")
        equal = self.analyze(weight_field="weight_equal")
        self.assertAlmostEqual(design_weighted["base"]["aggregate"]["savings"],
                               equal["base"]["aggregate"]["savings"])
        self.assertAlmostEqual(design_weighted["base"]["aggregate"]["weighted_savings"],
                               20 * 2 + 100 * 3)
        self.assertAlmostEqual(equal["base"]["aggregate"]["weighted_savings"], 120)
        self.assertEqual(equal["base"]["weight_attachment"]["without_design_weight"], 0)

    def test_extreme_merchant_sensitivity_removes_largest_buyer(self):
        result = self.analyze()
        current = result["sensitivity"]["extreme_merchants"]["base"]
        top_one = current["scenarios"][0]
        self.assertEqual(top_one["removed_count"], 1)
        self.assertAlmostEqual(top_one["total_savings"], 20)

    def test_missing_month_sensitivity_separates_complete_merchants(self):
        result = self.analyze()
        missing = result["sensitivity"]["missing_months"]
        self.assertEqual(missing["span_months"], ["2026-01", "2026-02"])
        self.assertGreaterEqual(missing["merchant_counts"]["complete_months"], 1)
        self.assertGreaterEqual(missing["merchant_counts"]["partial_or_no_purchase"], 1)
        self.assertNotEqual(missing["complete_months"]["aggregate"]["merchant_count"],
                            missing["all_operable"]["aggregate"]["merchant_count"])

    def test_gini_is_inapplicable_with_negative_savings(self):
        result = self.analyze(reference_mode="frozen")
        gini = result["distribution_indicators"]["gini"]["current_total_savings"]
        self.assertFalse(gini["applicable"])
        self.assertIsNone(gini["value"])
        self.assertTrue(gini["reason"])

    def test_bottom_coverage_exposes_both_definitions(self):
        tables = dict(self.tables)
        tables["merchants"] = [
            merchants("A", "小商户"),
            merchants("B", "大商户"),
            merchants("G", "大商户"),
        ]
        tables["purchases"] = [
            purchase("A-1", "A", "2026-01", "带鱼", "中规格", 10, "8"),
            purchase("B-1", "B", "2026-01", "带鱼", "中规格", 100, "9"),
            purchase("G-1", "G", "2026-01", "带鱼", "中规格", 200, "11"),
            purchase("A-2", "A", "2026-02", "带鱼", "中规格", 10, "8"),
        ]
        tables["price_reference"] = [reference("2026-01", "10"), reference("2026-02", "10")]
        tables["panel_design"] = [
            design("A-1", "A", "2026-01", 2, True),
            design("B-1", "B", "2026-01", 3, True),
            design("G-1", "G", "2026-01", 1, True),
            design("A-2", "A", "2026-02", 2, True),
        ]
        result = build_panel_analysis(panel_catalog(tables), base="2026-01",
                                      current="2026-02")
        base_bottom = result["base"]["overall"]["bottom_coverage"]
        self.assertIn("by_savings", base_bottom)
        self.assertIn("by_spend", base_bottom)
        self.assertIn("by_quantity", base_bottom)
        self.assertIn("大商户", base_bottom["by_savings"]["groups"])
        self.assertEqual(base_bottom["by_spend"]["groups"], {"小商户": 1})

    def test_entered_and_exited_between_periods_is_reported(self):
        tables = {
            "merchants": [
                merchants("A", "小商户"),
                merchants("F", "小商户", entry="2026-02", exit_month="2026-03"),
            ],
            "purchases": [
                purchase("A-1", "A", "2026-01", "带鱼", "中规格", 10, "8"),
                purchase("A-3", "A", "2026-03", "带鱼", "中规格", 10, "8"),
            ],
            "price_reference": [reference("2026-01", "10"),
                                reference("2026-02", "10"),
                                reference("2026-03", "10")],
            "panel_design": [
                design("A-1", "A", "2026-01", 1, True),
                design("A-3", "A", "2026-03", 1, True),
                design("F-2", "F", "2026-02", 1, True),
            ],
        }
        result = build_panel_analysis(panel_catalog(tables), base="2026-01",
                                      current="2026-03")
        pairs = result["comparison"]["matched_merchants"]
        self.assertIn("F", pairs["entered_and_exited_between_periods"])
        self.assertNotIn("F", pairs["entered"])
        self.assertNotIn("F", pairs["exited"])
        self.assertEqual(pairs["counts"]["entered_and_exited_between_periods"], 1)

    def test_design_structure_summarises_panel_without_duplicating_merchants(self):
        result = self.analyze()
        structure = result["base"]["design_structure"]
        self.assertEqual(structure["merchants"], result["base"]["merchant_count"])
        self.assertEqual(structure["observed_design_rows"],
                         sum(row["observed_month_count"] for row in result["base"]["merchants"]))
        self.assertGreaterEqual(structure["clusters"], 1)

    def test_operating_days_sensitivity_reports_scale_scaled_values(self):
        result = self.analyze()
        days = result["sensitivity"]["operating_days"]
        self.assertIn("observed", days)
        self.assertIn("full_month_equivalent", days)
        self.assertAlmostEqual(days["observed"]["quantity"], 127)
        # Every month in this fixture has days_open=30, so the equivalent scale is 1.
        self.assertAlmostEqual(days["full_month_equivalent"]["quantity"], 127)
        self.assertAlmostEqual(days["observed"]["savings"],
                               days["full_month_equivalent"]["savings"])

    def test_invalid_parameters_raise(self):
        with self.assertRaises(AnalysisError):
            build_panel_analysis(self.catalog, base="2026-01", current="2026-01")
        with self.assertRaises(AnalysisError):
            build_panel_analysis(self.catalog, base="2026-01", current="2026-02",
                                 reference_mode="unknown")
        with self.assertRaises(AnalysisError):
            build_panel_analysis(self.catalog, base="2026-01", current="2026-02",
                                 bottom_quantile=1.5)
        with self.assertRaises(AnalysisError):
            build_panel_analysis(self.catalog, base="2026-03", current="2026-02")


class PanelServiceTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.application = Application("data", Path(self.workspace.name) / "state")

    def test_panel_endpoint(self):
        status, result = self.application.dispatch("GET", "/panel")
        self.assertEqual(status, 200)
        self.assertIn("base", result)
        self.assertIn("distribution_indicators", result)
        self.assertGreater(result["base"]["aggregate"]["merchant_count"], 0)

    def test_panel_endpoint_rejects_bad_parameters(self):
        self.assertEqual(self.application.dispatch("GET", "/panel?reference=bad")[0], 400)
        self.assertEqual(self.application.dispatch("GET", "/panel?bottom=1.5")[0], 400)
        self.assertEqual(self.application.dispatch("GET", "/panel?base=2026-07&current=2026-07")[0], 400)

    def test_real_data_keeps_negative_savings_and_reports_overlap(self):
        status, result = self.application.dispatch(
            "GET", "/panel?base=2026-07&current=2026-10")
        self.assertEqual(status, 200)
        current = result["distribution_indicators"]["current"]["total_savings"]
        self.assertGreater(current["negative_count"], 0)
        self.assertTrue(result["distribution_indicators"]["changes"]["overlap"]["applicable"])
        self.assertFalse(result["distribution_indicators"]["gini"][
            "current_total_savings"]["applicable"])


if __name__ == "__main__":
    unittest.main()
