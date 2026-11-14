from collections import defaultdict
from decimal import Decimal


def observed_spending(catalog):
    totals = defaultdict(Decimal)
    for record in catalog.rows("purchases"):
        totals[record["merchant_id"]] += Decimal(str(record["quantity_kg"])) * Decimal(record["unit_price"])
    return {merchant: str(value) for merchant, value in sorted(totals.items())}


def overview(catalog):
    totals = observed_spending(catalog)
    return {"商户数": len(catalog.rows("merchants")), "有采购记录商户": len(totals), "已观察采购支出": str(sum((Decimal(value) for value in totals.values()), Decimal(0))), "未回收月数": sum(not record["observed"] for record in catalog.rows("panel_design"))}
