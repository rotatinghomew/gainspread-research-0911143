"""经营面板的商户层节省额分析。

本模块只做描述性分析，不建立反事实，也不把采购支出下降当作节省。

关键口径：

* 采购行节省额 = 数量（公斤） × （可比参考价 − 实付价）；负节省额保留。
* 商户在期间内的节省额先按商户汇总，同一商户跨月只形成一条期间记录，
  避免重复观测被当成多个独立商户。
* 采购量、支出和节省额分别报告。支出下降可能来自少买，不能据此判断受益。
* 样本权重来自 ``panel_design.weight``，是抽样权重，不是采购量。
* 参考价可按当月、按基期冻结或按全样本合并；三者含义不同，均非反事实。
* 分布指标若在数学上不适用（例如含负值的基尼系数），明确返回原因而不是
  给出一个容易被误读的数值。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
from math import ceil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ZERO = Decimal(0)

REFERENCE_MODES: dict[str, str] = {
    "matched": "使用与采购月份相同的同品规参考价；结果会同时反映参考价随月份的变化。",
    "frozen": "把每个品规的参考价冻结在基期月份的平均水平；重点看商户实付价相对固定参考价的变化。",
    "pooled": "使用分析期内全部月份的同品规平均参考价；压低月份间参考价波动，但会混合不同月份的市场水平。",
}

DEFAULT_BOTTOM_QUANTILE = 0.2
FULL_MONTH_DAYS = 30
DISTRIBUTION_BINS = 20

CAVEATS = [
    "节省额是相对参考价的采购成本差，不是利润、销售额或商户净收益。",
    "采购支出下降可能来自少买；采购量、支出与节省额分开报告，少买不称为受益。",
    "参考价模式不同会改变结论：matched 含参考价时序变化，frozen 固定基期参考价，pooled 混合分析期市场水平。",
    "未观察月份不按零采购处理，也不为缺失月份虚构采购行。",
    "负节省额保留在商户计数、总量和分布中，没有静默剔除。",
    "按商户汇总时同一商户跨月只计一次；期间值为其可观测月份的合计。",
    "设计权重是抽样权重，不是采购量或经营日数。",
    "未回收月不等于退出；退出月是不再经营的首月，两者分开报告。",
    "基尼系数等指标只对非负分布有直接解释；负节省额下相关指标返回不适用及原因。",
    "不发布单一“普惠程度”指数：负节省额和分布重叠会使一个有数学取值但经济含义误导的指数被当成结论。",
    "组均值分解只描述均值层面的构成、组内、交互与组出现/消失；组间差距单列；分布重叠和分位数变化另行报告，不能据此推断每个商户受益。",
]


class AnalysisError(ValueError):
    """分析参数或资料不足以完成请求。"""


# ---------------------------------------------------------------------------
# 数值与序列化小工具
# ---------------------------------------------------------------------------

def _dec(value: Any) -> Decimal:
    return Decimal(str(value))


def _num(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _metric(value: Any, applicable: bool = True, reason: str | None = None,
            digits: int = 6) -> dict:
    return {
        "value": _num(value, digits) if applicable and value is not None else None,
        "applicable": bool(applicable),
        "reason": None if applicable else reason,
    }


def _share(count: float, total: float) -> float | None:
    return (count / total) if total else None


def _ratio_metric(numerator, denominator) -> dict:
    if numerator is None or denominator is None:
        return _metric(None, False, "分子或分母缺失，比值不适用。")
    if denominator <= 0 or numerator < 0:
        return _metric(None, False, "负节省额下 p90/p10 比值会变号或没有解释意义，未计算。")
    return _metric(numerator / denominator)


# ---------------------------------------------------------------------------
# 权重、分位数与分布摘要
# ---------------------------------------------------------------------------

def _weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    if not values:
        raise AnalysisError("空样本没有分位数")
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(values_array)) or not np.all(np.isfinite(weights_array)):
        raise AnalysisError("分布包含非有限数值")
    if weights_array.sum() <= 0:
        raise AnalysisError("样本权重合计必须为正")
    order = np.argsort(values_array, kind="mergesort")
    sorted_values = values_array[order]
    sorted_weights = weights_array[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    index = int(np.searchsorted(cumulative, q, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    return _weighted_quantile(values, weights, 0.5)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float | None:
    total = float(np.sum(weights))
    if total <= 0:
        return None
    return float(np.dot(np.asarray(values, dtype=float), np.asarray(weights, dtype=float)) / total)


def _weighted_std(values: Sequence[float], weights: Sequence[float]) -> float | None:
    mean = _weighted_mean(values, weights)
    total = float(np.sum(weights))
    if mean is None or total <= 0:
        return None
    variance = float(np.sum(np.asarray(weights, dtype=float) *
                            (np.asarray(values, dtype=float) - mean) ** 2) / total)
    return variance ** 0.5


def _distribution_summary(values: Sequence[float], weights: Sequence[float],
                          weight_basis: str) -> dict:
    values_list = [float(value) for value in values]
    weights_list = [float(weight) for weight in weights]
    if not values_list:
        return {"count": 0, "weight_basis": weight_basis}
    mean = _weighted_mean(values_list, weights_list)
    median = _weighted_median(values_list, weights_list)
    deviations = [abs(value - median) for value in values_list]
    mad = _weighted_median(deviations, weights_list)
    q10 = _weighted_quantile(values_list, weights_list, 0.10)
    q25 = _weighted_quantile(values_list, weights_list, 0.25)
    q75 = _weighted_quantile(values_list, weights_list, 0.75)
    q90 = _weighted_quantile(values_list, weights_list, 0.90)
    positive_weight = sum(weight for value, weight in zip(values_list, weights_list) if value > 0)
    zero_weight = sum(weight for value, weight in zip(values_list, weights_list) if value == 0)
    negative_weight = sum(weight for value, weight in zip(values_list, weights_list) if value < 0)
    total_weight = sum(weights_list)
    std = _weighted_std(values_list, weights_list)
    return {
        "count": len(values_list),
        "weight_basis": weight_basis,
        "weight_total": _num(total_weight),
        "mean": _num(mean),
        "median": _num(median),
        "p10": _num(q10),
        "p25": _num(q25),
        "p75": _num(q75),
        "p90": _num(q90),
        "iqr": _num(q75 - q25),
        "mad": _num(mad),
        "min": _num(min(values_list)),
        "max": _num(max(values_list)),
        "std": _num(std),
        "coefficient_of_variation": _metric(
            std / mean if mean not in (None, 0) and mean > 0 else None,
            applicable=mean not in (None, 0) and mean > 0,
            reason="均值为零或为负时变异系数会放大或变号，没有解释意义。",
        ),
        "positive_count": sum(1 for value in values_list if value > 0),
        "negative_count": sum(1 for value in values_list if value < 0),
        "zero_count": sum(1 for value in values_list if value == 0),
        "positive_share": _num(_share(positive_weight, total_weight)),
        "negative_share": _num(_share(negative_weight, total_weight)),
        "zero_share": _num(_share(zero_weight, total_weight)),
    }


def _gini(values: Sequence[float], weights: Sequence[float]) -> dict:
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    if values_array.size == 0:
        return _metric(None, False, "空样本不计算基尼系数。")
    if np.any(values_array < 0):
        return _metric(None, False, "样本含负值；基尼系数只对非负变量有解释，未计算。")
    total_weight = float(weights_array.sum())
    if total_weight <= 0:
        return _metric(None, False, "样本权重合计非正，未计算。")
    mean = float(np.dot(values_array, weights_array) / total_weight)
    if mean <= 0:
        return _metric(None, False, "变量均值为零或非正，基尼系数没有解释意义，未计算。")
    order = np.argsort(values_array, kind="mergesort")
    sorted_values = values_array[order]
    sorted_weights = weights_array[order]
    cumulative_before = np.cumsum(sorted_weights) - sorted_weights
    weighted_prefix = np.cumsum(sorted_weights * sorted_values) - sorted_weights * sorted_values
    pair_weighted_difference = 2.0 * float(
        np.sum(sorted_weights * (sorted_values * cumulative_before - weighted_prefix)))
    gini_value = pair_weighted_difference / (2.0 * total_weight ** 2 * mean)
    return _metric(gini_value)


def _binned_overlap(first_values: Sequence[float], first_weights: Sequence[float],
                    second_values: Sequence[float], second_weights: Sequence[float],
                    bins: int = DISTRIBUTION_BINS) -> dict:
    if not first_values or not second_values:
        return {"value": None, "applicable": False, "reason": "一组分布为空，无法计算重叠。",
                "method": "等宽分箱后的重合质量近似", "bins": bins}
    first_array = np.asarray(first_values, dtype=float)
    second_array = np.asarray(second_values, dtype=float)
    first_weight = np.asarray(first_weights, dtype=float)
    second_weight = np.asarray(second_weights, dtype=float)
    low = float(min(first_array.min(), second_array.min()))
    high = float(max(first_array.max(), second_array.max()))
    if high <= low:
        equal = bool(np.allclose(first_array, second_array))
        return {"value": 1.0 if equal else 0.0, "applicable": True, "reason": None,
                "method": "单点分布", "bins": 1}
    edges = np.linspace(low, high, bins + 1)
    first_hist, _ = np.histogram(first_array, bins=edges, weights=first_weight)
    second_hist, _ = np.histogram(second_array, bins=edges, weights=second_weight)
    first_total = first_hist.sum()
    second_total = second_hist.sum()
    if first_total <= 0 or second_total <= 0:
        return {"value": None, "applicable": False, "reason": "分箱后权重合计非正。",
                "method": "等宽分箱后的重合质量近似", "bins": bins}
    first_mass = first_hist / first_total
    second_mass = second_hist / second_total
    overlap = float(np.minimum(first_mass, second_mass).sum())
    return {"value": _num(overlap), "applicable": True, "reason": None,
            "method": "等宽分箱后的重合质量近似，仅用于分布重叠描述", "bins": bins}


def _compare_distribution(base_values: Sequence[float], current_values: Sequence[float],
                          base_weights: Sequence[float], current_weights: Sequence[float]) -> dict:
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    rows = []
    for q in quantiles:
        base_value = _weighted_quantile(base_values, base_weights, q) if base_values else None
        current_value = _weighted_quantile(current_values, current_weights, q) if current_values else None
        rows.append({
            "quantile": q,
            "base": _num(base_value),
            "current": _num(current_value),
            "change": _num(current_value - base_value) if None not in (base_value, current_value) else None,
        })
    return {
        "quantiles": rows,
        "overlap": _binned_overlap(base_values, base_weights, current_values, current_weights),
        "note": "分位数变化与均值变化可能方向不同；分布重叠说明单看组均值会掩盖个体差异。",
    }


# ---------------------------------------------------------------------------
# 期间与资料读取
# ---------------------------------------------------------------------------

def _parse_months(spec: Any, label: str) -> list[str]:
    if isinstance(spec, str):
        parts = [part.strip() for part in spec.split(",")]
    elif isinstance(spec, (list, tuple)):
        parts = [str(part).strip() for part in spec]
    else:
        raise AnalysisError(f"{label} 期间应为月份或逗号分隔的月份")
    months = [part for part in parts if part]
    if not months:
        raise AnalysisError(f"{label} 期间不能为空")
    for month in months:
        year, separator, number = month.partition("-")
        if not separator or not number.isdigit() or not year.isdigit() or len(year) != 4:
            raise AnalysisError(f"{label} 期间月份格式无效：{month}")
        if not 1 <= int(number) <= 12:
            raise AnalysisError(f"{label} 期间月份格式无效：{month}")
    if len(set(months)) != len(months):
        raise AnalysisError(f"{label} 期间包含重复月份")
    return months


def _available_months(catalog) -> list[str]:
    months = {record["month"] for record in catalog.rows("purchases")}
    if not months:
        raise AnalysisError("采购表没有任何月份，无法分析")
    return sorted(months)


def resolve_periods(catalog, base=None, current=None) -> tuple[list[str], list[str]]:
    months = _available_months(catalog)
    base_months = _parse_months(base, "基期") if base else [months[0]]
    current_months = _parse_months(current, "当期") if current else [months[-1]]
    overlap = set(base_months) & set(current_months)
    if overlap:
        raise AnalysisError("基期与当期不能包含相同月份：" + "、".join(sorted(overlap)))
    for label, selected in (("基期", base_months), ("当期", current_months)):
        unknown = [month for month in selected if month not in months]
        if unknown:
            raise AnalysisError(f"{label}包含采购表中不存在的月份：" + "、".join(unknown))
    return sorted(base_months), sorted(current_months)


def _design_lookup(catalog) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    lookup: dict[tuple[str, str], dict] = {}
    duplicates = []
    for record in catalog.rows("panel_design"):
        key = (record["merchant_id"], record["month"])
        if key in lookup:
            duplicates.append(key)
            continue
        lookup[key] = record
    return lookup, duplicates


def _register_merchants(catalog) -> dict[str, dict]:
    return {record["id"]: record for record in catalog.rows("merchants")}


def _reference_prices(catalog, mode: str, base_months: Sequence[str],
                      analysis_months: Sequence[str]) -> tuple[dict, dict, dict]:
    rows = catalog.rows("price_reference")
    matched: dict[tuple[str, str, str], Decimal] = {}
    for record in rows:
        key = (record["month"], record["species"], record["grade"])
        matched[key] = Decimal(record["unit_price"])
    frozen: dict[tuple[str, str], Decimal] = {}
    missing_specs: dict[tuple[str, str], str] = {}
    if mode == "matched":
        return matched, frozen, missing_specs
    periods = base_months if mode == "frozen" else analysis_months
    grouped: dict[tuple[str, str], list[Decimal]] = defaultdict(list)
    for record in rows:
        if record["month"] in periods:
            grouped[(record["species"], record["grade"])].append(Decimal(record["unit_price"]))
    for key, values in grouped.items():
        frozen[key] = sum(values, ZERO) / len(values)
    wanted_specs = {(record["species"], record["grade"]) for record in catalog.rows("purchases")}
    for spec in sorted(wanted_specs):
        if spec not in frozen:
            missing_specs[spec] = "参考价期间内没有该品规的价格，无法形成可比参考价"
    return matched, frozen, missing_specs


def _price_for_line(mode: str, matched: Mapping, frozen: Mapping,
                    month: str, species: str, grade: str) -> Decimal | None:
    if mode == "matched":
        return matched.get((month, species, grade))
    return frozen.get((species, grade))


def _in_window(merchant: Mapping, month: str) -> bool:
    entry = merchant["entry_month"]
    exit_month = merchant["exit_month"]
    return entry <= month and (exit_month is None or month < exit_month)


def _merchant_period_records(catalog, months: Sequence[str], reference_mode: str,
                             base_months: Sequence[str],
                             analysis_months: Sequence[str]) -> tuple[dict, dict]:
    """把采购行聚成商户层期间记录。

    只有实际出现采购行的商户才形成记录；已观察但没有采购行的商户单列在
    诊断中，不以节省额为零的假记录进入分布。未观察月份没有采购行，也不
    形成零采购或零节省额记录。
    """
    design, duplicate_keys = _design_lookup(catalog)
    merchants = _register_merchants(catalog)
    matched_prices, frozen_prices, missing_specs = _reference_prices(
        catalog, reference_mode, base_months, analysis_months)
    matched = matched_prices if reference_mode == "matched" else None
    frozen = None if reference_mode == "matched" else frozen_prices
    period_months = set(months)

    records: dict[str, dict] = {}
    diagnostics = {
        "duplicate_design_rows": [{"merchant_id": key[0], "month": key[1]}
                                  for key in duplicate_keys],
        "purchases_without_design_row": [],
        "purchases_outside_operation_window": [],
        "panel_observed_outside_window": [],
        "observed_without_purchase_rows": [],
        "observed_without_purchase_merchants": [],
        "eligible_without_purchase_merchants": [],
        "nonresponse_with_purchase_rows": [],
        "reference_missing_lines": [],
        "reference_missing_specs": missing_specs,
    }

    def ensure(merchant_id: str) -> dict:
        if merchant_id not in records:
            merchant = merchants[merchant_id]
            records[merchant_id] = {
                "merchant_id": merchant_id,
                "group": merchant["group"],
                "entry_month": merchant["entry_month"],
                "exit_month": merchant["exit_month"],
                "observed_months": [],
                "nonresponse_months": [],
                "eligible_months": [],
                "quantity": ZERO,
                "priced_quantity": ZERO,
                "spend": ZERO,
                "savings": ZERO,
                "line_count": 0,
                "priced_line_count": 0,
                "weight_sum": 0.0,
                "weight_count": 0,
                "clusters": set(),
                "strata": set(),
                "days_open_sum": 0,
                "days_open_count": 0,
            }
        return records[merchant_id]

    purchase_keys = set()
    for line in catalog.rows("purchases"):
        if line["month"] not in period_months:
            continue
        merchant_id = line["merchant_id"]
        month = line["month"]
        merchant = merchants.get(merchant_id)
        if merchant is None:
            raise AnalysisError(f"采购记录引用了不存在的商户：{merchant_id}")
        if not _in_window(merchant, month):
            diagnostics["purchases_outside_operation_window"].append(
                {"merchant_id": merchant_id, "month": month})
            continue
        key = (merchant_id, month)
        purchase_keys.add(key)
        record = ensure(merchant_id)
        quantity = _dec(line["quantity_kg"])
        paid = _dec(line["unit_price"])
        spend = quantity * paid
        reference = _price_for_line(reference_mode, matched, frozen,
                                    month, line["species"], line["grade"])
        record["quantity"] += quantity
        record["spend"] += spend
        record["line_count"] += 1
        if reference is None:
            diagnostics["reference_missing_lines"].append(
                {"merchant_id": merchant_id, "month": month,
                 "species": line["species"], "grade": line["grade"]})
        else:
            record["priced_quantity"] += quantity
            record["savings"] += quantity * (reference - paid)
            record["priced_line_count"] += 1

    observed_no_purchase = set()
    eligible_no_purchase: dict[str, dict] = {}
    for key, design_row in design.items():
        merchant_id, month = key
        if month not in period_months:
            continue
        merchant = merchants.get(merchant_id)
        if merchant is None:
            raise AnalysisError(f"面板设计引用了不存在的商户：{merchant_id}")
        if not _in_window(merchant, month):
            if design_row["observed"]:
                diagnostics["panel_observed_outside_window"].append(
                    {"merchant_id": merchant_id, "month": month,
                     "reason": "回收标记为已观察，但该月不在入场至退出之间的经营窗口"})
            continue
        if merchant_id not in records:
            detail = eligible_no_purchase.setdefault(
                merchant_id, {"merchant_id": merchant_id, "group": merchant["group"],
                              "entry_month": merchant["entry_month"],
                              "exit_month": merchant["exit_month"],
                              "observed_months": [], "nonresponse_months": []})
            if design_row["observed"]:
                observed_no_purchase.add(merchant_id)
                detail["observed_months"].append(month)
                diagnostics["observed_without_purchase_rows"].append(
                    {"merchant_id": merchant_id, "month": month,
                     "reason": "该月标记为已观察，但没有采购行；不按零节省额记入商户分布"})
            else:
                detail["nonresponse_months"].append(month)
            continue
        record = records[merchant_id]
        record["eligible_months"].append(month)
        if design_row["observed"]:
            record["observed_months"].append(month)
            record["weight_sum"] += float(design_row["weight"])
            record["weight_count"] += 1
            record["clusters"].add(design_row["cluster"])
            record["strata"].add(design_row["stratum"])
            record["days_open_sum"] += int(design_row["days_open"])
            record["days_open_count"] += 1
            if (merchant_id, month) not in purchase_keys:
                diagnostics["observed_without_purchase_rows"].append(
                    {"merchant_id": merchant_id, "month": month,
                     "reason": "该月标记为已观察但该商户没有采购行；该月不形成零采购记录"})
        else:
            record["nonresponse_months"].append(month)
            if (merchant_id, month) in purchase_keys:
                diagnostics["nonresponse_with_purchase_rows"].append(
                    {"merchant_id": merchant_id, "month": month})

    diagnostics["observed_without_purchase_merchants"] = sorted(observed_no_purchase)
    diagnostics["eligible_without_purchase_merchants"] = [
        {"merchant_id": detail["merchant_id"], "group": detail["group"],
         "entry_month": detail["entry_month"], "exit_month": detail["exit_month"],
         "observed_months": sorted(detail["observed_months"]),
         "nonresponse_months": sorted(detail["nonresponse_months"]),
         "reason": "期间内可经营但没有采购行，不形成商户记录，也不虚构零节省额"}
        for _, detail in sorted(eligible_no_purchase.items())
    ]
    for key in purchase_keys:
        if key not in design:
            diagnostics["purchases_without_design_row"].append(
                {"merchant_id": key[0], "month": key[1]})

    for record in records.values():
        record["observed_months"] = sorted(record["observed_months"])
        record["nonresponse_months"] = sorted(record["nonresponse_months"])
        record["eligible_months"] = sorted(record["eligible_months"])
        record["observed_month_count"] = len(record["observed_months"])
        record["eligible_month_count"] = len(record["eligible_months"])
        record["incomplete_month_count"] = (record["eligible_month_count"]
                                            - record["observed_month_count"])
        record["total_month_count"] = record["eligible_month_count"]
        record["weight_design"] = (record["weight_sum"] / record["weight_count"]
                                  if record["weight_count"] else 1.0)
        record["weight_equal"] = 1.0
        record["clusters"] = sorted(record["clusters"])
        record["strata"] = sorted(record["strata"])
        record["days_open_mean"] = (record["days_open_sum"] / record["days_open_count"]
                                    if record["days_open_count"] else None)
        count = record["observed_month_count"] or 1
        record["monthly_quantity"] = float(record["quantity"] / count)
        record["monthly_spend"] = float(record["spend"] / count)
        record["monthly_savings"] = float(record["savings"] / count)
        priced = record["priced_quantity"]
        record["unit_saving"] = float(record["savings"] / priced) if priced > 0 else None
        record["priced_quantity_share"] = (float(priced / record["quantity"])
                                           if record["quantity"] > 0 else None)
        for field_name in ("quantity", "priced_quantity", "spend", "savings"):
            record[field_name] = float(record[field_name])
    return records, diagnostics


# ---------------------------------------------------------------------------
# 汇总与加权
# ---------------------------------------------------------------------------

def _weighted_totals(records: Iterable[Mapping], weight_field: str) -> dict:
    rows = list(records)
    totals = {
        "merchant_count": len(rows),
        "quantity": 0.0,
        "priced_quantity": 0.0,
        "spend": 0.0,
        "savings": 0.0,
        "weight_total": 0.0,
        "weighted_quantity": 0.0,
        "weighted_spend": 0.0,
        "weighted_savings": 0.0,
        "unweighted_savings": 0.0,
    }
    for row in rows:
        weight = float(row.get(weight_field) or 1.0)
        totals["weight_total"] += weight
        totals["quantity"] += float(row["quantity"])
        totals["priced_quantity"] += float(row["priced_quantity"])
        totals["spend"] += float(row["spend"])
        totals["savings"] += float(row["savings"])
        totals["weighted_quantity"] += float(row["quantity"]) * weight
        totals["weighted_spend"] += float(row["spend"]) * weight
        totals["weighted_savings"] += float(row["savings"]) * weight
        totals["unweighted_savings"] += float(row["savings"])
    for field_name in ("quantity", "priced_quantity", "spend", "savings",
                       "weight_total", "weighted_quantity", "weighted_spend",
                       "weighted_savings", "unweighted_savings"):
        totals[field_name] = _num(totals[field_name])
    return totals


def _spending_levels(records: Iterable[Mapping]) -> dict:
    """把支出变化拆成数量效应与价格效应，避免把少买说成受益。

    支出 = 数量 × 实付均价。相对参考价的口径下，支出下降可能同时来自
    数量减少和实付价靠近参考价；这里只按会计恒等式分解观察到的支出。
    """
    rows = list(records)
    quantity = sum(float(row["quantity"]) for row in rows)
    spend = sum(float(row["spend"]) for row in rows)
    savings = sum(float(row["savings"]) for row in rows)
    return {
        "observed_quantity": _num(quantity),
        "observed_spend": _num(spend),
        "observed_savings": _num(savings),
        "savings_per_kg": _num(savings / quantity) if quantity > 0 else None,
        "spend_per_kg": _num(spend / quantity) if quantity > 0 else None,
        "note": "支出、数量与节省额分别报告；支出下降本身不等于节省额上升。",
    }


def _weight_attachment(records: Iterable[Mapping], weight_field: str) -> dict:
    rows = list(records)
    with_design = sum(1 for row in rows if row.get("weight_count"))
    return {
        "merchants": len(rows),
        "with_design_weight": with_design,
        "without_design_weight": len(rows) - with_design,
        "weight_field": weight_field,
        "note": ("使用 panel_design 中已观察月份的 design 权重期间均值。"
                 if weight_field == "weight_design" else
                 "使用等权 1，仅用于与设计权重结果对照；此时每个商户只计一次。"),
    }


def _group_summary(records: Sequence[Mapping], weight_field: str,
                   bottom_quantile: float) -> dict:
    rows = list(records)
    values = [float(row["savings"]) for row in rows]
    weights = [float(row.get(weight_field) or 1.0) for row in rows]
    totals = _weighted_totals(rows, weight_field)
    saving_values = [float(row["unit_saving"]) for row in rows if row["unit_saving"] is not None]
    saving_weights = [float(row.get(weight_field) or 1.0) for row in rows
                      if row["unit_saving"] is not None]
    summary = {
        "merchant_count": len(rows),
        "weight_note": ("按商户一条期间记录；权重为期间内已观察月份的设计权重均值。"
                        if weight_field == "weight_design" else
                        "按商户一条期间记录，等权 1；用于与设计权重结果对照。"),
        "quantity": totals["quantity"],
        "spend": totals["spend"],
        "savings": totals["savings"],
        "savings_per_kg": _num(totals["savings"] / totals["priced_quantity"])
        if totals["priced_quantity"] else None,
        "weighted_quantity": totals["weighted_quantity"],
        "weighted_spend": totals["weighted_spend"],
        "weighted_savings": totals["weighted_savings"],
        "total_savings_distribution": _distribution_summary(values, weights, weight_field),
        "unit_savings_distribution": _distribution_summary(saving_values, saving_weights,
                                                          weight_field) if saving_values else None,
        "gini_total_savings": _gini(values, weights),
        "bottom_quantile": bottom_quantile,
    }
    summary["bottom_coverage"] = {
        "by_savings": _bottom_coverage(rows, "savings", weight_field, bottom_quantile,
                                       "结果最差：按商户节省额从低到高"),
        "by_spend": _bottom_coverage(rows, "spend", weight_field, bottom_quantile,
                                     "规模最小：按商户期间支出从低到高"),
        "by_quantity": _bottom_coverage(rows, "quantity", weight_field, bottom_quantile,
                                        "采购最少：按商户期间采购量从低到高"),
    }
    return summary


def _bottom_coverage(records: Sequence[Mapping], rank_field: str, weight_field: str,
                     quantile: float, label: str) -> dict:
    """按某一字段从低到高累计到指定权重份额的底部群体。

    ``rank_field`` 可取 ``savings``、``spend`` 或 ``quantity``。按节省额排序
    看到的是“结果最差”的商户；按支出或采购量排序看到的是“规模最小”的
    商户。两者含义不同，不能互相替代。
    """
    rows = list(records)
    if not rows:
        return {"merchant_count": 0, "rank_field": rank_field,
                "note": "没有可排序商户。"}
    weights = [float(row.get(weight_field) or 1.0) for row in rows]
    total_weight = sum(weights)
    if total_weight <= 0:
        return {"merchant_count": 0, "rank_field": rank_field,
                "note": "权重合计非正，无法定义底部。"}
    ranked = [row for row in rows if row.get(rank_field) is not None]
    if not ranked:
        return {"merchant_count": 0, "rank_field": rank_field,
                "note": "该字段全部缺失，无法定义底部。"}
    ranked_weights = [float(row.get(weight_field) or 1.0) for row in ranked]
    total_ranked_weight = sum(ranked_weights)
    order = np.argsort([float(row[rank_field]) for row in ranked], kind="mergesort")
    target = total_ranked_weight * quantile
    selected = []
    accumulated = 0.0
    for index in order:
        selected.append(ranked[int(index)])
        accumulated += ranked_weights[int(index)]
        if accumulated >= target - 1e-12:
            break
    selected_weight = sum(float(row.get(weight_field) or 1.0) for row in selected)
    negative = [row for row in selected if float(row["savings"]) < 0]
    positive = [row for row in selected if float(row["savings"]) > 0]
    negative_weight = sum(float(row.get(weight_field) or 1.0) for row in negative)
    positive_weight = sum(float(row.get(weight_field) or 1.0) for row in positive)
    return {
        "quantile": quantile,
        "rank_field": rank_field,
        "label": label,
        "merchant_count": len(selected),
        "weight_share": _num(_share(selected_weight, total_ranked_weight)),
        "negative_count": len(negative),
        "negative_share_within_bottom": _num(_share(negative_weight, selected_weight)),
        "positive_count": len(positive),
        "positive_share_within_bottom": _num(_share(positive_weight, selected_weight)),
        "share_basis": "按样本权重份额，不是商户个数比例。",
        "savings_share_of_total": _num(
            _share(sum(float(row["savings"]) for row in selected),
                   sum(float(row["savings"]) for row in rows))),
        "groups": dict(Counter(row["group"] for row in selected)),
        "definition": "按该字段从低到高累计到指定权重份额的商户；负节省额保留在结果分布中。",
    }


# ---------------------------------------------------------------------------
# 两期比较与分解
# ---------------------------------------------------------------------------

def _matched_pairs(base_records: Mapping[str, Mapping],
                   current_records: Mapping[str, Mapping],
                   base_months: Sequence[str],
                   current_months: Sequence[str],
                   merchants: Mapping[str, Mapping]) -> dict:
    """只比较两期都出现的商户，并单列入场、退出与不完整月份。

    退出以 ``merchants.exit_month`` 为准，不把当期缺行自动当作退出。
    当期缺行且没有登记退出月的商户，标为缺少当期观察，而不是退出。
    基期缺行且入场月不晚于基期最后一月的商户，同样标为不完整观察。
    """
    base_ids = set(base_records)
    current_ids = set(current_records)
    common = sorted(base_ids & current_ids)
    entered = sorted(current_ids - base_ids)
    exited = sorted(base_ids - current_ids)
    last_base_month = max(base_months)
    first_current_month = min(current_months)
    exited_registered = [identifier for identifier in exited
                         if base_records[identifier]["exit_month"] is not None
                         and base_records[identifier]["exit_month"] <= first_current_month]
    missing_current = [identifier for identifier in exited
                       if identifier not in exited_registered]
    entered_registered = [identifier for identifier in entered
                          if current_records[identifier]["entry_month"] > last_base_month]
    missing_base = [identifier for identifier in entered
                    if identifier not in entered_registered]
    # 两期都没有期间记录的已登记商户：可能是两期之间入场又退出，也可能
    # 有可经营月份却始终没有采购行；两者都不静默丢弃，分开列出。
    present = set(base_records) | set(current_records)
    absent_both = sorted(identifier for identifier in merchants
                         if identifier not in present)
    entered_and_exited_between = [
        identifier for identifier in absent_both
        if merchants[identifier]["entry_month"] > last_base_month
        and merchants[identifier]["exit_month"] is not None
        and merchants[identifier]["exit_month"] <= first_current_month]
    registered_without_records = [identifier for identifier in absent_both
                                  if identifier not in entered_and_exited_between]
    return {
        "common": common,
        "entered": entered,
        "exited": exited,
        "exited_with_registered_exit": exited_registered,
        "missing_current_without_exit": missing_current,
        "entered_with_registered_entry": entered_registered,
        "missing_base_without_entry": missing_base,
        "entered_and_exited_between_periods": entered_and_exited_between,
        "registered_without_records": registered_without_records,
        "record_absent_reasons": {
            "entered_and_exited_between_periods": "登记入场月晚于基期末月且退出月不晚于当期首月，两期都没有可经营月份。",
            "registered_without_records": "登记商户在两期均未形成期间记录：可能没有采购行，也可能可经营月份都已落在两期之间；不等同于退出。",
        },
        "counts": {
            "base": len(base_ids),
            "current": len(current_ids),
            "common": len(common),
            "entered": len(entered),
            "exited": len(exited),
            "exited_with_registered_exit": len(exited_registered),
            "missing_current_without_exit": len(missing_current),
            "entered_with_registered_entry": len(entered_registered),
            "missing_base_without_entry": len(missing_base),
            "entered_and_exited_between_periods": len(entered_and_exited_between),
            "registered_without_records": len(registered_without_records),
        },
        "note": "入场与退出以商户登记月为准；某期未出现但登记月不能解释的，标为不完整观察，不当作入场或退出。两期之间入场又退出的商户单列。已观察但没有采购行的商户不会形成期间记录，见 diagnostics.observed_without_purchase。",
        "first_current_month": first_current_month,
    }


def _common_merchant_changes(base_records: Mapping[str, Mapping],
                             current_records: Mapping[str, Mapping],
                             weight_field: str) -> dict:
    """共同商户的逐户变化：采购量、支出、节省额分开，便于核对。"""
    rows = []
    for identifier in sorted(set(base_records) & set(current_records)):
        base = base_records[identifier]
        current = current_records[identifier]
        base_weight = float(base.get(weight_field) or 1.0)
        current_weight = float(current.get(weight_field) or 1.0)
        base_saving = float(base["savings"])
        current_saving = float(current["savings"])
        base_quantity = float(base["quantity"])
        current_quantity = float(current["quantity"])
        rows.append({
            "merchant_id": identifier,
            "group": base["group"],
            "base_quantity": _num(base_quantity),
            "current_quantity": _num(current_quantity),
            "quantity_change": _num(current_quantity - base_quantity),
            "quantity_ratio": _num(current_quantity / base_quantity)
            if base_quantity > 0 else None,
            "base_spend": _num(base["spend"]),
            "current_spend": _num(current["spend"]),
            "spend_change": _num(float(current["spend"]) - float(base["spend"])),
            "base_savings": _num(base_saving),
            "current_savings": _num(current_saving),
            "savings_change": _num(current_saving - base_saving),
            "base_weight": _num(base_weight),
            "current_weight": _num(current_weight),
            "base_observed_months": base["observed_months"],
            "current_observed_months": current["observed_months"],
            "base_incomplete_months": base["incomplete_month_count"],
            "current_incomplete_months": current["incomplete_month_count"],
        })
    rows.sort(key=lambda row: row["base_quantity"], reverse=True)
    return {
        "merchant_count": len(rows),
        "rows": rows,
        "note": "quantity_change 为观察到的采购量变化；减少采购即支出下降，不代表节省额或受益增加。",
    }


def _quantity_change_summary(base_records: Mapping[str, Mapping],
                             current_records: Mapping[str, Mapping]) -> dict:
    common = sorted(set(base_records) & set(current_records))
    entered = sorted(set(current_records) - set(base_records))
    exited = sorted(set(base_records) - set(current_records))

    def total(records, identifier_list, field):
        return sum(float(records[identifier][field]) for identifier in identifier_list)

    base_quantity = total(base_records, common, "quantity")
    current_quantity = total(current_records, common, "quantity")
    return {
        "common": {
            "merchant_count": len(common),
            "base_quantity": _num(base_quantity),
            "current_quantity": _num(current_quantity),
            "quantity_change": _num(current_quantity - base_quantity),
            "quantity_change_ratio": _num(current_quantity / base_quantity)
            if base_quantity > 0 else None,
            "base_savings": _num(total(base_records, common, "savings")),
            "current_savings": _num(total(current_records, common, "savings")),
            "savings_change": _num(total(current_records, common, "savings")
                                   - total(base_records, common, "savings")),
        },
        "entrant_quantity": _num(total(current_records, entered, "quantity")),
        "exiter_quantity": _num(total(base_records, exited, "quantity")),
        "note": "量、支出与节省额并列：采购量下降会让支出下降，但节省额由单位价差决定，两者不能混同。",
    }


def _composition_decomposition(base_records: Sequence[Mapping],
                               current_records: Sequence[Mapping],
                               weight_field: str) -> dict:
    """加权组均值的精确可加分解，并另报组间差距。

    记 w_g 为群体加权份额，m_g 为组内均值，0/1 表示基期/当期：

        Δ = Σ (w1g − w0g) · m0g        构成项（商户构成变化）
          + Σ w0g · (m1g − m0g)        组内项（各组自身均值变化）
          + Σ (w1g − w0g) · (m1g − m0g)  交互项

    三项相加恰等于加权总均值差。交互项说明构成变化与组内变化同时发生时
    互相放大或抵消，不能省掉后再把差额归给某一项。若某一商户组只在其中
    一期出现，该组的构成与交互项没有对应基期或当期均值，整组出现或消失
    单列为组周转项，可加分解相应变成四项。

    组间差距是横截面概念：各期组均值之间的差距及其变化，与可加分解并列
    展示，但不计入Δ，避免重复计算。分布重叠和组内异质性在
    ``distribution`` 中另行报告；本分解只解释均值，不解释商户个体受益。
    """
    base_rows: dict[str, list] = {}
    current_rows: dict[str, list] = {}
    for row in base_records:
        base_rows.setdefault(row["group"], []).append(row)
    for row in current_records:
        current_rows.setdefault(row["group"], []).append(row)
    groups = sorted(set(base_rows) | set(current_rows))

    def shares(rows_by_group):
        totals = {group: sum(float(row.get(weight_field) or 1.0)
                             for row in rows_by_group.get(group, []))
                  for group in groups}
        grand = sum(totals.values())
        return {group: (totals[group] / grand if grand else 0.0) for group in groups}

    def means(rows_by_group):
        output = {}
        for group in groups:
            rows = rows_by_group.get(group, [])
            group_weight = sum(float(row.get(weight_field) or 1.0) for row in rows)
            output[group] = (sum(float(row["savings"]) * float(row.get(weight_field) or 1.0)
                                 for row in rows) / group_weight) if group_weight > 0 else None
        return output

    base_shares = shares(base_rows)
    current_shares = shares(current_rows)
    base_means = means(base_rows)
    current_means = means(current_rows)
    total_base = sum((base_means[group] or 0.0) * base_shares[group] for group in groups)
    total_current = sum((current_means[group] or 0.0) * current_shares[group] for group in groups)

    composition_terms = []
    within_terms = []
    interaction_terms = []
    group_turnover_terms = []
    base_only = sorted(group for group in groups if group not in current_rows)
    current_only = sorted(group for group in groups if group not in base_rows)
    for group in groups:
        base_mean = base_means[group]
        current_mean = current_means[group]
        if group in base_only:
            composition = 0.0
            turnover = -(base_shares[group] * base_mean) if base_mean is not None else None
            within = interaction = None
        elif group in current_only:
            composition = 0.0
            turnover = current_shares[group] * current_mean if current_mean is not None else None
            within = interaction = None
        else:
            composition = ((current_shares[group] - base_shares[group]) * base_mean
                           if base_mean is not None else None)
            turnover = None
            if base_mean is not None and current_mean is not None:
                within = base_shares[group] * (current_mean - base_mean)
                interaction = ((current_shares[group] - base_shares[group])
                               * (current_mean - base_mean))
            else:
                within = interaction = None
        group_turnover_terms.append({"group": group, "contribution": _num(turnover)})
        composition_terms.append({"group": group, "contribution": _num(composition)})
        within_terms.append({"group": group, "contribution": _num(within)})
        interaction_terms.append({"group": group, "contribution": _num(interaction)})

    def total(terms):
        return sum(term["contribution"] for term in terms if term["contribution"] is not None)

    ordered_base = [(group, base_means[group]) for group in groups
                    if base_means[group] is not None]
    if len(ordered_base) < 2:
        base_gap = current_gap = None
        base_pair = current_pair = None
    else:
        high_group = max(ordered_base, key=lambda item: item[1])[0]
        low_group = min(ordered_base, key=lambda item: item[1])[0]
        base_pair = {"high_group": high_group, "low_group": low_group}
        current_pair = {"high_group": high_group, "low_group": low_group}
        base_gap_value = base_means[high_group] - base_means[low_group]
        if current_means[high_group] is None or current_means[low_group] is None:
            current_gap = None
        else:
            current_gap = current_means[high_group] - current_means[low_group]
        base_gap = base_gap_value
    total_change = total_current - total_base
    explained = (total(composition_terms) + total(within_terms)
                 + total(interaction_terms) + total(group_turnover_terms))
    return {
        "base": {
            "groups": {group: {"mean_savings": _num(base_means[group]),
                               "weight_share": _num(base_shares[group]),
                               "merchant_count": len(base_rows.get(group, []))}
                       for group in groups},
            "overall_mean_savings": _num(total_base),
        },
        "current": {
            "groups": {group: {"mean_savings": _num(current_means[group]),
                               "weight_share": _num(current_shares[group]),
                               "merchant_count": len(current_rows.get(group, []))}
                       for group in groups},
            "overall_mean_savings": _num(total_current),
        },
        "total_mean_change": _num(total_change),
        "composition": {"total": _num(total(composition_terms)),
                        "groups": composition_terms,
                        "meaning": "商户构成变化按基期组均值加权；入场、退出与组间比例变化都在这里。"},
        "within": {"total": _num(total(within_terms)),
                   "groups": within_terms,
                   "meaning": "按基期权重的组内均值变化；各组自身变好或变差。"},
        "interaction": {"total": _num(total(interaction_terms)),
                        "groups": interaction_terms,
                        "meaning": "构成与组内变化同时发生的交叉项；存在分布重叠时，它不能解释个体差异。"},
        "group_turnover": {"total": _num(total(group_turnover_terms)),
                           "groups": group_turnover_terms,
                           "meaning": "只在其中一期出现的商户组；整组出现或消失单独列出，其构成项记为零，不混入组内变化。"},
        "between_group_gap": {
            "base": _num(base_gap),
            "current": _num(current_gap),
            "change": _num(current_gap - base_gap) if None not in (base_gap, current_gap) else None,
            "base_pair": base_pair,
            "current_pair": current_pair,
            "meaning": "横截面组均值差距，与可加分解并列报告，不计入Δ，避免重复计算。",
        },
        "residual": _num(total_change - explained),
        "residual_note": "残差为六位小数舍入差；未舍入时各项相加等于加权总均值差。",
        "identity": "Δ = 构成项 + 组内项 + 交互项 + 组出现/消失项；四项相加应等于加权总均值差，组间差距单列不计入。",
        "limits": "分解只作用于加权组均值。分布重叠、组内异质性和极端大户会使组均值差无法代表个体受益；见 distribution 与 sensitivity。",
    }

def _flow_summary(records: Sequence[Mapping], weight_field: str, definition: str) -> dict:
    rows = list(records)
    return {
        "definition": definition,
        "merchant_count": len(rows),
        **_weighted_totals(rows, weight_field),
        "groups": dict(Counter(row["group"] for row in rows)),
        "note": "入场与退出分别报告数量、支出与节省额；退出不等于未回收，也不把少买视为受益。",
    }


# ---------------------------------------------------------------------------
# 敏感性：极端大户与缺失月份
# ---------------------------------------------------------------------------

def _trimmed_and_winsorized(values: Sequence[float], weights: Sequence[float],
                            trim: float = 0.05) -> dict:
    if not values:
        return {"trimmed_mean": None, "winsorized_mean": None}
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    order = np.argsort(values_array, kind="mergesort")
    ordered_values = values_array[order]
    ordered_weights = weights_array[order]
    trim_count = int(ceil(len(values_array) * trim))
    if len(values_array) >= 3 and 2 * trim_count < len(values_array):
        kept_values = ordered_values[trim_count:len(ordered_values) - trim_count]
        kept_weights = ordered_weights[trim_count:len(ordered_weights) - trim_count]
        trimmed = _weighted_mean(kept_values, kept_weights)
    else:
        trimmed = None
    if trim_count >= 1:
        lower = ordered_values[trim_count - 1]
        upper = ordered_values[-trim_count]
        clipped = np.clip(values_array, lower, upper)
        winsorized = _weighted_mean(clipped, weights_array)
    else:
        winsorized = _weighted_mean(values_array, weights_array)
    return {"trimmed_mean": _num(trimmed), "winsorized_mean": _num(winsorized),
            "trim_fraction": trim, "trim_count": trim_count}


def _influence_sensitivity(records: Sequence[Mapping], weight_field: str) -> dict:
    """极端大户敏感性：删除最大的若干商户后，总量与加权均值如何变化。

    “大”按期间采购量定义，与节省额符号无关；不能据此判断谁受益。
    """
    rows = list(records)
    if not rows:
        return {"applicable": False, "reason": "没有商户记录。"}
    weights = [float(row.get(weight_field) or 1.0) for row in rows]
    values = [float(row["savings"]) for row in rows]
    period_total = sum(values)
    base_weight = sum(weights)
    base_mean = period_total / base_weight if base_weight else None
    by_quantity = sorted(rows, key=lambda row: float(row["quantity"]), reverse=True)
    scenarios = []
    for label, count in (("top_1_by_quantity", 1), ("top_5_by_quantity", 5),
                         ("top_1pct_by_quantity", max(1, ceil(len(rows) * 0.01)))):
        removed = {row["merchant_id"] for row in by_quantity[:count]}
        kept = [row for row in rows if row["merchant_id"] not in removed]
        kept_weight = sum(float(row.get(weight_field) or 1.0) for row in kept)
        kept_total = sum(float(row["savings"]) for row in kept)
        scenarios.append({
            "label": label,
            "removed_count": len(removed),
            "remaining_count": len(kept),
            "total_savings": _num(kept_total),
            "total_change": _num(kept_total - period_total),
            "weighted_mean_savings": _num(kept_total / kept_weight) if kept_weight else None,
            "weighted_mean_change": _num(kept_total / kept_weight - base_mean)
            if kept_weight and base_mean is not None else None,
        })
    return {
        "applicable": True,
        "period_total_savings": _num(period_total),
        "period_weighted_mean_savings": _num(base_mean),
        "scenarios": scenarios,
        "definition": "按期间采购量从大到小删除；“大”不等于受益或受损。",
    }


def _calendar_span(months: Sequence[str]) -> list[str]:
    available = sorted(months)
    start = min(available)
    end = max(available)
    start_year, start_month = (int(part) for part in start.split("-"))
    end_year, end_month = (int(part) for part in end.split("-"))
    output = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        output.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            year += 1
            month = 1
    return output


def _missing_month_sensitivity(catalog, months: Sequence[str], reference_mode: str,
                               base_months: Sequence[str], analysis_months: Sequence[str],
                               weight_field: str) -> dict:
    """缺失月份敏感性：完整月份商户与全部可观察商户的汇总对比。

    完整月份指商户在分析期间内每个可经营月份都有已观察采购记录。期初入场、
    期内退出的商户不会被误判为“缺月”；未回收月份明确不进入商户层分布。
    """
    design, _ = _design_lookup(catalog)
    merchants = _register_merchants(catalog)
    span_months = _calendar_span(months)
    available_months = set(_available_months(catalog))
    span_months = [month for month in span_months if month in available_months]
    purchases_present = {(record["merchant_id"], record["month"])
                         for record in catalog.rows("purchases")}
    matched_prices, frozen_prices, missing_specs = _reference_prices(
        catalog, reference_mode, base_months, analysis_months)
    matched = matched_prices if reference_mode == "matched" else None
    frozen = None if reference_mode == "matched" else frozen_prices

    complete_ids = set()
    operable_ids = set()
    for identifier, merchant in merchants.items():
        eligible = [month for month in span_months if _in_window(merchant, month)]
        if not eligible:
            continue
        operable_ids.add(identifier)
        if all((identifier, month) in purchases_present
               and design.get((identifier, month), {}).get("observed")
               for month in eligible):
            complete_ids.add(identifier)

    def aggregate(identifier_set):
        quantity = ZERO
        spend = ZERO
        savings = ZERO
        weighted_quantity = 0.0
        weighted_savings = 0.0
        weight_sum = 0.0
        for line in catalog.rows("purchases"):
            if line["month"] not in span_months or line["merchant_id"] not in identifier_set:
                continue
            identifier = line["merchant_id"]
            if not _in_window(merchants[identifier], line["month"]):
                continue
            design_row = design.get((identifier, line["month"]))
            if design_row is None or not design_row["observed"]:
                continue
            qty = _dec(line["quantity_kg"])
            paid = _dec(line["unit_price"])
            reference = _price_for_line(reference_mode, matched, frozen,
                                        line["month"], line["species"], line["grade"])
            quantity += qty
            spend += qty * paid
            weight = float(design_row["weight"])
            weighted_quantity += float(qty) * weight
            if reference is not None:
                savings += qty * (reference - paid)
                weighted_savings += float(qty * (reference - paid)) * weight
        for identifier in identifier_set:
            for month in span_months:
                design_row = design.get((identifier, month))
                if (design_row and design_row["observed"]
                        and _in_window(merchants[identifier], month)):
                    weight_sum += float(design_row["weight"])
        return {
            "merchant_count": len(identifier_set),
            "quantity": _num(quantity),
            "spend": _num(spend),
            "savings": _num(savings),
            "savings_per_kg": _num(float(savings) / float(quantity)) if quantity > 0 else None,
            "weight_sum": _num(weight_sum),
            "weighted_savings": _num(weighted_savings),
            "weighted_savings_per_weighted_kg": _num(weighted_savings / weighted_quantity)
            if weighted_quantity > 0 else None,
        }

    return {
        "complete_months": {
            "definition": "分析期间每个可经营月份都有已观察采购记录的商户；入场、退出月份不强行算作缺失。",
            "aggregate": aggregate(complete_ids),
        },
        "all_operable": {"aggregate": aggregate(operable_ids)},
        "merchant_counts": {
            "complete_months": len(complete_ids),
            "all_operable": len(operable_ids),
            "partial_or_no_purchase": len(operable_ids - complete_ids),
            "eligible_without_any_purchase": len(operable_ids - set(
                record["merchant_id"] for record in catalog.rows("purchases")
                if record["month"] in span_months)),
        },
        "span_months": span_months,
        "note": "完整月份与全部可观察商户不可互换：前者排除入场、退出与未回收月份，差额显示缺失月份对结果的影响；跨度含两期之间的所有自然月。",
        "reference_missing_specs": missing_specs,
    }


def _operating_days_sensitivity(catalog, months: Sequence[str], reference_mode: str,
                                base_months: Sequence[str],
                                analysis_months: Sequence[str]) -> dict:
    """按实际经营日数比较：少开门可能导致少买，不能直接算作受益。

    以商户月份的实际经营日数相对 30 天折算采购量；仅用于观察口径敏感性，
    不声称这是反事实。经营日数来自 panel_design.days_open。
    """
    design, _ = _design_lookup(catalog)
    merchants = _register_merchants(catalog)
    matched_prices, frozen_prices, missing_specs = _reference_prices(
        catalog, reference_mode, base_months, analysis_months)
    matched = matched_prices if reference_mode == "matched" else None
    frozen = None if reference_mode == "matched" else frozen_prices

    totals = {
        "observed": {"quantity": ZERO, "savings": ZERO, "spend": ZERO, "days": 0},
        "full_month_equivalent": {"quantity": 0.0, "savings": 0.0, "spend": 0.0},
    }
    for line in catalog.rows("purchases"):
        if line["month"] not in months:
            continue
        identifier = line["merchant_id"]
        merchant = merchants.get(identifier)
        if merchant is None or not _in_window(merchant, line["month"]):
            continue
        design_row = design.get((identifier, line["month"]))
        if design_row is None or not design_row["observed"]:
            continue
        days = int(design_row["days_open"])
        if days <= 0:
            continue
        scale = FULL_MONTH_DAYS / days
        quantity = _dec(line["quantity_kg"])
        paid = _dec(line["unit_price"])
        reference = _price_for_line(reference_mode, matched, frozen,
                                    line["month"], line["species"], line["grade"])
        totals["observed"]["quantity"] += quantity
        totals["observed"]["spend"] += quantity * paid
        totals["observed"]["days"] += days
        totals["full_month_equivalent"]["quantity"] += float(quantity) * scale
        totals["full_month_equivalent"]["spend"] += float(quantity * paid) * scale
        if reference is not None:
            saving = float(quantity * (reference - paid))
            totals["observed"]["savings"] += quantity * (reference - paid)
            totals["full_month_equivalent"]["savings"] += saving * scale
    observed = totals["observed"]
    equivalent = totals["full_month_equivalent"]
    return {
        "observed": {
            "quantity": _num(observed["quantity"]),
            "spend": _num(observed["spend"]),
            "savings": _num(observed["savings"]),
            "savings_per_kg": _num(float(observed["savings"]) / float(observed["quantity"]))
            if observed["quantity"] > 0 else None,
            "observed_days": observed["days"],
        },
        "full_month_equivalent": {
            "quantity": _num(equivalent["quantity"]),
            "spend": _num(equivalent["spend"]),
            "savings": _num(equivalent["savings"]),
            "savings_per_kg": _num(equivalent["savings"] / equivalent["quantity"])
            if equivalent["quantity"] > 0 else None,
        },
        "note": "经营日数折算只调整规模，不改变单价与节省额方向；少开门导致的少买仍不等同于受益。",
        "reference_missing_specs": missing_specs,
    }

# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def _design_structure(records: Sequence[Mapping]) -> dict:
    rows = list(records)
    weights = [row["weight_design"] for row in rows if row.get("weight_count")]
    clusters = sorted({cluster for row in rows for cluster in row.get("clusters", [])})
    strata = sorted({stratum for row in rows for stratum in row.get("strata", [])})
    return {
        "merchants": len(rows),
        "observed_design_rows": sum(row.get("weight_count", 0) for row in rows),
        "clusters": len(clusters),
        "strata": len(strata),
        "cluster_examples": clusters[:5],
        "weight_min": _num(min(weights)) if weights else None,
        "weight_max": _num(max(weights)) if weights else None,
        "note": "权重按商户期间均值展示；同一商户跨月重复出现已合并为一条期间记录，不作为独立观测重复计数。",
    }


def _period_payload(records: Mapping[str, Mapping], diagnostics: Mapping,
                    months: Sequence[str], weight_field: str,
                    bottom_quantile: float) -> dict:
    rows = list(records.values())
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    return {
        "months": list(months),
        "merchant_count": len(rows),
        "observed_without_purchase": {
            "merchant_count": len(diagnostics.get("observed_without_purchase_merchants", [])),
            "merchant_ids": diagnostics.get("observed_without_purchase_merchants", []),
            "note": "这些商户在期间内至少一个月标记为已观察但没有采购行；未按零节省额计入商户分布。",
        },
        "eligible_without_purchase": {
            "merchant_count": len(diagnostics.get("eligible_without_purchase_merchants", [])),
            "merchants": diagnostics.get("eligible_without_purchase_merchants", []),
            "note": "期间内可经营但没有任何采购行的商户；不进入商户层分布，但在此列出以免被静默遗漏。",
        },
        "incomplete_month_counts": {
            "merchants": sum(1 for row in rows if row["incomplete_month_count"] > 0),
            "total_incomplete_months": sum(row["incomplete_month_count"] for row in rows),
            "note": "不完整月份按可经营月份减去已观察月份；不含期初入场和期内退出所带来的自然缺月。",
        },
        "aggregate": _weighted_totals(rows, weight_field),
        "spending_levels": _spending_levels(rows),
        "design_structure": _design_structure(rows),
        "weight_attachment": _weight_attachment(rows, weight_field),
        "groups": {group: _group_summary(group_rows, weight_field, bottom_quantile)
                   for group, group_rows in sorted(by_group.items())},
        "overall": _group_summary(rows, weight_field, bottom_quantile),
        "top_group": _top_coverage(rows, weight_field, bottom_quantile),
        "diagnostics": diagnostics,
        "merchants": sorted(rows, key=lambda row: row["merchant_id"]),
        "notes": [
            "商户层节省额 = Σ 公斤数 × (可比参考价 − 实付价)，按商户跨月汇总。",
            "负节省额保留；已观察但没有采购行的月份不按零节省额进入分布，另在诊断中计数；未回收月份不虚构采购行。",
            "加权使用设计权重的期间均值；权重与采购量分开。",
        ],
    }


def _composition_overlap_summary(base_rows, current_rows, weight_field):
    by_group = defaultdict(lambda: ([], [], [], []))
    for row in base_rows:
        bucket = by_group[row["group"]]
        bucket[0].append(float(row["savings"]))
        bucket[1].append(float(row.get(weight_field) or 1.0))
    for row in current_rows:
        bucket = by_group[row["group"]]
        bucket[2].append(float(row["savings"]))
        bucket[3].append(float(row.get(weight_field) or 1.0))
    output = {}
    for group, (base_values, base_weights, current_values, current_weights) in sorted(by_group.items()):
        output[group] = {
            "base_count": len(base_values),
            "current_count": len(current_values),
            "base_distribution": _distribution_summary(base_values, base_weights, weight_field),
            "current_distribution": _distribution_summary(current_values, current_weights, weight_field),
            "comparison": _compare_distribution(base_values, current_values,
                                                base_weights, current_weights),
        }
    return output


def build_panel_analysis(catalog, base: Any = None, current: Any = None,
                         reference_mode: str = "matched",
                         weight_field: str = "weight_design",
                         bottom_quantile: float = DEFAULT_BOTTOM_QUANTILE) -> dict:
    """构建经营面板分析。

    参数：

    * ``base`` / ``current``：月份或逗号分隔月份；默认首月与末月。
    * ``reference_mode``：``matched``、``frozen`` 或 ``pooled``。
    * ``weight_field``：``weight_design``（设计权重）或 ``weight_equal``（等权）。
    * ``bottom_quantile``：底部覆盖的分位比例，介于 0 与 1 之间。
    """
    if reference_mode not in REFERENCE_MODES:
        raise AnalysisError("参考价模式应为 matched、frozen 或 pooled")
    if weight_field not in ("weight_design", "weight_equal"):
        raise AnalysisError("权重口径应为 weight_design 或 weight_equal")
    if not 0 < float(bottom_quantile) < 1:
        raise AnalysisError("底部比例须介于 0 与 1 之间")
    base_months, current_months = resolve_periods(catalog, base, current)
    analysis_months = sorted(set(base_months) | set(current_months))

    base_records, base_diagnostics = _merchant_period_records(
        catalog, base_months, reference_mode, base_months, analysis_months)
    current_records, current_diagnostics = _merchant_period_records(
        catalog, current_months, reference_mode, base_months, analysis_months)

    base_rows = list(base_records.values())
    current_rows = list(current_records.values())
    pairs = _matched_pairs(base_records, current_records, base_months, current_months,
                           _register_merchants(catalog))

    common_base = [base_records[identifier] for identifier in pairs["common"]]
    common_current = [current_records[identifier] for identifier in pairs["common"]]
    common_savings_values = [float(row["savings"]) for row in common_base]
    common_current_values = [float(row["savings"]) for row in common_current]
    common_weights_base = [float(row.get(weight_field) or 1.0) for row in common_base]
    common_weights_current = [float(row.get(weight_field) or 1.0) for row in common_current]

    entered_rows = [current_records[identifier] for identifier in pairs["entered"]]
    exited_rows = [base_records[identifier] for identifier in pairs["exited"]]

    decomposition = _composition_decomposition(base_rows, current_rows, weight_field)
    decomposition["distribution"] = _composition_overlap_summary(base_rows, current_rows,
                                                                  weight_field)

    for payload in (base_diagnostics, current_diagnostics):
        payload["reference_missing_specs"] = {
            f"{species}·{grade}": reason
            for (species, grade), reason in payload["reference_missing_specs"].items()
        }

    base_payload = _period_payload(base_records, base_diagnostics, base_months,
                                   weight_field, bottom_quantile)
    current_payload = _period_payload(current_records, current_diagnostics, current_months,
                                      weight_field, bottom_quantile)

    return {
        "source_fingerprint": getattr(catalog, "fingerprint", None),
        "parameters": {
            "base": base_months,
            "current": current_months,
            "reference_mode": reference_mode,
            "reference_mode_meaning": REFERENCE_MODES[reference_mode],
            "weight_field": weight_field,
            "weight_meaning": (
                "设计权重：panel_design 中已观察月份的 weight 期间均值；不是采购量。"
                if weight_field == "weight_design" else
                "等权：每个商户在期间内只计一次；用于与设计权重结果对照。"
            ),
            "bottom_quantile": bottom_quantile,
            "analysis_months": analysis_months,
            "period_selection": "未指定时基期取采购表最早月份、当期取最晚月份；可通过 base 与 current 指定单月或逗号分隔多个月。",
        },
        "base": base_payload,
        "current": current_payload,
        "comparison": {
            "matched_merchants": {
                "counts": pairs["counts"],
                "entered": pairs["entered"],
                "exited": pairs["exited"],
                "exited_with_registered_exit": pairs["exited_with_registered_exit"],
                "missing_current_without_exit": pairs["missing_current_without_exit"],
                "entered_with_registered_entry": pairs["entered_with_registered_entry"],
                "missing_base_without_entry": pairs["missing_base_without_entry"],
                "entered_and_exited_between_periods":
                    pairs["entered_and_exited_between_periods"],
                "registered_without_records": pairs["registered_without_records"],
                "note": pairs["note"],
            },
            "common_merchants": {
                "merchant_count": len(pairs["common"]),
                "base_total_savings": _num(sum(common_savings_values)),
                "current_total_savings": _num(sum(common_current_values)),
                "base_weighted_mean_savings": _num(_weighted_mean(common_savings_values,
                                                                  common_weights_base)),
                "current_weighted_mean_savings": _num(_weighted_mean(common_current_values,
                                                                     common_weights_current)),
                "weighted_mean_change": _num(
                    (_weighted_mean(common_current_values, common_weights_current) or 0.0)
                    - (_weighted_mean(common_savings_values, common_weights_base) or 0.0)),
                "composition_free": True,
                "note": "共同商户比较排除了入场/退出构成变化，但仍受共同商户内部构成变化与分布重叠影响。",
            },
            "common_merchant_changes": _common_merchant_changes(
                base_records, current_records, weight_field),
            "quantity_change": _quantity_change_summary(base_records, current_records),
            "entrants": _flow_summary(entered_rows, weight_field, "当期出现、基期未出现且入场月晚于基期末月"),
            "exiters": _flow_summary(exited_rows, weight_field, "基期出现、当期未出现且有登记退出月"),
            "decomposition": decomposition,
            "totals": {
                "base_weighted_savings": base_payload["aggregate"]["weighted_savings"],
                "current_weighted_savings": current_payload["aggregate"]["weighted_savings"],
                "base_merchant_count": base_payload["merchant_count"],
                "current_merchant_count": current_payload["merchant_count"],
            },
        },
        "sensitivity": {
            "reference_modes": _reference_mode_sensitivity(
                catalog, base_months, current_months, weight_field),
            "extreme_merchants": {
                "base": _influence_sensitivity(base_rows, weight_field),
                "current": _influence_sensitivity(current_rows, weight_field),
            },
            "missing_months": _missing_month_sensitivity(
                catalog, analysis_months, reference_mode, base_months,
                analysis_months, weight_field),
            "operating_days": _operating_days_sensitivity(
                catalog, analysis_months, reference_mode, base_months,
                analysis_months),
            "trimmed": {
                "base": _trimmed_and_winsorized(common_savings_values, common_weights_base),
                "current": _trimmed_and_winsorized(common_current_values,
                                                   common_weights_current),
                "note": "仅对共同商户计算；截尾与缩尾按商户节省额排序，权重仍为设计权重。",
            },
        },
        "distribution_indicators": {
            "base": {
                "total_savings": _distribution_summary(
                    [float(row["savings"]) for row in base_rows],
                    [float(row.get(weight_field) or 1.0) for row in base_rows], weight_field),
                "unit_savings": _distribution_summary(
                    [float(row["unit_saving"]) for row in base_rows
                     if row["unit_saving"] is not None],
                    [float(row.get(weight_field) or 1.0) for row in base_rows
                     if row["unit_saving"] is not None], weight_field),
            },
            "current": {
                "total_savings": _distribution_summary(
                    [float(row["savings"]) for row in current_rows],
                    [float(row.get(weight_field) or 1.0) for row in current_rows], weight_field),
                "unit_savings": _distribution_summary(
                    [float(row["unit_saving"]) for row in current_rows
                     if row["unit_saving"] is not None],
                    [float(row.get(weight_field) or 1.0) for row in current_rows
                     if row["unit_saving"] is not None], weight_field),
            },
            "changes": _compare_distribution(
                [float(row["savings"]) for row in base_rows],
                [float(row["savings"]) for row in current_rows],
                [float(row.get(weight_field) or 1.0) for row in base_rows],
                [float(row.get(weight_field) or 1.0) for row in current_rows]),
            "gini": {
                "base_total_savings": _gini(
                    [float(row["savings"]) for row in base_rows],
                    [float(row.get(weight_field) or 1.0) for row in base_rows]),
                "current_total_savings": _gini(
                    [float(row["savings"]) for row in current_rows],
                    [float(row.get(weight_field) or 1.0) for row in current_rows]),
                "note": "基尼系数只对非负分布有直接解释；含负节省额时返回不适用，不发布“普惠程度”综合指数。",
            },
            "inequality_change": _inequality_change(base_rows, current_rows, weight_field),
        },
        "caveats": CAVEATS,
    }


def _reference_mode_sensitivity(catalog, base_months, current_months, weight_field):
    results = {}
    for mode in REFERENCE_MODES:
        analysis = sorted(set(base_months) | set(current_months))
        base_records, _ = _merchant_period_records(catalog, base_months, mode,
                                                   base_months, analysis)
        current_records, _ = _merchant_period_records(catalog, current_months, mode,
                                                      base_months, analysis)
        base_rows = list(base_records.values())
        current_rows = list(current_records.values())
        results[mode] = {
            "base_total_savings": _num(sum(float(row["savings"]) for row in base_rows)),
            "current_total_savings": _num(sum(float(row["savings"]) for row in current_rows)),
            "base_weighted_mean_savings": _num(_weighted_mean(
                [float(row["savings"]) for row in base_rows],
                [float(row.get(weight_field) or 1.0) for row in base_rows])),
            "current_weighted_mean_savings": _num(_weighted_mean(
                [float(row["savings"]) for row in current_rows],
                [float(row.get(weight_field) or 1.0) for row in current_rows])),
            "base_negative_share": _num(sum(
                1 for row in base_rows if float(row["savings"]) < 0) / len(base_rows))
            if base_rows else None,
            "current_negative_share": _num(sum(
                1 for row in current_rows if float(row["savings"]) < 0) / len(current_rows))
            if current_rows else None,
            "meaning": REFERENCE_MODES[mode],
        }
    return {"modes": results,
            "negative_share_basis": "按商户个数比例，不是权重份额。",
            "note": "不同参考价模式回答不同问题，不能互相替代；模式间差异本身就是不确定性的一部分。"}


def _top_coverage(records, weight_field, quantile):
    rows = list(records)
    if not rows:
        return {"merchant_count": 0}
    weights = [float(row.get(weight_field) or 1.0) for row in rows]
    total_weight = sum(weights)
    if total_weight <= 0:
        return {"merchant_count": 0, "note": "权重合计非正，无法定义顶部。"}
    order = np.argsort([float(row["savings"]) for row in rows], kind="mergesort")[::-1]
    target = total_weight * quantile
    selected = []
    accumulated = 0.0
    for index in order:
        selected.append(rows[int(index)])
        accumulated += weights[int(index)]
        if accumulated >= target - 1e-12:
            break
    selected_weight = sum(float(row.get(weight_field) or 1.0) for row in selected)
    total_savings = sum(float(row["savings"]) for row in rows)
    return {
        "quantile": quantile,
        "merchant_count": len(selected),
        "weight_share": _num(_share(selected_weight, total_weight)),
        "savings_share_of_total": _num(_share(sum(float(row["savings"]) for row in selected),
                                              total_savings)),
        "groups": dict(Counter(row["group"] for row in selected)),
        "note": "顶部按节省额从高到低；总节省额为负时，份额符号仍保留。",
    }


def _inequality_change(base_rows, current_rows, weight_field):
    """不平衡变化：只报可在两期直接比较的分位数差与重叠。

    分布含负值时基尼系数不适用，这里不强行给单一“普惠程度”。
    """
    base_values = [float(row["savings"]) for row in base_rows]
    current_values = [float(row["savings"]) for row in current_rows]
    base_weights = [float(row.get(weight_field) or 1.0) for row in base_rows]
    current_weights = [float(row.get(weight_field) or 1.0) for row in current_rows]
    base_summary = _distribution_summary(base_values, base_weights, weight_field)
    current_summary = _distribution_summary(current_values, current_weights, weight_field)
    return {
        "base": base_summary,
        "current": current_summary,
        "p90_p10_ratio": {
            "base": _ratio_metric(base_summary.get("p90"), base_summary.get("p10")),
            "current": _ratio_metric(current_summary.get("p90"), current_summary.get("p10")),
            "note": "仅当 p10 为正且 p90 非负时比值才有解释；负节省额下常用不平衡比率不可直接比较。",
        },
        "overlap": _binned_overlap(base_values, base_weights, current_values, current_weights),
        "note": "存在负值时，基尼等非负指标不适用；分布重叠与分位变化优先，避免发布有数值但经济含义误导的“普惠程度”。",
    }
