from __future__ import annotations

from datetime import date, datetime
import json
from typing import Any
from urllib.parse import urlencode

from django.core.paginator import EmptyPage, Page, Paginator
from django.utils import timezone

from main.models import Assessment, User


def _display_assessment_dt(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _parse_bp(value: Any) -> tuple[float | None, float | None]:
    """
    解析血压字符串（如 120/80, 120-80, 120 80），返回 (收缩压, 舒张压)。
    解析失败返回 (None, None)。
    """
    if value is None:
        return (None, None)
    s = str(value).strip()
    if not s:
        return (None, None)
    for sep in ("/", "-", " "):
        if sep in s:
            parts = [p for p in s.split(sep) if p.strip()]
            if len(parts) >= 2:
                try:
                    return (float(parts[0].strip()), float(parts[1].strip()))
                except Exception:
                    return (None, None)
    return (None, None)


def delete_assessment_by_user_and_dt(user_id: str, assessment_date: str) -> int:
    """
    按 userID + assessment_date 精确删除一条 Assessment 记录。
    assessment_date 需为 ISO 格式字符串（例如 '2026-04-24 12:34:56.123456'）。
    返回删除条数。
    """
    user_id = (user_id or "").strip()
    dt = datetime.fromisoformat((assessment_date or "").strip())
    deleted, _ = Assessment.objects.filter(userID=user_id, assessment_date=dt).delete()
    return int(deleted)


def build_user_results_context(
    user_id: str,
    page: int = 1,
    page_size: int = 20,
    start_date: str = "",
    end_date: str = "",
    health_status: str = "",
) -> dict[str, Any]:
    user_id = (user_id or "").strip()
    if not user_id:
        return {
            "user_id": "",
            "user": None,
            "rows": [],
            "page_obj": None,
            "error": "缺少 user_id",
        }

    user = (
        User.objects.filter(userID=user_id)
        .values("userID", "userName", "userBirth", "userTel")
        .first()
    )

    status_options = (
        Assessment.objects.filter(userID=user_id)
        .exclude(health_status__isnull=True)
        .exclude(health_status__exact="")
        .values_list("health_status", flat=True)
        .distinct()
        .order_by("health_status")
    )

    qs_base = Assessment.objects.filter(userID=user_id)
    start_date = (start_date or "").strip()
    end_date = (end_date or "").strip()
    health_status = (health_status or "").strip()
    if health_status:
        qs_base = qs_base.filter(health_status=health_status)
    if start_date:
        sd = date.fromisoformat(start_date)
        qs_base = qs_base.filter(assessment_date__date__gte=sd)
    if end_date:
        ed = date.fromisoformat(end_date)
        qs_base = qs_base.filter(assessment_date__date__lte=ed)

    qs = list(
        qs_base.order_by("-assessment_date").values(
            "assessment_date",
            "height",
            "weight",
            "bmi",
            "blood_pressure",
            "blood_sugar",
            "health_status",
        )
    )

    rows_all: list[dict[str, Any]] = []
    for r in qs:
        dt: datetime | None = r.get("assessment_date")
        rows_all.append(
            {
                **r,
                "assessment_date_display": _display_assessment_dt(dt),
                "assessment_date_iso": (dt.isoformat(sep=" ", timespec="microseconds") if dt else ""),
            }
        )

    # 趋势图数据：按时间升序，便于前端绘制
    trend_rows = list(reversed(qs))
    trend_labels: list[str] = []
    trend_weight: list[float | None] = []
    trend_bmi: list[float | None] = []
    trend_blood_sugar: list[float | None] = []
    trend_bp_sys: list[float | None] = []
    trend_bp_dia: list[float | None] = []
    for r in trend_rows:
        dt: datetime | None = r.get("assessment_date")
        trend_labels.append(_display_assessment_dt(dt))
        trend_weight.append(r.get("weight"))
        trend_bmi.append(r.get("bmi"))
        trend_blood_sugar.append(r.get("blood_sugar"))
        sys_v, dia_v = _parse_bp(r.get("blood_pressure"))
        trend_bp_sys.append(sys_v)
        trend_bp_dia.append(dia_v)

    paginator = Paginator(rows_all, page_size)
    try:
        page_obj: Page[dict[str, Any]] = paginator.page(page)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages or 1)

    extra_params: dict[str, str] = {}
    if start_date:
        extra_params["start_date"] = start_date
    if end_date:
        extra_params["end_date"] = end_date
    if health_status:
        extra_params["health_status"] = health_status
    query_extra = f"&{urlencode(extra_params)}" if extra_params else ""

    return {
        "user_id": user_id,
        "user": user,
        "rows": list(page_obj.object_list),
        "page_obj": page_obj,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "health_status": health_status,
        },
        "health_status_options": list(status_options),
        "query_extra": query_extra,
        "trend": {
            "labels": trend_labels,
            "weight": trend_weight,
            "bmi": trend_bmi,
            "blood_sugar": trend_blood_sugar,
            "bp_sys": trend_bp_sys,
            "bp_dia": trend_bp_dia,
        },
        "trend_json": json.dumps(
            {
                "labels": trend_labels,
                "weight": trend_weight,
                "bmi": trend_bmi,
                "blood_sugar": trend_blood_sugar,
                "bp_sys": trend_bp_sys,
                "bp_dia": trend_bp_dia,
            },
            ensure_ascii=False,
        ),
        "now_display": timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": "" if user else "未找到该用户（userID 不存在）",
    }

