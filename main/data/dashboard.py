"""
仪表板数据：User.userID 与 Assessment.userID 对应，
按 assessment_date 取最新一条的 height、weight、bmi、health_status。
使用 ORM：先查 User，再按 userID 批量查 Assessment，内存合并（避免 Subquery 日期注解在 USE_TZ 下触发 utcoffset 问题）。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from main.models import Assessment, User


def _normalize_to_date(value: date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _calc_age(birth: date | None) -> int | None:
    if birth is None:
        return None
    today = date.today()
    age = today.year - birth.year
    if (today.month, today.day) < (birth.month, birth.day):
        age -= 1
    return age


def _fetch_users_for_dashboard() -> list[tuple[Any, ...]]:
    """User ORM：按建档时间倒序，只取展示所需列。"""
    return list(
        User.objects.order_by("-created_at").values_list(
            "userID",
            "userName",
            "userBirth",
            "userTel",
        )
    )


def _latest_assessment_by_user_id(user_ids: list[str]) -> dict[str, dict[str, Any]]:
    """
    Assessment ORM：按 userID + assessment_date 倒序，每个 userID 取第一条。
    表无 id 列，不按 id 排序；同日多条时任选一条。
    """
    if not user_ids:
        return {}

    qs = (
        Assessment.objects.filter(userID__in=user_ids)
        .order_by("userID", "-assessment_date")
        .values("userID", "height", "weight", "bmi", "health_status", "assessment_date")
    )

    out: dict[str, dict[str, Any]] = {}
    for row in qs:
        uid = str(row["userID"])
        if uid not in out:
            out[uid] = row
    return out


def build_dashboard_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    查询全部 User（created_at 倒序）；
    对每个 userID 匹配 Assessment.userID，取 assessment_date 最新一条的
    height、weight、bmi、health_status。
    """
    user_rows = _fetch_users_for_dashboard()
    user_ids = [str(r[0]) for r in user_rows]
    latest_map = _latest_assessment_by_user_id(user_ids)

    rows: list[dict[str, Any]] = []
    for uid, name, birth, _tel in user_rows:
        uid_s = str(uid)
        latest = latest_map.get(uid_s)
        ad = _normalize_to_date(latest["assessment_date"] if latest else None)
        updated_attr = f"{ad.isoformat()} 00:00:00" if ad else ""
        hs = (latest["health_status"] if latest else None) or ""

        rows.append(
            {
                "pk": uid_s,
                "userID": uid_s,
                "userName": name,
                "age": _calc_age(_normalize_to_date(birth)),
                "height": latest["height"] if latest else None,
                "weight": latest["weight"] if latest else None,
                "bmi": latest["bmi"] if latest else None,
                "health_status": hs,
                "assessment_date_display": ad.strftime("%Y-%m-%d") if ad else None,
                "updated_at_attr": updated_attr,
            }
        )

    stats = {"user_count": len(rows)}
    return rows, stats
