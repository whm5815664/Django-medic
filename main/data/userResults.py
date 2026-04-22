from __future__ import annotations

from datetime import datetime
from typing import Any

from django.core.paginator import EmptyPage, Page, Paginator
from django.utils import timezone

from main.models import Assessment, User


def _display_assessment_dt(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.strftime("%Y-%m-%d %H:%M:%S")


def build_user_results_context(user_id: str, page: int = 1, page_size: int = 20) -> dict[str, Any]:
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

    qs = list(
        Assessment.objects.filter(userID=user_id)
        .order_by("-assessment_date")
        .values(
            "assessment_date",
            "height",
            "weight",
            "bmi",
            "blood_pressure",
            "blood_sugar",
            "health_status",
        )
    )

    rows_all: list[dict[str, Any]] = [
        {**r, "assessment_date_display": _display_assessment_dt(r.get("assessment_date"))} for r in qs
    ]

    paginator = Paginator(rows_all, page_size)
    try:
        page_obj: Page[dict[str, Any]] = paginator.page(page)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages or 1)

    return {
        "user_id": user_id,
        "user": user,
        "rows": list(page_obj.object_list),
        "page_obj": page_obj,
        "now_display": timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": "" if user else "未找到该用户（userID 不存在）",
    }

