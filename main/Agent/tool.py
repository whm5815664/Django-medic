from __future__ import annotations

import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from main.models import Assessment

from .brain_agent import send_message_sse

_SESSION_SENT_USER_DATA: dict[str, set[str]] = {}


def get_patient_physiological_data(user_id: str) -> dict:
    """
    查询指定患者（userID）的前 30 条生理数据（Assessment），返回可 JSON 序列化的 dict。
    """
    user_id = (user_id or "").strip()
    if not user_id:
        return {"ok": False, "error": "user_id 不能为空", "rows": []}

    rows = list(
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
        )[:30]
    )

    for r in rows:
        dt = r.get("assessment_date")
        r["assessment_date"] = dt.isoformat(sep=" ", timespec="seconds") if dt else ""
        bs = r.get("blood_sugar")
        if bs is not None:
            r["blood_sugar"] = str(bs)

    return {"ok": True, "userID": user_id, "rows": rows}


@csrf_exempt
@require_POST
def agent_health_evolution_view(request):
    """健康演化：获取患者数据 + 预设提示词，然后发送给智能体（SSE 返回输出）。"""
    try:
        data = json.loads(request.body) or {}
        session_id = (data.get("session_id") or "").strip()
        user_id = (data.get("user_id") or "").strip()
        analysis_type = (data.get("analysis_type") or "health_evolution").strip()
        if not session_id:
            return JsonResponse({"success": False, "error": "缺少 session_id"})
        if not user_id:
            return JsonResponse({"success": False, "error": "缺少 user_id"})

        if analysis_type == "treatment_plan":
            PRESET_PROMPT = """
你是健康管理智能体，请根据“患者最近30条生理数据”输出【治疗方案建议】。

要求：
1) 先给出问题与风险点（引用数据依据），再给出建议。
2) 建议分层：生活方式 / 复测与随访 / 必要时就医与检查（用语谨慎，避免绝对化诊断）。
"""
        else:
            PRESET_PROMPT = """
你是健康管理智能体，请根据“患者最近30条生理数据”进行【健康演化分析】。

要求：
1) 用中文输出，给出总体趋势（身高/体重/BMI/血压/血糖/健康状况）。
2) 估计未来一段时间（30天）的身高/体重/BMI/血压/血糖/健康状况。
"""
        sent_users = _SESSION_SENT_USER_DATA.setdefault(session_id, set())
        already_sent = user_id in sent_users

        if already_sent:
            message = (
                PRESET_PROMPT.strip()
                + "\n\n注意：本会话中已提供过数据，请基于先前数据继续分析。"
            )
        else:
            payload = get_patient_physiological_data(user_id)
            message = PRESET_PROMPT.strip() + "\n\n患者数据(JSON)：\n" + json.dumps(payload, ensure_ascii=False)
            sent_users.add(user_id)

        return send_message_sse(session_id, message)

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})
