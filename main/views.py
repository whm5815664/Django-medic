from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse

import os
import json
import base64
import numpy as np
import cv2
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from deepface import DeepFace


from main.data.dashboard import build_dashboard_rows
from main.data.addUser import add_user_from_request, delete_user_and_assessments
from main.data.userResults import build_user_results_context, delete_assessment_by_user_and_dt
from main.SRGA.SRGA_form import srga_submit
from main.SRGA.SRGA_form import srga_reset_temp
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.clickjacking import xframe_options_sameorigin
def dashboard(request):
    """仪表板页面视图"""
    dashboard_rows, dashboard_stats = build_dashboard_rows()
    return render(
        request,
        "main/dashboard.html",
        {
            "dashboard_rows": dashboard_rows,
            "dashboard_stats": dashboard_stats,
        },
    )


def user_results(request):
    page = int(request.GET.get("page", "1") or "1")
    context = build_user_results_context(
        request.GET.get("user_id", ""),
        page=page,
        page_size=20,
        start_date=request.GET.get("start_date", ""),
        end_date=request.GET.get("end_date", ""),
        health_status=request.GET.get("health_status", ""),
    )
    # 统一使用实际在前端维护的模板文件名（useResults.html）
    return render(request, "userResults/useResults.html", context)


@xframe_options_sameorigin
def agent_view(request):
    return render(request, "agent/agent.html")


from django.views.decorators.http import require_POST


@require_POST
def user_result_delete(request):
    try:
        payload = json.loads((request.body or b"{}").decode("utf-8"))
        user_id = (payload.get("user_id") or "").strip()
        assessment_date = (payload.get("assessment_date") or "").strip()
        deleted = delete_assessment_by_user_and_dt(user_id=user_id, assessment_date=assessment_date)
        return JsonResponse({"ok": True, "deleted": deleted})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

def add_user_api(request):
    try:
        payload = add_user_from_request(request)
        return JsonResponse({"ok": True, "data": payload})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)


@require_POST
def delete_user_api(request):
    try:
        payload = json.loads((request.body or b"{}").decode("utf-8"))
        user_id = (payload.get("user_id") or "").strip()
        data = delete_user_and_assessments(user_id=user_id)
        return JsonResponse({"ok": True, "data": data})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)


def srga_record_form(request):
    """SRGA 健康评估：身高/体重 + 摄像头采集表单页"""
    #return render(request, "SRGA/srga_form.html")
    user_id = request.GET.get("user_id", "")
    name = request.GET.get("name", "")
    age = request.GET.get("age", "")

    return render(request, "SRGA/srga_phone.html", {
        "user_id": user_id,
        "name": name,
        "age": age,
    })


def srga_result(request):
    """
    SRGA 推理结果页：根据 user_id 读取 temp 中的图像/音频，并用表单的身高体重（可选年龄）做推理。
    """
    user_id = (request.GET.get("user_id") or "").strip()
    name = (request.GET.get("name") or "").strip()
    collect_time = (request.GET.get("collect_time") or "").strip()

    def _to_float(v, default=0.0):
        try:
            return float(str(v).strip())
        except Exception:
            return float(default)

    height_cm = _to_float(request.GET.get("height_cm"), 0.0)
    weight_kg = _to_float(request.GET.get("weight_kg"), 0.0)
    age = _to_float(request.GET.get("age"), 0.0)

    context = {
        "user_id": user_id,
        "name": name,
        "collect_time": collect_time,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "age": age,
    }

    if not user_id:
        context["error"] = "缺少 user_id，无法推理。请从采集页面提交后进入结果页。"
        return render(request, "SRGA/srga_result.html", context)
    if height_cm <= 0 or weight_kg <= 0:
        context["error"] = "身高/体重不合法（必须为正数），无法推理。"
        return render(request, "SRGA/srga_result.html", context)

    try:
        base_dir = str(settings.BASE_DIR / "main" / "SRGA")
        try:
            from main.SRGA.function import run_srga_inference
        except Exception as e:
            raise RuntimeError(
                f"SRGA 推理依赖未就绪（可能未安装 torch/torchaudio/torchvision 等）。原始错误：{e}"
            )

        result = run_srga_inference(
            base_dir=base_dir,
            user_id=user_id,
            height_cm=height_cm,
            weight_kg=weight_kg,
            age=age,
        )
        context["result"] = result
    except Exception as e:
        context["error"] = f"推理失败：{e}"

    return render(request, "SRGA/srga_result.html", context)



import traceback

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .services.assessment_service import analyze_and_save_assessment



def index(request):
    return render(request, 'index.html')


@csrf_exempt
@require_POST
def analyze(request):
    try:
        result = analyze_and_save_assessment(request.body)
        return JsonResponse(result)
    except ValueError as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e)}, status=500)