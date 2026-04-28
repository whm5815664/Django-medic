import base64

import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont


ABNORMAL_EMOTIONS = {"sad", "angry", "fear", "disgust"}


def _cv2_put_text_unicode(img_bgr, text, org, font_size=22, color_bgr=(255, 255, 255)):
    """
    OpenCV 的 Hershey 字体不支持中文，使用 Pillow 渲染 Unicode 文本后再转回 OpenCV 图像。
    仅针对本项目当前 Windows 运行环境（微软雅黑）。
    """
    if img_bgr is None:
        return img_bgr

    x, y = int(org[0]), int(org[1])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", font_size)
    r, g, b = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])

    outline = (0, 0, 0)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=(r, g, b))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def get_bmi_info(profile=None):
    profile = profile or {}

    try:
        height = float(profile.get("height", 0))
        weight = float(profile.get("weight", 0))

        if height <= 0 or weight <= 0:
            return {
                "valid": False,
                "bmi": None,
                "bmi_category": None,
                "bmi_abnormal": False,
            }

        bmi = round(weight / ((height / 100) ** 2), 2)

        if bmi < 18.5:
            category = "偏瘦"
            abnormal = True
        elif bmi < 24:
            category = "正常"
            abnormal = False
        elif bmi < 28:
            category = "超重"
            abnormal = True
        else:
            category = "肥胖"
            abnormal = True

        return {
            "valid": True,
            "bmi": bmi,
            "bmi_category": category,
            "bmi_abnormal": abnormal,
        }
    except Exception:
        return {
            "valid": False,
            "bmi": None,
            "bmi_category": None,
            "bmi_abnormal": False,
        }


def get_blood_pressure_info(blood_pressure):
    if not blood_pressure:
        return {
            "valid": False,
            "systolic": None,
            "diastolic": None,
            "blood_pressure_category": None,
            "blood_pressure_abnormal": False,
        }

    try:
        bp = str(blood_pressure).strip().replace(" ", "")
        if "/" not in bp:
            raise ValueError("血压格式应为 120/80")

        systolic_str, diastolic_str = bp.split("/", 1)
        systolic = int(systolic_str)
        diastolic = int(diastolic_str)

        if systolic < 90 or diastolic < 60:
            category = "偏低"
            abnormal = True
        elif systolic >= 140 or diastolic >= 90:
            category = "偏高"
            abnormal = True
        else:
            category = "正常"
            abnormal = False

        return {
            "valid": True,
            "systolic": systolic,
            "diastolic": diastolic,
            "blood_pressure_category": category,
            "blood_pressure_abnormal": abnormal,
        }
    except Exception:
        return {
            "valid": False,
            "systolic": None,
            "diastolic": None,
            "blood_pressure_category": None,
            "blood_pressure_abnormal": False,
        }


def get_blood_sugar_info(blood_sugar):
    if blood_sugar in [None, ""]:
        return {
            "valid": False,
            "blood_sugar": None,
            "blood_sugar_category": None,
            "blood_sugar_abnormal": False,
        }

    try:
        sugar = float(blood_sugar)

        if sugar < 3.9:
            category = "偏低"
            abnormal = True
        elif sugar <= 6.1:
            category = "正常"
            abnormal = False
        else:
            category = "偏高"
            abnormal = True

        return {
            "valid": True,
            "blood_sugar": round(sugar, 2),
            "blood_sugar_category": category,
            "blood_sugar_abnormal": abnormal,
        }
    except Exception:
        return {
            "valid": False,
            "blood_sugar": None,
            "blood_sugar_category": None,
            "blood_sugar_abnormal": False,
        }


def analyze_emotion_logic(image_data, profile=None, blood_pressure=None, blood_sugar=None):
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = DeepFace.analyze(
        img,
        actions=['emotion'],
        enforce_detection=False,
        silent=True
    )

    dominant_emotion = ""
    face_location = None

    if isinstance(result, list):
        dominant_emotion = result[0].get('dominant_emotion', '')
        face_location = result[0].get('region')
    else:
        dominant_emotion = result.get('dominant_emotion', '')
        face_location = result.get('region')

    emotion_abnormal = dominant_emotion.lower() in ABNORMAL_EMOTIONS

    bmi_info = get_bmi_info(profile)
    bp_info = get_blood_pressure_info(blood_pressure)
    sugar_info = get_blood_sugar_info(blood_sugar)

    abnormal_reasons = []

    if bmi_info["valid"] and bmi_info["bmi_abnormal"]:
        abnormal_reasons.append(f"BMI异常（{bmi_info['bmi']}，{bmi_info['bmi_category']}）")

    if bp_info["valid"] and bp_info["blood_pressure_abnormal"]:
        abnormal_reasons.append(
            f"血压异常（{bp_info['systolic']}/{bp_info['diastolic']}，{bp_info['blood_pressure_category']}）"
        )

    if sugar_info["valid"] and sugar_info["blood_sugar_abnormal"]:
        abnormal_reasons.append(
            f"血糖异常（{sugar_info['blood_sugar']}，{sugar_info['blood_sugar_category']}）"
        )

    if abnormal_reasons:
        status = "不健康"
        final_reason = "；".join(abnormal_reasons)
    else:
        status = "不健康" if emotion_abnormal else "正常"
        normal_parts = []

        if bmi_info["valid"] and bmi_info["bmi"] is not None:
            normal_parts.append(f"BMI正常（{bmi_info['bmi']}）")
        if bp_info["valid"]:
            normal_parts.append(f"血压正常（{bp_info['systolic']}/{bp_info['diastolic']}）")
        if sugar_info["valid"] and sugar_info["blood_sugar"] is not None:
            normal_parts.append(f"血糖正常（{sugar_info['blood_sugar']}）")

        if emotion_abnormal:
            final_reason = (
                "；".join(normal_parts) + f"；情绪异常（{dominant_emotion}）"
                if normal_parts else f"情绪异常（{dominant_emotion}）"
            )
        else:
            final_reason = "；".join(normal_parts) if normal_parts else "未提供有效BMI/血压/血糖信息"

    box_color = (0, 0, 255) if status != "正常" else (0, 255, 0)

    if face_location:
        x = face_location.get('x', 0)
        y = face_location.get('y', 0)
        w = face_location.get('w', 0)
        h = face_location.get('h', 0)

        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 3)

        label = f"{status}: {dominant_emotion}"
        if bmi_info["bmi"] is not None:
            label += f" | BMI {bmi_info['bmi']}"
        if bp_info["valid"]:
            label += f" | BP {bp_info['systolic']}/{bp_info['diastolic']}"
        if sugar_info["blood_sugar"] is not None:
            label += f" | BS {sugar_info['blood_sugar']}"

        img = _cv2_put_text_unicode(
            img,
            label,
            (x, max(8, y - 28)),
            font_size=22,
            color_bgr=(255, 255, 255),
        )

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return_image_data = "data:image/jpeg;base64," + img_base64

    return {
        "success": True,
        "emotion": dominant_emotion,
        "status": status,
        "image": return_image_data,
        "bmi": bmi_info["bmi"],
        "bmi_category": bmi_info["bmi_category"],
        "bmi_abnormal": bmi_info["bmi_abnormal"],
        "blood_pressure": f"{bp_info['systolic']}/{bp_info['diastolic']}" if bp_info["valid"] else None,
        "blood_pressure_category": bp_info["blood_pressure_category"],
        "blood_pressure_abnormal": bp_info["blood_pressure_abnormal"],
        "blood_sugar": sugar_info["blood_sugar"],
        "blood_sugar_category": sugar_info["blood_sugar_category"],
        "blood_sugar_abnormal": sugar_info["blood_sugar_abnormal"],
        "reason": final_reason,
    }