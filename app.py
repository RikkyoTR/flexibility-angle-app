import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import math
from PIL import Image


# ==========================================
# ページ設定
# ==========================================

st.set_page_config(
    page_title="柔軟性角度測定",
    page_icon="📐",
    layout="centered"
)


# ==========================================
# タイトル
# ==========================================

st.title("📐 柔軟性角度測定")

st.write(
    "画像から上に上げている脚を自動判定し、"
    "その側の肩・股関節・足首から角度を測定します。"
)

st.info(
    "測定方法：肩 → 股関節 → 上げている側の足首"
    "の角度を計算し、180°から引きます。"
)


# ==========================================
# MediaPipe
# ==========================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ==========================================
# 角度計算
# ==========================================

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator == 0:
        return None

    cosine_angle = (
        np.dot(ba, bc)
        / denominator
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return angle


# ==========================================
# 画像アップロード
# ==========================================

uploaded_file = st.file_uploader(
    "測定する写真をアップロードしてください",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# アップロード後
# ==========================================

if uploaded_file is not None:

    # --------------------------------------
    # 画像読み込み
    # --------------------------------------

    pil_image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_rgb = np.array(
        pil_image
    )


    # --------------------------------------
    # MediaPipeで人体検出
    # --------------------------------------

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:

        results = pose.process(
            image_rgb
        )


    # ======================================
    # 人体検出成功
    # ======================================

    if results.pose_landmarks:

        landmarks = (
            results.pose_landmarks.landmark
        )


        # ==================================
        # 左右の足首
        # ==================================

        left_ankle = [
            landmarks[
                mp_pose.PoseLandmark.LEFT_ANKLE
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_ANKLE
            ].y
        ]

        right_ankle = [
            landmarks[
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ].y
        ]


        # ==================================
        # 上にある足首を判定
        # ==================================

        if left_ankle[1] < right_ankle[1]:

            # ------------------------------
            # 左脚が上
            # ------------------------------

            side = "LEFT"

            shoulder_landmark = (
                mp_pose.PoseLandmark.LEFT_SHOULDER
            )

            hip_landmark = (
                mp_pose.PoseLandmark.LEFT_HIP
            )

            ankle_landmark = (
                mp_pose.PoseLandmark.LEFT_ANKLE
            )

            side_text = "左"


        else:

            # ------------------------------
            # 右脚が上
            # ------------------------------

            side = "RIGHT"

            shoulder_landmark = (
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            )

            hip_landmark = (
                mp_pose.PoseLandmark.RIGHT_HIP
            )

            ankle_landmark = (
                mp_pose.PoseLandmark.RIGHT_ANKLE
            )

            side_text = "右"


        # ==================================
        # 選択された側の3点
        # ==================================

        shoulder = [
            landmarks[
                shoulder_landmark
            ].x,

            landmarks[
                shoulder_landmark
            ].y
        ]


        hip = [
            landmarks[
                hip_landmark
            ].x,

            landmarks[
                hip_landmark
            ].y
        ]


        ankle = [
            landmarks[
                ankle_landmark
            ].x,

            landmarks[
                ankle_landmark
            ].y
        ]


        # ==================================
        # 3点の角度
        # ==================================

        angle = calculate_angle(
            shoulder,
            hip,
            ankle
        )


        # ==================================
        # 結果計算
        # ==================================

        if angle is not None:

            flexibility_angle = (
                180 - angle
            )


            # ==================================
            # 画像に骨格を描画
            # ==================================

            annotated_image = (
                image_rgb.copy()
            )

            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )


            # ==================================
            # 画像サイズ
            # ==================================

            height, width = (
                annotated_image.shape[:2]
            )


            # ==================================
            # 座標を画像座標へ変換
            # ==================================

            shoulder_px = (
                int(shoulder[0] * width),
                int(shoulder[1] * height)
            )

            hip_px = (
                int(hip[0] * width),
                int(hip[1] * height)
            )

            ankle_px = (
                int(ankle[0] * width),
                int(ankle[1] * height)
            )


            # ==================================
            # 肩 → 股関節
            # ==================================

            cv2.line(
                annotated_image,
                shoulder_px,
                hip_px,
                (255, 0, 0),
                5
            )


            # ==================================
            # 股関節 → 足首
            # ==================================

            cv2.line(
                annotated_image,
                hip_px,
                ankle_px,
                (255, 0, 0),
                5
            )


            # ==================================
            # 股関節に角度表示
            # ==================================

            text = (
                f"{flexibility_angle:.1f} deg"
            )

            text_position = (
                hip_px[0] + 15,
                hip_px[1] - 20
            )

            cv2.putText(
                annotated_image,
                text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 0),
                3,
                cv2.LINE_AA
            )


            # ==================================
            # 結果
            # ==================================

            st.subheader("測定結果")


            st.metric(
                "柔軟性角度",
                f"{flexibility_angle:.1f}°"
            )


            st.write(
                f"上げている脚：{side_text}脚"
            )


            st.write(
                f"3点が作る角度："
                f"{angle:.1f}°"
            )


            st.write(
                f"180° − {angle:.1f}°"
                f" = {flexibility_angle:.1f}°"
            )


            # ==================================
            # 解析画像
            # ==================================

            st.subheader(
                "解析画像"
            )

            st.image(
                annotated_image,
                use_container_width=True
            )


        else:

            st.error(
                "角度を計算できませんでした。"
            )


    # ======================================
    # 人体検出失敗
    # ======================================

    else:

        st.error(
            "人体を検出できませんでした。"
            "できるだけ全身が写っている写真を使用してください。"
        )
