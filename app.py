import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os
import json
from datetime import timedelta

st.set_page_config(page_title="Warehouse HR Vision Assistant", layout="wide")

st.title("🏭 Warehouse HR Vision Assistant")
st.caption("Offline Streamlit demo for CCTV-led attendance, floor activity, productivity proxies, congestion, and workforce best-practice analytics.")

DEFAULT_VIDEO = Path("HRMS & Ground ops.mp4")

# ----------------------------
# Helpers
# ----------------------------

def save_uploaded(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def fmt_seconds(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def frame_zone_boxes(w: int, h: int):
    # Percent-based default zones for typical warehouse CCTV
    return {
        "entry": (int(0.02 * w), int(0.18 * h), int(0.18 * w), int(0.88 * h)),
        "work": (int(0.20 * w), int(0.18 * h), int(0.76 * w), int(0.84 * h)),
        "idle": (int(0.78 * w), int(0.18 * h), int(0.97 * w), int(0.88 * h)),
    }


def clamp_roi(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def contour_count(binary_mask, min_area=1200):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    areas = []
    for c in contours:
        a = cv2.contourArea(c)
        if a >= min_area:
            count += 1
            areas.append(a)
    return count, areas


def roi_activity(fgmask, box):
    x1, y1, x2, y2 = box
    roi = fgmask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0
    active_pixels = int(np.count_nonzero(roi))
    ratio = active_pixels / roi.size
    return ratio, active_pixels


def brightness(gray):
    return float(np.mean(gray))


def blur_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@st.cache_data(show_spinner=False)
def analyze_video(video_path: str, sample_every_sec: int = 2, max_minutes: int = 5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    duration_sec = frame_count / fps if fps else 0

    sample_step = max(1, int(fps * sample_every_sec))
    frame_limit = min(frame_count, int(max_minutes * 60 * fps)) if max_minutes > 0 else frame_count

    back_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)
    boxes = frame_zone_boxes(width, height)
    rows = []
    workforce_events = []
    last_entry_active = False

    idx = 0
    while idx < frame_limit:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = back_sub.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        ts = idx / fps if fps else 0.0
        frame_brightness = brightness(gray)
        frame_blur = blur_score(gray)

        zones = {name: clamp_roi(box, width, height) for name, box in boxes.items()}
        entry_ratio, _ = roi_activity(fgmask, zones["entry"])
        work_ratio, _ = roi_activity(fgmask, zones["work"])
        idle_ratio, _ = roi_activity(fgmask, zones["idle"])

        object_count, areas = contour_count(fgmask)
        mean_area = float(np.mean(areas)) if areas else 0.0

        # Heuristic proxies. These are non-identifying and demo-oriented.
        est_people = min(max(int(round(object_count * 0.7)), 0), 25)
        attendance_signal = 1 if entry_ratio > 0.035 or (object_count > 2 and work_ratio > 0.02) else 0
        active_work = 1 if work_ratio > 0.018 else 0
        idle_signal = 1 if idle_ratio > 0.014 and work_ratio < 0.02 else 0
        congestion_score = min(100.0, est_people * 4 + work_ratio * 900)
        productivity_proxy = min(100.0, (work_ratio * 2200) + (est_people * 2.2) - (idle_signal * 12))
        compliance_score = max(0.0, min(100.0, 78 + (8 if frame_brightness > 60 else -6) + (6 if frame_blur > 80 else -8) - (8 if congestion_score > 75 else 0)))

        if entry_ratio > 0.04 and not last_entry_active:
            workforce_events.append({"time": fmt_seconds(ts), "event": "Likely attendance / entry movement at gate"})
            last_entry_active = True
        elif entry_ratio <= 0.04:
            last_entry_active = False

        if congestion_score > 78:
            workforce_events.append({"time": fmt_seconds(ts), "event": "Possible floor congestion / supervisor review suggested"})
        if idle_signal and est_people >= 2:
            workforce_events.append({"time": fmt_seconds(ts), "event": "Idle-zone clustering observed"})
        if frame_brightness < 45:
            workforce_events.append({"time": fmt_seconds(ts), "event": "Low visibility conditions on camera feed"})

        rows.append({
            "time_sec": round(ts, 1),
            "time_label": fmt_seconds(ts),
            "estimated_people": est_people,
            "attendance_signal": attendance_signal,
            "active_work_signal": active_work,
            "idle_signal": idle_signal,
            "entry_activity_ratio": round(entry_ratio, 4),
            "work_activity_ratio": round(work_ratio, 4),
            "idle_activity_ratio": round(idle_ratio, 4),
            "productivity_proxy": round(productivity_proxy, 1),
            "congestion_score": round(congestion_score, 1),
            "compliance_score": round(compliance_score, 1),
            "brightness": round(frame_brightness, 1),
            "blur_score": round(frame_blur, 1),
            "motion_objects": object_count,
            "mean_motion_area": round(mean_area, 1),
        })

        idx += sample_step

    cap.release()
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No frames were analyzed from the selected video.")

    # Attendance sessions are approximated from sampled attendance signals.
    sampled_seconds = sample_every_sec
    on_floor_minutes = df["attendance_signal"].sum() * sampled_seconds / 60.0
    productive_minutes = df["active_work_signal"].sum() * sampled_seconds / 60.0
    idle_minutes = df["idle_signal"].sum() * sampled_seconds / 60.0

    summary = {
        "video_duration_sec": round(duration_sec, 1),
        "analyzed_duration_sec": round(df["time_sec"].max(), 1),
        "fps": round(fps, 2),
        "frame_width": width,
        "frame_height": height,
        "avg_people": round(df["estimated_people"].mean(), 1),
        "peak_people": int(df["estimated_people"].max()),
        "attendance_proxy_minutes": round(on_floor_minutes, 1),
        "productive_proxy_minutes": round(productive_minutes, 1),
        "idle_proxy_minutes": round(idle_minutes, 1),
        "avg_productivity_proxy": round(df["productivity_proxy"].mean(), 1),
        "avg_congestion_score": round(df["congestion_score"].mean(), 1),
        "avg_compliance_score": round(df["compliance_score"].mean(), 1),
        "visibility_risk_frames": int((df["brightness"] < 45).sum()),
        "low_clarity_frames": int((df["blur_score"] < 80).sum()),
        "recommended_hr_best_practices": [
            "Use CCTV analytics as a transparent workforce operations aid, not a sole disciplinary system.",
            "Publish policy notices: what is measured, why it matters, retention period, and access rights.",
            "Combine attendance proxies with badge, shift roster, or supervisor validation before action.",
            "Use productivity metrics for coaching, staffing, layout redesign, and congestion reduction.",
            "Review camera blind spots, lighting, and privacy-sensitive areas before scaling.",
        ],
    }

    return df, summary, workforce_events


def answer_query(query: str, df: pd.DataFrame, summary: dict, events: list[dict]) -> str:
    q = query.lower().strip()
    peak_row = df.loc[df["productivity_proxy"].idxmax()]
    cong_row = df.loc[df["congestion_score"].idxmax()]
    idle_row = df.loc[df["idle_signal"].idxmax()]

    if any(k in q for k in ["attendance", "present", "on floor", "absent"]):
        return (
            f"Attendance proxy suggests about {summary['attendance_proxy_minutes']} analyzed minutes with likely worker presence. "
            f"Average visible staffing proxy was {summary['avg_people']} people, with a peak of {summary['peak_people']}. "
            f"For HR action, validate against shift rosters, access control, or supervisor logs before concluding attendance."
        )
    if any(k in q for k in ["productivity", "productive", "performance", "output"]):
        return (
            f"Average productivity proxy was {summary['avg_productivity_proxy']}/100. The strongest observed activity window was around {peak_row['time_label']} "
            f"with productivity proxy {peak_row['productivity_proxy']} and estimated visible staffing {int(peak_row['estimated_people'])}. "
            f"This is best used for staffing and process coaching, not direct individual scoring."
        )
    if any(k in q for k in ["congestion", "crowd", "bottleneck", "busy"]):
        return (
            f"Peak congestion was {cong_row['congestion_score']}/100 around {cong_row['time_label']}. "
            f"Average congestion across the analyzed window was {summary['avg_congestion_score']}. "
            f"This usually points to lane conflicts, staging pileups, or supervisor intervention needs."
        )
    if any(k in q for k in ["idle", "waiting", "not working", "loiter"]):
        return (
            f"Estimated idle proxy time was {summary['idle_proxy_minutes']} minutes. "
            f"The strongest idle-zone signal appeared around {idle_row['time_label']}. "
            f"Review whether this reflects break clustering, waiting for work release, or layout inefficiency."
        )
    if any(k in q for k in ["best practice", "policy", "hr", "privacy", "ethic"]):
        return "Recommended HR and governance best practices: " + " ".join(summary["recommended_hr_best_practices"])
    if any(k in q for k in ["how is the workforce", "how are workers", "how is the warehouse team"]):
        return (
            f"The workforce appears moderately active overall: avg staffing proxy {summary['avg_people']}, avg productivity proxy {summary['avg_productivity_proxy']}, "
            f"avg congestion {summary['avg_congestion_score']}, and avg compliance/visibility score {summary['avg_compliance_score']}."
        )

    event_text = "; ".join([f"{e['time']} - {e['event']}" for e in events[:4]]) if events else "No major events were flagged in the analyzed sample."
    return (
        "Try questions like: 'How is attendance looking?', 'Was there congestion?', 'How productive was the floor?', or 'What HR best practices apply here?'. "
        f"Sample flagged events: {event_text}"
    )


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Configuration")
    sample_every_sec = st.slider("Analyze one frame every N seconds", 1, 10, 2)
    max_minutes = st.slider("Analyze up to N minutes", 6, 20, 6)
    uploaded_video = st.file_uploader("Optional: upload video", type=["mp4", "mov", "avi", "mkv"])
    st.markdown("**Metrics included**")
    st.write("Attendance proxy, floor activity, idle clustering, congestion, compliance/visibility, and productivity proxy.")

video_path = None
if uploaded_video is not None:
    video_path = save_uploaded(uploaded_video)
elif DEFAULT_VIDEO.exists():
    video_path = str(DEFAULT_VIDEO)

if not video_path:
    st.warning("Please upload a video from the sidebar or place 'HRMS & Ground ops.mp4' in the repo root.")
    st.stop()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("🎥 CCTV / Ground Operations Video")
    st.video(video_path, start_time=0, loop=True, autoplay=True, muted=True)
    components.html(
        """
        <script>
        const forceVideoPlayback = () => {
          const videos = window.parent.document.querySelectorAll("video");
          videos.forEach((video) => {
            video.setAttribute("autoplay", "");
            video.setAttribute("muted", "");
            video.setAttribute("loop", "");
            video.setAttribute("playsinline", "");
            video.setAttribute("webkit-playsinline", "true");
            video.muted = true;
            video.loop = true;
            video.playsInline = true;
            const playPromise = video.play();
            if (playPromise && typeof playPromise.catch === "function") {
              playPromise.catch(() => {});
            }
          });
        };

        forceVideoPlayback();
        setTimeout(forceVideoPlayback, 400);
        setInterval(forceVideoPlayback, 2500);
        </script>
        """,
        height=0,
    )

with st.spinner("Analyzing video for workforce and HR operations signals..."):
    df, summary, events = analyze_video(video_path, sample_every_sec, max_minutes)

if len(df) <= 5:
    st.error("The number of analyzed data points must be higher than 5. Increase analysis duration or lower sampling interval.")
    st.stop()

with right:
    st.subheader("📌 Executive Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Visible Staff", summary["avg_people"])
    c2.metric("Attendance Proxy (min)", summary["attendance_proxy_minutes"])
    c3.metric("Productivity Proxy", summary["avg_productivity_proxy"])
    c4, c5, c6 = st.columns(3)
    c4.metric("Idle Proxy (min)", summary["idle_proxy_minutes"])
    c5.metric("Congestion Score", summary["avg_congestion_score"])
    c6.metric("Compliance / Visibility", summary["avg_compliance_score"])

st.markdown("---")

# Charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Workforce Activity Over Time")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["time_sec"], df["estimated_people"], label="Estimated visible staff")
    ax.plot(df["time_sec"], df["productivity_proxy"], label="Productivity proxy")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader("🚦 Congestion and Idle Signals")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(df["time_sec"], df["congestion_score"], label="Congestion score")
    ax2.plot(df["time_sec"], df["idle_signal"] * 100, label="Idle signal x100")
    ax2.set_xlabel("Time (sec)")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

# Summary and best practices
left2, right2 = st.columns([1.1, 0.9])
with left2:
    st.subheader("🧾 Interpreted HR / Operations Summary")
    st.markdown(
        f"""
- **Analyzed duration:** `{fmt_seconds(summary['analyzed_duration_sec'])}` sampled from a `{fmt_seconds(summary['video_duration_sec'])}` video
- **Average staffing proxy:** `{summary['avg_people']}` visible workers, **peak:** `{summary['peak_people']}`
- **Attendance proxy:** `{summary['attendance_proxy_minutes']}` minutes with likely active presence
- **Productive-work proxy:** `{summary['productive_proxy_minutes']}` minutes
- **Idle / waiting proxy:** `{summary['idle_proxy_minutes']}` minutes
- **Camera visibility issues:** `{summary['visibility_risk_frames']}` low-light samples, `{summary['low_clarity_frames']}` low-clarity samples
        """
    )

    st.markdown("**Recommended HR best practices**")
    for item in summary["recommended_hr_best_practices"]:
        st.write(f"- {item}")

with right2:
    st.subheader("🚨 Flagged Workforce Events")
    if events:
        st.dataframe(pd.DataFrame(events[:25]), use_container_width=True, hide_index=True)
    else:
        st.info("No notable events were flagged in the analyzed sample.")

st.subheader("💬 Ask about this warehouse video")
example_prompts = [
    "How is attendance looking?",
    "How productive is the floor?",
    "Was there congestion?",
    "What HR best practices apply here?",
]
st.caption("Examples: " + " | ".join(example_prompts))
query = st.chat_input("Ask a question about attendance, productivity, congestion, or HR best practices...")
if query:
    st.chat_message("user").write(query)
    st.chat_message("assistant").write(answer_query(query, df, summary, events))

with st.expander("View analyzed data table"):
    st.dataframe(df, use_container_width=True, hide_index=True)

with st.expander("Export summary JSON"):
    st.code(json.dumps(summary, indent=2), language="json")
