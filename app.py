from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
import time

import streamlit as st
import torch
from ultralytics import YOLO

from database import ViolationRecord, fetch_recent, save_violation
from stream import FrameStream, parse_source
from vision import detect, find_violations, save_snapshot


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = "runs/smoke_test/weights/best.pt"
DEFAULT_STREAM = "rtmp://127.0.0.1/live/drone"

st.set_page_config(page_title="АэроКонтроль", page_icon="▣", layout="wide")
st.markdown(
    """
<style>
.stApp {background:#f2f5f7;color:#172536;font-family:"Segoe UI",sans-serif}
[data-testid="stHeader"] {background:transparent}
.block-container {max-width:1440px;padding-top:1.6rem;padding-bottom:2.6rem}
[data-testid="stSidebar"] {background:#ffffff;border-right:1px solid #dce4eb}
[data-testid="stMetric"] {background:#fff;border:1px solid #dce4eb;border-radius:8px;padding:14px 16px}
[data-testid="stMetricLabel"] {color:#647487;font-size:13px}
[data-testid="stMetricValue"] {color:#102a42;font-size:27px;font-weight:650}
[data-testid="stButton"] button {border-radius:6px;min-height:42px;font-weight:600}
[data-testid="stButton"] button[kind="primary"] {background:#153a57;border-color:#153a57;color:white}
[data-testid="stImage"] img {border-radius:8px;border:1px solid #dce4eb}
h1,h2,h3 {color:#102a42;font-weight:650!important;letter-spacing:0}
.brand {font-size:24px;font-weight:750;color:#102a42;margin:4px 0 2px}
.muted {color:#65768a;font-size:13px;line-height:1.55}
.section-label {color:#7c8b9a;font-size:10px;letter-spacing:1.5px;font-weight:700;margin:22px 0 10px}
.page-header {display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid #d7e0e8;padding-bottom:18px;margin-bottom:20px;gap:20px}
.page-header p {color:#65768a;font-size:14px;margin:6px 0 0}
.tag {font-size:12px;color:#52657a;background:#fff;border:1px solid #dce4eb;border-radius:999px;padding:7px 11px;white-space:nowrap}
.status-ok {border-left:4px solid #2d8a5f;background:#edf8f2;padding:11px 14px;border-radius:6px;color:#245f46}
.status-warn {border-left:4px solid #d27a32;background:#fff6ec;padding:11px 14px;border-radius:6px;color:#764a25}
.status-idle {border-left:4px solid #8da0b5;background:#f7f9fb;padding:11px 14px;border-radius:6px;color:#4f6278}
.empty {background:#fff;border:1px solid #dce4eb;border-radius:8px;text-align:center;padding:52px 24px;margin-top:10px}
.frame-meta {color:#65768a;font-size:12px;margin:8px 0 14px}
@media(max-width:700px){.page-header{display:block}.tag{display:inline-block;margin-top:12px}.block-container{padding-top:1.1rem}.empty{padding:34px 15px}}
</style>
""",
    unsafe_allow_html=True,
)


def close_stream():
    stream = st.session_state.pop("stream", None)
    if stream is not None:
        stream.close()


def reset_result():
    st.session_state.pop("result", None)


def get_model(weights: str, device):
    path = (ROOT / weights).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Файл модели не найден: {path}")

    signature = (str(path), path.stat().st_mtime_ns, device)
    if st.session_state.get("model_signature") != signature:
        st.session_state.model = YOLO(str(path))
        st.session_state.model_signature = signature
    return st.session_state.model


def get_current_coordinates():
    lat = st.session_state.get("latitude")
    lon = st.session_state.get("longitude")
    if lat in (None, 0.0) or lon in (None, 0.0):
        return None, None
    return float(lat), float(lon)


def analyze_frame(frame, source, weights, confidence, device):
    model = get_model(weights, device)
    started = time.perf_counter()
    annotated, rows = detect(model, frame, confidence, device)
    result = {
        "original": frame,
        "annotated": annotated,
        "rows": rows,
        "violations": find_violations(rows),
        "elapsed": time.perf_counter() - started,
        "source": source,
        "weights": weights,
        "threshold": confidence,
        "frame_time": datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state.result = result
    return result


def persist_violation(result, cooldown_seconds):
    if not result["violations"]:
        return None

    now = time.monotonic()
    last_saved = st.session_state.get("last_violation_saved_at", 0.0)
    if now - last_saved < cooldown_seconds:
        return None

    latitude, longitude = get_current_coordinates()
    metadata = {
        "source": result["source"],
        "weights": result["weights"],
        "threshold": result["threshold"],
        "rows": result["rows"],
        "violations": result["violations"],
        "latitude": latitude,
        "longitude": longitude,
    }
    folder = save_snapshot(ROOT, result["original"], result["annotated"], metadata)
    violation = max(result["violations"], key=lambda item: item["confidence"])

    record_id = save_violation(
        ViolationRecord(
            source=result["source"],
            violation_type=violation["label"],
            confidence=violation["confidence"],
            latitude=latitude,
            longitude=longitude,
            original_image_path=folder / "original.jpg",
            annotated_image_path=folder / "annotated.jpg",
            metadata=metadata,
        )
    )
    st.session_state.last_violation_saved_at = now
    st.session_state.last_saved_violation_id = record_id
    return record_id


def render_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <div class="muted">АЭРОКОНТРОЛЬ</div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <div class="tag">DJI Mini 2 SE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown('<div class="brand">АэроКонтроль</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Онлайн-контроль строительной площадки с автоматической фиксацией нарушений.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">РАЗДЕЛ</div>', unsafe_allow_html=True)
    page = st.radio("Раздел", ["Мониторинг", "Журнал нарушений"], label_visibility="collapsed")

    st.markdown('<div class="section-label">ПОДКЛЮЧЕНИЕ</div>', unsafe_allow_html=True)
    stream_url = st.text_input(
        "Адрес видеопотока",
        value=DEFAULT_STREAM,
        help="Для DJI Mini 2 SE обычно используется RTMP-адрес локального сервера.",
    )
    st.session_state.latitude = st.number_input("Широта", value=0.0, format="%.6f")
    st.session_state.longitude = st.number_input("Долгота", value=0.0, format="%.6f")

    with st.expander("Настройки распознавания"):
        choice = st.selectbox("Обработка", ["CPU", "NVIDIA CUDA"])
        device = "cpu" if choice == "CPU" else 0
        confidence = st.slider("Чувствительность", 0.1, 0.9, 0.35, 0.05)
        step = st.select_slider("Частота анализа", [1, 2, 5, 10, 15, 30], value=5)
        cooldown = st.slider("Пауза между снимками, сек", 5, 120, 20, 5)
        weights = st.text_input("Файл модели", DEFAULT_WEIGHTS)


if choice == "NVIDIA CUDA" and not torch.cuda.is_available():
    st.error("Видеокарта NVIDIA CUDA недоступна. Выбери CPU в настройках распознавания.")
    st.stop()


if page == "Журнал нарушений":
    render_header(
        "Журнал нарушений",
        "Здесь сохраняются снимки, время, координаты и тип обнаруженного нарушения.",
    )
    try:
        records = fetch_recent(100)
    except Exception:
        st.error("Не удалось открыть журнал. Проверь, что SQL Server запущен и приложение открыто обычным терминалом Windows.")
        st.stop()

    if not records:
        st.markdown(
            '<div class="empty"><h3>Нарушений пока нет</h3><p>Когда система обнаружит отсутствие каски, маски или жилета, запись появится здесь.</p></div>',
            unsafe_allow_html=True,
        )

    for record in records:
        created = record["CreatedAtUtc"]
        title = f"#{record['Id']} · {record['ViolationType']} · {created}"
        with st.expander(title):
            left, right = st.columns([3, 1])
            image_path = Path(record["AnnotatedImagePath"])
            if image_path.is_file():
                left.image(str(image_path), use_container_width=True)
            else:
                left.warning("Снимок не найден на диске.")
            right.write("Уверенность:", f"{record['Confidence']:.0%}")
            right.write("Широта:", record["Latitude"] or "не указана")
            right.write("Долгота:", record["Longitude"] or "не указана")
            right.write("Статус:", "требует проверки")
    st.stop()


render_header(
    "Мониторинг",
    "Подключи видеопоток, запусти обработку и следи за ситуацией в реальном времени.",
)

controls, status = st.columns([2, 3])
with controls:
    start, stop = st.columns(2)
    if start.button("Начать мониторинг", type="primary", use_container_width=True):
        close_stream()
        reset_result()
        try:
            get_model(weights, device)
            st.session_state.stream = FrameStream(parse_source(stream_url))
            st.session_state.last_violation_saved_at = 0.0
            st.session_state.last_saved_violation_id = None
        except Exception as error:
            close_stream()
            st.error(f"Не удалось подключить видеопоток: {error}")
    if stop.button("Остановить", use_container_width=True):
        close_stream()

with status:
    if st.session_state.get("stream") is None:
        st.markdown(
            '<div class="status-idle">Поток не подключен. Укажи адрес видеопотока и нажми «Начать мониторинг».</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="status-ok">Мониторинг активен. Кадры анализируются автоматически.</div>', unsafe_allow_html=True)


@st.fragment(run_every=0.4)
def live_panel():
    stream = st.session_state.get("stream")
    if stream is not None:
        try:
            frame = stream.read()
            if frame is None:
                close_stream()
                st.warning("Поток остановлен: кадр не получен.")
            else:
                result = analyze_frame(frame, stream_url, weights, confidence, device)
                stream.skip(step - 1)
                try:
                    persist_violation(result, cooldown)
                except Exception:
                    st.error("Нарушение найдено, но запись в журнал не сохранена. Проверь подключение к базе.")
        except Exception as error:
            close_stream()
            st.error(f"Обработка остановлена: {error}")

    result = st.session_state.get("result")
    if result is None:
        st.markdown(
            '<div class="empty"><h3>Ожидание видеопотока</h3><p>После запуска здесь появится изображение с разметкой и краткая сводка по объектам.</p></div>',
            unsafe_allow_html=True,
        )
        return

    counts = Counter(row["class"] for row in result["rows"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Люди", counts["Person"])
    m2.metric("Каски", counts["Hardhat"])
    m3.metric("Нарушения", len(result["violations"]))
    m4.metric("Кадр", f"{result['elapsed']:.2f} c")

    if result["violations"]:
        st.markdown(
            '<div class="status-warn">Обнаружены признаки нарушения. Снимок сохранится в журнал автоматически.</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([3, 1.15])
    left.image(result["annotated"], channels="BGR", use_container_width=True)
    right.markdown(f'<div class="frame-meta">Источник: {escape(result["source"])}</div>', unsafe_allow_html=True)
    if st.session_state.get("last_saved_violation_id"):
        right.success(f"Запись сохранена: #{st.session_state.last_saved_violation_id}")

    table_rows = [
        {
            "Объект": row["label"],
            "Уверенность": f"{row['confidence']:.0%}",
            "Статус": "нарушение" if row["is_violation"] else "норма",
        }
        for row in result["rows"]
    ]
    right.dataframe(table_rows, use_container_width=True, hide_index=True)


live_panel()
