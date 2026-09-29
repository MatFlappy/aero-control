from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
import time

import streamlit as st
import torch
from ultralytics import YOLO

from database import ViolationRecord, fetch_recent, init_db, save_violation
from stream import FrameStream, parse_source
from vision import detect, find_violations, save_snapshot


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = "runs/smoke_test/weights/best.pt"

st.set_page_config(page_title="АэроКонтроль", layout="wide")
st.markdown(
    """
<style>
.stApp {background:#eef2f5;color:#182637;font-family:"Segoe UI",sans-serif}
[data-testid="stHeader"] {background:transparent}
.block-container {max-width:1480px;padding-top:1.8rem;padding-bottom:3rem}
[data-testid="stSidebar"] {background:#ffffff;border-right:1px solid #d9e1e8}
[data-testid="stMetric"] {background:#fff;border:1px solid #dce4eb;border-radius:8px;padding:15px 18px}
[data-testid="stMetricLabel"] {color:#637489;font-size:13px}
[data-testid="stMetricValue"] {color:#0f2d44;font-size:28px;font-weight:650}
[data-testid="stButton"] button {border-radius:6px;min-height:40px;font-weight:550}
[data-testid="stButton"] button[kind="primary"] {background:#173a58;border-color:#173a58;color:white}
[data-testid="stImage"] img {border-radius:8px;border:1px solid #dce4eb}
h1,h2,h3 {color:#102a42;font-weight:650!important;letter-spacing:0}
.brand {font-size:23px;font-weight:700;color:#102a42;margin:4px 0}
.muted {color:#65768a;font-size:13px;line-height:1.6}
.section-label {color:#7c8b9a;font-size:10px;letter-spacing:1.6px;font-weight:700;margin:22px 0 10px}
.page-header {display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid #d7e0e8;padding-bottom:20px;margin-bottom:22px;gap:20px}
.page-header p {color:#65768a;font-size:14px;margin:6px 0 0}
.status-ok {border-left:4px solid #2d8a5f;background:#edf8f2;padding:11px 14px;border-radius:6px;color:#245f46}
.status-warn {border-left:4px solid #d27a32;background:#fff6ec;padding:11px 14px;border-radius:6px;color:#764a25}
.status-idle {border-left:4px solid #8da0b5;background:#f7f9fb;padding:11px 14px;border-radius:6px;color:#4f6278}
.empty {background:#fff;border:1px solid #dce4eb;border-radius:8px;text-align:center;padding:54px 24px;margin-top:10px}
.frame-meta {color:#65768a;font-size:12px;margin:8px 0 14px}
@media(max-width:700px){.page-header{display:block}.block-container{padding-top:1.2rem}.empty{padding:35px 15px}}
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
        raise FileNotFoundError(f"Файл весов не найден: {path}")

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
                <div class="muted">АЭРОКОНТРОЛЬ / ОПЕРАТОРСКАЯ ПАНЕЛЬ</div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <div class="muted">Прототип 0.3<br>HTTP stream + SQL Server</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown('<div class="brand">АэроКонтроль</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Мониторинг видеопотока с дрона и фиксация нарушений СИЗ.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">РАЗДЕЛ</div>', unsafe_allow_html=True)
    page = st.radio(
        "Раздел",
        ["Онлайн-мониторинг", "Архив нарушений", "Диагностика"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">ИСТОЧНИК</div>', unsafe_allow_html=True)
    stream_url = st.text_input(
        "HTTP-поток дрона",
        value="http://192.168.10.1:8080/video",
        help="Можно указать HTTP/MJPEG URL или номер локальной камеры: 0",
    )
    st.session_state.latitude = st.number_input("Широта", value=0.0, format="%.6f")
    st.session_state.longitude = st.number_input("Долгота", value=0.0, format="%.6f")

    st.markdown('<div class="section-label">МОДЕЛЬ</div>', unsafe_allow_html=True)
    choice = st.selectbox("Устройство обработки", ["CPU", "NVIDIA CUDA"])
    device = "cpu" if choice == "CPU" else 0
    confidence = st.slider("Порог уверенности", 0.1, 0.9, 0.35, 0.05)
    step = st.select_slider("Анализировать каждый N-й кадр", [1, 2, 5, 10, 15, 30], value=5)
    cooldown = st.slider("Пауза между фиксациями, сек", 5, 120, 20, 5)
    weights = st.text_input("Файл модели", DEFAULT_WEIGHTS)


if choice == "NVIDIA CUDA" and not torch.cuda.is_available():
    st.error("CUDA недоступна. Выбери CPU.")
    st.stop()


if page == "Архив нарушений":
    render_header(
        "Архив нарушений",
        "События, сохраненные в SQL Server: дата, тип нарушения, координаты и снимок.",
    )
    try:
        records = fetch_recent(100)
    except Exception as error:
        st.error(f"Не удалось прочитать SQL Server: {error}")
        st.stop()

    if not records:
        st.markdown(
            '<div class="empty"><h3>Архив пока пуст</h3><p>Когда система зафиксирует нарушение, запись появится здесь.</p></div>',
            unsafe_allow_html=True,
        )
    for record in records:
        title = f"#{record['Id']} · {record['ViolationType']} · {record['CreatedAtUtc']}"
        with st.expander(title):
            left, right = st.columns([3, 1])
            image_path = Path(record["AnnotatedImagePath"])
            if image_path.is_file():
                left.image(str(image_path), use_container_width=True)
            else:
                left.warning("Файл снимка не найден.")
            right.write("Источник:", record["Source"])
            right.write("Уверенность:", f"{record['Confidence']:.0%}")
            right.write("Статус:", record["Status"])
            right.write("Широта:", record["Latitude"])
            right.write("Долгота:", record["Longitude"])
            with st.expander("Метаданные"):
                st.json(record["MetadataJson"])
    st.stop()


if page == "Диагностика":
    render_header(
        "Диагностика",
        "Проверка модели, видеопотока и таблицы SQL Server перед реальным вылетом.",
    )
    a, b, c = st.columns(3)
    if a.button("Проверить модель", type="primary", use_container_width=True):
        try:
            get_model(weights, device)
            st.success("Модель загружена.")
        except Exception as error:
            st.error(str(error))
    if b.button("Создать таблицу SQL Server", use_container_width=True):
        try:
            init_db()
            st.success("Таблица dbo.Violations готова.")
        except Exception as error:
            st.error(f"Ошибка SQL Server: {error}")
    if c.button("Проверить HTTP-поток", use_container_width=True):
        try:
            stream = FrameStream(parse_source(stream_url))
            frame = stream.read()
            stream.close()
            if frame is None:
                st.error("Поток открылся, но кадр не получен.")
            else:
                st.image(frame, channels="BGR", use_container_width=True)
                st.success("Кадр получен.")
        except Exception as error:
            st.error(str(error))
    st.stop()


render_header(
    "Онлайн-мониторинг",
    "Подключи HTTP-поток дрона, система будет анализировать кадры и автоматически сохранять нарушения в SQL Server.",
)

controls, status = st.columns([2, 3])
with controls:
    start, stop = st.columns(2)
    if start.button("Подключить поток", type="primary", use_container_width=True):
        close_stream()
        reset_result()
        try:
            get_model(weights, device)
            st.session_state.stream = FrameStream(parse_source(stream_url))
            st.session_state.last_violation_saved_at = 0.0
        except Exception as error:
            close_stream()
            st.error(str(error))
    if stop.button("Остановить", use_container_width=True):
        close_stream()

with status:
    if st.session_state.get("stream") is None:
        st.markdown(
            '<div class="status-idle">Поток не подключен. Укажи HTTP URL дрона и нажми «Подключить поток».</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="status-ok">Поток активен. Идет обработка кадров.</div>', unsafe_allow_html=True)


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
                except Exception as error:
                    st.error(f"Нарушение найдено, но не сохранено в БД: {error}")
        except Exception as error:
            close_stream()
            st.error(f"Обработка остановлена: {error}")

    result = st.session_state.get("result")
    if result is None:
        st.markdown(
            '<div class="empty"><h3>Ожидание видеопотока</h3><p>После подключения здесь появится live-кадр с разметкой и список обнаруженных объектов.</p></div>',
            unsafe_allow_html=True,
        )
        return

    counts = Counter(row["class"] for row in result["rows"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Люди", counts["Person"])
    m2.metric("Каски", counts["Hardhat"])
    m3.metric("Нарушения", len(result["violations"]))
    m4.metric("Обработка кадра", f"{result['elapsed']:.2f} c")

    if result["violations"]:
        st.markdown(
            '<div class="status-warn">Обнаружены признаки нарушения. Снимок сохраняется с паузой, заданной в настройках.</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([3, 1.15])
    left.image(result["annotated"], channels="BGR", use_container_width=True)
    right.markdown(f'<div class="frame-meta">Источник: {escape(result["source"])}</div>', unsafe_allow_html=True)
    if st.session_state.get("last_saved_violation_id"):
        right.success(f"Последняя запись в БД: #{st.session_state.last_saved_violation_id}")

    table_rows = [
        {
            "Объект": row["label"],
            "Уверенность": f"{row['confidence']:.0%}",
            "Нарушение": "да" if row["is_violation"] else "нет",
        }
        for row in result["rows"]
    ]
    right.dataframe(table_rows, use_container_width=True, hide_index=True)


live_panel()
