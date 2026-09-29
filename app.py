from pathlib import Path
from collections import Counter
import json
import tempfile
import time
from datetime import datetime
from html import escape

import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO

from vision import detect, save_snapshot

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='АэроКонтроль', layout='wide')
st.markdown('''<style>
.stApp {background:#f4f5f7;color:#243447;font-family:"Segoe UI",sans-serif}
[data-testid="stHeader"] {background:transparent}
.block-container {max-width:1440px;padding-top:2.2rem;padding-bottom:3rem}
h1,h2,h3 {color:#122e48;font-weight:600!important;letter-spacing:-.025em}
h1 {font-size:2.3rem!important}
h3 {font-size:1.2rem!important}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid #e0e5eb}
[data-testid="stSidebar"] .block-container {padding-top:1.5rem}
[data-testid="stMetric"] {background:#fff;border:1px solid #e0e5eb;border-radius:8px;padding:18px 20px}
[data-testid="stMetricLabel"] {color:#647487;font-size:13px}
[data-testid="stMetricValue"] {color:#122e48;font-size:30px;font-weight:600}
[data-testid="stButton"] button {border-radius:6px;min-height:42px;font-weight:500}
[data-testid="stButton"] button[kind="primary"] {background:#173a58;border-color:#173a58;color:white}
[data-testid="stButton"] button[kind="primary"]:hover {background:#245373;border-color:#245373}
[data-testid="stFileUploader"] {background:white;border:1px solid #dce2e9;border-radius:8px;padding:16px}
[data-testid="stFileUploaderDropzone"] {background:#f7f8fa;border-radius:6px}
[data-testid="stImage"] img {border-radius:8px}
[data-testid="stExpander"] {background:#fff;border-radius:8px}
[data-testid="stTextInput"] input {background:#f8f9fb}
.brand {font-size:22px;font-weight:650;color:#122e48;letter-spacing:-.7px;margin:6px 0}
.brand:before {content:"";display:inline-block;width:5px;height:21px;background:#db743c;margin-right:10px;vertical-align:-3px}
.muted {color:#728092;font-size:12px;line-height:1.7}
.section-label {color:#83909e;font-size:10px;letter-spacing:1.7px;font-weight:650;margin:24px 0 12px}
.page-header {display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid #dce2e9;padding-bottom:24px;margin-bottom:26px;gap:20px}
.page-header h1 {margin:4px 0 8px;padding:0}
.page-header p {color:#68788b;font-size:14px;margin:0}
.eyebrow {font-size:10px;letter-spacing:2px;color:#9b633f;font-weight:650}
.version {font-size:11px;color:#718095;border:1px solid #d8dfe7;padding:7px 11px;border-radius:4px;white-space:nowrap}
.notice {border-left:3px solid #dba06e;padding:10px 15px;background:#fbf6ef;color:#73543b;font-size:13px;margin:14px 0 20px}
.empty {background:#fff;border:1px solid #e0e5eb;border-radius:8px;text-align:center;padding:65px 24px;margin-top:24px}
.empty h3 {margin-bottom:10px}.empty p {font-size:14px;color:#758194;max-width:480px;margin:auto;line-height:1.7}
.frame-meta {color:#728092;font-size:12px;margin:10px 0 18px}
@media(max-width:700px){.page-header {display:block}.version {display:none}.block-container{padding-top:1.5rem}.empty{padding:35px 15px}}
</style>''', unsafe_allow_html=True)


class VideoSession:
    def __init__(self, upload):
        self.temp = tempfile.TemporaryDirectory(prefix='aerocontrol_')
        self.path = Path(self.temp.name) / ('input' + Path(upload.name).suffix.lower())
        self.path.write_bytes(upload.getbuffer())
        self.cap = cv2.VideoCapture(str(self.path))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.index = 0
        if not self.cap.isOpened():
            self.close()
            raise ValueError('Видео не открывается. Попробуй MP4 с кодеком H.264.')

    def close(self):
        if getattr(self, 'cap', None) is not None:
            self.cap.release()
        if getattr(self, 'temp', None) is not None:
            self.temp.cleanup()

    def __del__(self):
        self.close()


def stop_video():
    video = st.session_state.pop('video', None)
    if video is not None:
        video.close()


def reset():
    stop_video()
    st.session_state.pop('result', None)


def get_model():
    path = (ROOT / weights).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Файл весов не найден: {path}')
    # One model per browser session; never share a mutable predictor across users.
    signature = (str(path), path.stat().st_mtime_ns, device)
    if st.session_state.get('model_signature') != signature:
        st.session_state.model = YOLO(str(path))
        st.session_state.model_signature = signature
    return st.session_state.model


def analyze(frame, source, video_time=None):
    model = get_model()
    started = time.perf_counter()
    annotated, rows = detect(model, frame, confidence, device)
    st.session_state.result = dict(
        original=frame, annotated=annotated, rows=rows,
        elapsed=time.perf_counter() - started, source=source,
        video_time=video_time, weights=weights, threshold=confidence)


with st.sidebar:
    st.markdown('<div class="brand">АэроКонтроль</div><div class="muted">Визуальный контроль средств защиты</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">РАБОЧАЯ ОБЛАСТЬ</div>', unsafe_allow_html=True)
    page = st.radio('Раздел', ['Анализ', 'Сохранённые снимки'], on_change=reset, label_visibility='collapsed')
    st.markdown('<div class="section-label">ПАРАМЕТРЫ АНАЛИЗА</div>', unsafe_allow_html=True)
    choice = st.selectbox('Устройство обработки', ['CPU', 'NVIDIA CUDA'], on_change=reset)
    device = 'cpu' if choice == 'CPU' else 0
    confidence = st.slider('Порог уверенности', 0.1, 0.9, 0.35, 0.05, on_change=reset,
                           help='Более высокий порог скрывает менее уверенные находки, но может увеличить пропуски.')
    step = st.select_slider('Анализировать каждый N-й кадр', [1, 2, 5, 10, 15, 30], value=5,
                            on_change=reset)
    st.caption('При шаге больше 1 часть кадров не проверяется.')
    with st.expander('Модель и веса', expanded=True):
        weights = st.text_input('Файл модели', 'runs/smoke_test/weights/best.pt', on_change=reset)
        st.caption('Во время обучения выбери CPU и веса пробного запуска smoke_test.')
    st.divider()
    st.markdown('<div class="muted">Версия 0.2<br>Локальная обработка фото и видео</div>', unsafe_allow_html=True)

title = 'Анализ материалов' if page == 'Анализ' else 'Сохранённые снимки'
subtitle = ('Обнаружение людей, техники и средств индивидуальной защиты.' if page == 'Анализ'
            else 'Кадры и результаты распознавания, сохранённые для проверки.')
st.markdown(f'<div class="page-header"><div><div class="eyebrow">АЭРОКОНТРОЛЬ / РАБОЧАЯ ОБЛАСТЬ</div>'
            f'<h1>{title}</h1><p>{subtitle}</p></div><span class="version">ПРОТОТИП 0.2</span></div>',
            unsafe_allow_html=True)

if page == 'Сохранённые снимки':
    records = sorted((ROOT / 'storage' / 'snapshots').glob('*/metadata.json'),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not records:
        st.markdown('<div class="empty"><h3>Здесь появятся сохранённые кадры</h3><p>Проанализируй материал и сохрани нужный снимок. Он останется доступен после перезапуска приложения.</p></div>', unsafe_allow_html=True)
    else:
        st.caption(f'Записей: {len(records)} · Показаны последние {min(len(records), 100)}')
    for path in records[:100]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            saved = datetime.fromisoformat(data['saved_at_utc']).strftime('%d.%m.%Y · %H:%M:%S UTC')
            with st.expander(f"{saved} — {data['source']}"):
                picture, details = st.columns([3, 1])
                picture.image(str(path.parent / 'annotated.jpg'), use_container_width=True)
                with details:
                    st.markdown('**Требует проверки**')
                    st.caption('Время выше — время сохранения снимка.')
                    st.write('Источник:', data['source'])
                    st.write('Порог:', data.get('threshold', '—'))
                    if data.get('video_time') is not None:
                        st.write(f"Позиция: {data['video_time']:.2f} с")
                    st.caption('Координаты не определены')
                with st.expander('Подробные данные'):
                    st.json(data)
        except (OSError, ValueError, KeyError) as error:
            st.warning(f'Не удалось прочитать запись: {error}')
    st.stop()

st.subheader('Исходный материал')
upload = st.file_uploader('Выбери фотографию или видеозапись', type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
                          on_change=reset, help='JPG, PNG, MP4, AVI или MOV. До 200 МБ при стандартных настройках Streamlit.')
st.markdown('<div class="notice">Находки модели требуют проверки. Наличие каски в кадре не подтверждает, что она надета на каждого человека.</div>', unsafe_allow_html=True)
if choice == 'NVIDIA CUDA' and not torch.cuda.is_available():
    st.error('CUDA недоступна. Выбери CPU.')
    st.stop()

is_video = upload is not None and Path(upload.name).suffix.lower() in {'.mp4', '.avi', '.mov'}
c1, c2, _ = st.columns([1.4, 1.4, 4])
if c1.button('Начать анализ', type='primary', disabled=upload is None, use_container_width=True):
    reset()
    try:
        get_model()
        if is_video:
            st.session_state.video = VideoSession(upload)
        else:
            frame = cv2.imdecode(np.frombuffer(upload.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Не удалось прочитать изображение')
            analyze(frame, upload.name)
    except Exception as error:
        reset()
        st.error(str(error))
if c2.button('Остановить видео', use_container_width=True):
    stop_video()


@st.fragment(run_every=0.5)
def preview():
    video = st.session_state.get('video')
    if video is not None:
        try:
            ret, frame = video.cap.read()
            if not ret:
                stop_video()
                st.success('Обработка завершена.')
            else:
                position = video.index / video.fps if video.fps > 0 else None
                analyze(frame, upload.name, position)
                video.index += 1
                for _ in range(step - 1):
                    if not video.cap.grab():
                        break
                    video.index += 1
                if video.total > 0:
                    st.progress(min(video.index / video.total, 1.0))
        except Exception as error:
            stop_video()
            st.error(f'Обработка остановлена: {error}')
    result = st.session_state.get('result')
    if result is None:
        heading = 'Материал готов к проверке' if upload is not None else 'Загрузи первый материал'
        st.markdown(f'<div class="empty"><div class="eyebrow">ОБЛАСТЬ ПРОСМОТРА</div><h3>{heading}</h3>'
                    '<p>После запуска здесь появятся изображение с разметкой и список обнаруженных объектов.</p></div>', unsafe_allow_html=True)
        return
    st.divider()
    st.subheader('Результат анализа')
    counts = Counter(row['class'] for row in result['rows'])
    a, b, c, d = st.columns(4)
    a.metric('Люди в кадре', counts['Person'])
    b.metric('Каски в кадре', counts['Hardhat'])
    c.metric('Признаки отсутствия каски', counts['NO-Hardhat'])
    d.metric('Транспорт и техника', counts['vehicle'] + counts['machinery'])
    st.markdown(f'<div class="frame-meta">Источник: {escape(result["source"])}</div>', unsafe_allow_html=True)
    display = st.radio('Отображение', ['С разметкой', 'Исходный кадр'], horizontal=True, key='frame_display', label_visibility='collapsed')
    st.image(result['annotated'] if display == 'С разметкой' else result['original'], channels='BGR', use_container_width=True)
    st.caption(f"Обработка кадра: {result['elapsed']:.2f} с • Порог: {result['threshold']:.2f}")
    if result['video_time'] is not None:
        st.caption(f"Позиция в видео: {result['video_time']:.2f} с (не время съёмки)")
    suspects = [r for r in result['rows'] if r['class'].startswith('NO-')]
    if suspects:
        st.warning('Найдены признаки отсутствия средств защиты. Требуется проверка.')
    else:
        st.caption('Классы отсутствия защиты не обнаружены. Это не подтверждает безопасность.')
    with st.expander(f'Обнаруженные объекты — {len(result["rows"])}', expanded=True):
        if result['rows']:
            st.dataframe([{'Объект': r['Объект'], 'Уверенность': f"{r['Уверенность']:.0%}"}
                          for r in result['rows']], use_container_width=True, hide_index=True)
        else:
            st.caption('Объекты при выбранном пороге не найдены.')
    if st.button('Сохранить текущий снимок', disabled=st.session_state.get('video') is not None):
        try:
            metadata = {k: result[k] for k in ['source', 'video_time', 'weights', 'threshold', 'rows']}
            folder = save_snapshot(ROOT, result['original'], result['annotated'], metadata)
            st.success(f'Сохранено: {folder.relative_to(ROOT)}')
        except Exception as error:
            st.error(f'Не удалось сохранить: {error}')
    if st.session_state.get('video') is not None:
        st.caption('Для сохранения выбранного кадра сначала нажми «Остановить видео».')


preview()
