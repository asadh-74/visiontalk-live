"""VisionTalk Live - Streamlit demo: camera + (optional YOLO) + Gemini VLM + talking avatar."""
import os, io, time, base64, asyncio
from collections import Counter
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from google import genai
from google.genai import types
import edge_tts

st.set_page_config(page_title="VisionTalk Live", page_icon="🎥", layout="wide")
MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
LANGS = {"English": ("English", "en-US-AriaNeural"), "Urdu": ("Urdu", "ur-PK-UzmaNeural"),
         "Arabic": ("Arabic", "ar-SA-ZariyahNeural"), "Spanish": ("Spanish", "es-ES-ElviraNeural"),
         "French": ("French", "fr-FR-DeniseNeural"), "Hindi": ("Hindi", "hi-IN-SwaraNeural")}
PERSONAS = {"Host": "You are Aria, a friendly live-stream host.",
            "Receptionist": "You are a polite virtual receptionist.",
            "Tutor": "You are a patient tutor. Explain simply.",
            "Sales": "You are an enthusiastic sales assistant."}
BASE = ("You can SEE the viewer's camera frame. Answer in 1-3 short spoken sentences, no markdown, "
        "no emojis. Describe only what is visible; if unsure, say so.")

CARTOON = """<svg viewBox="0 0 200 220" style="width:100%;display:block;background:#181c30">
<path d="M30 120C20 20 180 20 170 120C165 70 35 70 30 120Z" fill="#2d1b3d"/><rect x="85" y="165" width="30" height="35" fill="#f3b392"/>
<path d="M40 220C45 190 155 190 160 220Z" fill="#6c5ce7"/><ellipse cx="100" cy="110" rx="58" ry="68" fill="#f7c4a5"/>
<path d="M38 105C45 40 155 40 162 105C140 70 60 70 38 105Z" fill="#2d1b3d"/>
<ellipse cx="78" cy="108" rx="8" ry="10" fill="#fff"/><ellipse cx="122" cy="108" rx="8" ry="10" fill="#fff"/>
<circle cx="78" cy="110" r="5" fill="#3b2a1e"/><circle cx="122" cy="110" r="5" fill="#3b2a1e"/>
<ellipse id="m" cx="100" cy="158" rx="14" ry="2" fill="#a8324a"/></svg>"""
TEMPLATE = """<body style="margin:0;background:transparent">
<div id="w" style="position:relative;max-width:420px;margin:auto;border-radius:16px;overflow:hidden">%FACE%</div>%AUDIO%
<script>
const m=document.getElementById('m'),isS=m.tagName.toLowerCase()==='ellipse',W=document.getElementById('w').clientWidth;
let sp=false;const a=document.getElementById('a');
if(a){a.onplay=()=>sp=true;a.onended=a.onpause=()=>sp=false;}
setInterval(()=>{const k=sp?0.1+Math.random()*0.6:0.03;
 if(isS){m.setAttribute('ry',2+k*20);m.setAttribute('rx',sp?10+Math.random()*7:14)}
 else{const w=W*%SIZE%/100;m.style.width=w+'px';m.style.height=Math.max(2,w*k)+'px'}},90);
</script></body>"""

def avatar_html(audio=None, photo=None, mx=50, my=72, size=14):
    if photo:
        face = (f'<img src="data:image/jpeg;base64,{photo}" style="width:100%;display:block">'
                f'<div id="m" style="position:absolute;left:{mx}%;top:{my}%;background:#3a0f18;'
                f'border-radius:50%;transform:translate(-50%,-50%);opacity:.9"></div>')
    else:
        face = CARTOON
    au = (f'<audio id="a" controls autoplay style="width:100%;margin-top:8px" '
          f'src="data:audio/mpeg;base64,{audio}"></audio>') if audio else ""
    return TEMPLATE.replace("%FACE%", face).replace("%AUDIO%", au).replace("%SIZE%", str(size))

async def _tts(text, voice):
    out = b""
    async for c in edge_tts.Communicate(text, voice).stream():
        if c["type"] == "audio":
            out += c["data"]
    return out

@st.cache_resource
def yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

def detect(img):
    try:
        r = yolo().predict(img, conf=0.4, imgsz=320, verbose=False)[0]
        names = [r.names[int(c)] for c in r.boxes.cls.tolist()]
        return ", ".join(f"{v} {k}" for k, v in Counter(names).items()), Image.fromarray(r.plot()[..., ::-1])
    except Exception:
        return None, img

def ask_vlm(client, img, q, system, objs):
    parts = []
    if img is not None:
        buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=80)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    parts.append(f"YOLO detector sees: {objs or 'n/a'}\nViewer says: {q}")
    for m in MODELS:
        try:
            r = client.models.generate_content(model=m, contents=parts,
                config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=300))
            if r.text: return r.text.strip()
        except Exception:
            time.sleep(1)
    return "Sorry, the AI service is busy right now. Please try again."

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    persona = st.selectbox("Persona", list(PERSONAS))
    lang = st.selectbox("Language", list(LANGS))
    use_yolo = st.checkbox("YOLO detection (hybrid)", value=False)
    photo_file = st.file_uploader("Use my photo as avatar", type=["jpg", "jpeg", "png"])
    mx = st.slider("Mouth X %", 20, 80, 50); my = st.slider("Mouth Y %", 40, 90, 72)
    size = st.slider("Mouth size", 6, 26, 14)
try:
    key = st.secrets["GEMINI_API_KEY"]
except Exception:
    key = os.getenv("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API key", type="password")

photo = None
if photo_file:
    im = Image.open(photo_file).convert("RGB"); im.thumbnail((600, 800))
    b = io.BytesIO(); im.save(b, "JPEG", quality=85); photo = base64.b64encode(b.getvalue()).decode()

st.title("🎥 VisionTalk Live · AI Avatar + Vision-Language Model")
if "hist" not in st.session_state: st.session_state.hist = []
if "audio" not in st.session_state: st.session_state.audio = None
col_av, col_cam = st.columns(2)           # avatar LEFT, camera RIGHT
with col_cam:
    shot = st.camera_input("Camera")
q = st.chat_input("Ask about what the camera sees...")

if q:
    if not key:
        st.warning("Add your Gemini API key in the sidebar."); st.stop()
    img = Image.open(shot) if shot else None
    objs = None
    if img is not None and use_yolo:
        objs, annotated = detect(img)
        with col_cam: st.image(annotated, caption=f"YOLO: {objs or 'nothing detected'}")
    lname, voice = LANGS[lang]
    system = f"{PERSONAS[persona]} {BASE} Reply in {lname}."
    with st.spinner("Thinking..."):
        reply = ask_vlm(genai.Client(api_key=key), img, q, system, objs)
        try:
            st.session_state.audio = base64.b64encode(asyncio.run(_tts(reply, voice))).decode()
        except Exception:
            st.session_state.audio = None
    st.session_state.hist += [("user", q), ("assistant", reply)]

with col_av:
    components.html(avatar_html(st.session_state.audio, photo, mx, my, size), height=620)
for role, text in st.session_state.hist[-8:]:
    with st.chat_message(role): st.write(text)