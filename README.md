# VisionTalk Live – AI Avatar + Vision-Language Model

A real-time, camera-aware AI host. The viewer talks (or types); the app grabs the current
webcam frame, sends it with the question to a Vision-Language Model, and an animated
avatar speaks the answer with lip-sync and blinking. An optional auto-commentary mode
makes it behave like a live streamer reacting to what it sees.

## Architecture
Browser (webcam + Web Speech STT) → FastAPI `/api/chat` (frame + text + history) → VLM
→ reply → Browser (SVG avatar + speech synthesis lip-sync)

## Run
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your API key
uvicorn app:app --reload    # open http://localhost:8000 (use Chrome)
```
## Stream it
Add `http://localhost:8000` as an OBS **Browser Source** to broadcast the avatar on
YouTube/Twitch/TikTok Live.

## Tech
Python · FastAPI · Vision-Language Model API · WebRTC getUserMedia · Web Speech API ·
SVG animation · OBS integration

## Ideas to extend (upsell to clients)
Custom brand avatar (HeyGen/D-ID/Ready Player Me), ElevenLabs voice, RAG over client docs,
object detection (YOLO) + VLM hybrid, multilingual support, Shopify/WhatsApp integration.
