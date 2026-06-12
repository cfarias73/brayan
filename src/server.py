"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import litert_lm
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import tts

from dotenv import load_dotenv
load_dotenv()

HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
HF_FILENAME = "gemma-4-E2B-it.litertlm"


def resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        return path
    from huggingface_hub import hf_hub_download
    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)...")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


MODEL_PATH = resolve_model_path()
SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and showing you their camera. "
    "You MUST always use the respond_to_user tool to reply to the user. "
    "If the user asks you to open a website, search something on the web, or check a price, "
    "you MUST call the open_web_browser tool and/or search_web_for_prices tool as needed, "
    "and then call respond_to_user. Do not wait for confirmation to open or search. "
    "If you write or suggest any email, letter, message, or document, you MUST "
    "immediately call the write_draft_file tool first to write it to their Desktop (e.g. 'borrador.txt'), "
    "and then call respond_to_user. Do not wait for confirmation to save it."
)

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

engine = None
tts_backend = None
engine_executor = ThreadPoolExecutor(max_workers=1)


def load_models():
    global engine, tts_backend
    print(f"Loading Gemma 4 E2B from {MODEL_PATH}...")
    engine = litert_lm.Engine(
        MODEL_PATH,
        backend=litert_lm.Backend.GPU,
        vision_backend=litert_lm.Backend.GPU,
        audio_backend=litert_lm.Backend.CPU,
    )
    engine.__enter__()
    print("Engine loaded.")

    tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(engine_executor, load_models)
    yield


app = FastAPI(lifespan=lifespan)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


@app.get("/")
async def root():
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return HTMLResponse(
        content=(Path(__file__).parent / "index.html").read_text(),
        headers=headers
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    # Per-connection tool state captured via closure
    tool_result = {}

    def respond_to_user(transcription: str, response: str) -> str:
        """Respond to the user's voice message.

        Args:
            transcription: Exact transcription of what the user said in the audio.
            response: Your conversational response to the user. Keep it to 1-4 short sentences.
        """
        tool_result["transcription"] = transcription
        tool_result["response"] = response
        return "OK"

    def write_draft_file(filename: str, content: str) -> str:
        """Create or write a text file with the drafted email, letter, or document to the user's Desktop.

        Args:
            filename: The name of the file to create (e.g. 'borrador_correo.txt' or 'carta_pago.txt').
            content: The complete text content of the email, letter, or document.
        """
        try:
            desktop_path = Path.home() / "Desktop"
            file_path = desktop_path / filename
            file_path.write_text(content)
            
            # Send notification back to WebSocket client thread-safely
            asyncio.run_coroutine_threadsafe(
                ws.send_text(json.dumps({
                    "type": "file_written",
                    "filename": filename,
                    "filepath": str(file_path),
                    "content": content
                })),
                loop
            )
            return f"Successfully saved draft to {file_path}"
        except Exception as e:
            return f"Error saving file: {str(e)}"

    def open_web_browser(url: str) -> str:
        """Open the web browser to the specified URL.

        Args:
            url: The URL to open (e.g. 'https://www.walmart.com/search?q=milk').
        """
        try:
            import webbrowser
            webbrowser.open(url)
            return f"Successfully opened browser to {url}"
        except Exception as e:
            return f"Error opening browser: {str(e)}"

    def search_web_for_prices(query: str) -> str:
        """Search the web for product prices, comparison details, or other information.

        Args:
            query: The search term or query (e.g. 'walmart milk price' or 'precio de la leche walmart').
        """
        try:
            import urllib.request
            import urllib.parse
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            url = 'https://search.yahoo.com/search?p=' + urllib.parse.quote_plus(query)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
            
            items = html.split('<li')
            results = []
            for item in items:
                if 'algo-sr' not in item:
                    continue
                
                href_match = re.search(r'href="([^"]+)"', item)
                title_match = re.search(r'class="title[^"]*">.*?<span[^>]*>(.*?)</span>', item, re.DOTALL)
                snippet_match = re.search(r'<div class="compText[^"]*">.*?<p[^>]*>(.*?)</p>', item, re.DOTALL)
                
                if href_match and title_match:
                    href = href_match.group(1)
                    title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
                    
                    ru_match = re.search(r'/RU=([^/&]+)', href)
                    if ru_match:
                        decoded_url = urllib.parse.unquote(ru_match.group(1))
                    else:
                        decoded_url = href
                        
                    snippet = ""
                    if snippet_match:
                        snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
                        
                    results.append(f"Title: {title}\nURL: {decoded_url}\nSnippet: {snippet}")
            
            if not results:
                return "No search results found."
            return "\n\n".join(results[:5])
        except Exception as e:
            return f"Error searching: {str(e)}"

    def init_conversation():
        conv = engine.create_conversation(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}],
            tools=[respond_to_user, write_draft_file, open_web_browser, search_web_for_prices],
        )
        conv.__enter__()
        return conv

    conversation = await asyncio.get_event_loop().run_in_executor(engine_executor, init_conversation)

    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

    async def receiver():
        """Receive messages from WebSocket and route them."""
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            interrupted.clear()

            language = msg.get("language", "en")

            content = []
            if msg.get("audio"):
                content.append({"type": "audio", "blob": msg["audio"]})
            if msg.get("image"):
                content.append({"type": "image", "blob": msg["image"]})

            if msg.get("audio") and msg.get("image"):
                if language == "es":
                    content.append({"type": "text", "text": "El usuario te acaba de hablar (audio) mientras te muestra su cámara (imagen). Responde a lo que dijo, haciendo referencia a lo que ves si es relevante. Responde en español."})
                else:
                    content.append({"type": "text", "text": "The user just spoke to you (audio) while showing their camera (image). Respond to what they said, referencing what you see if relevant."})
            elif msg.get("audio"):
                if language == "es":
                    content.append({"type": "text", "text": "El usuario te acaba de hablar. Responde a lo que dijo en español."})
                else:
                    content.append({"type": "text", "text": "The user just spoke to you. Respond to what they said."})
            elif msg.get("image"):
                if language == "es":
                    content.append({"type": "text", "text": "El usuario te está mostrando su cámara. Describe lo que ves en español."})
                else:
                    content.append({"type": "text", "text": "The user is showing you their camera. Describe what you see."})
            else:
                default_text = "¡Hola!" if language == "es" else "Hello!"
                content.append({"type": "text", "text": msg.get("text", default_text)})

            # LLM inference
            t0 = time.time()
            tool_result.clear()
            response = await asyncio.get_event_loop().run_in_executor(
                engine_executor, lambda: conversation.send_message({"role": "user", "content": content})
            )
            llm_time = time.time() - t0

            # Extract response from tool call or fallback to raw text
            if tool_result:
                strip = lambda s: s.replace('<|"|>', "").strip()
                transcription = strip(tool_result.get("transcription", ""))
                text_response = strip(tool_result.get("response", ""))
                print(f"LLM ({llm_time:.2f}s) [tool] heard: {transcription!r} → {text_response}")
            else:
                transcription = None
                text_response = response["content"][0]["text"]
                print(f"LLM ({llm_time:.2f}s) [no tool]: {text_response}")

            if interrupted.is_set():
                print("Interrupted after LLM, skipping response")
                continue

            reply = {"type": "text", "text": text_response, "llm_time": round(llm_time, 2)}
            if transcription:
                reply["transcription"] = transcription
            await ws.send_text(json.dumps(reply))

            if interrupted.is_set():
                print("Interrupted before TTS, skipping audio")
                continue

            # Streaming TTS: split into sentences and send chunks progressively
            sentences = split_sentences(text_response)
            if not sentences:
                sentences = [text_response]

            tts_start = time.time()

            # Signal start of audio stream
            await ws.send_text(json.dumps({
                "type": "audio_start",
                "sample_rate": tts_backend.sample_rate,
                "sentence_count": len(sentences),
            }))

            tts_voice = "ef_dora" if language == "es" else "af_heart"
            tts_lang = "es" if language == "es" else "en"

            for i, sentence in enumerate(sentences):
                if interrupted.is_set():
                    print(f"Interrupted during TTS (sentence {i+1}/{len(sentences)})")
                    break

                # Generate audio for this sentence
                pcm = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=sentence: tts_backend.generate(s, voice=tts_voice, lang_code=tts_lang)
                )

                if interrupted.is_set():
                    break

                # Convert to 16-bit PCM and send as base64
                pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                await ws.send_text(json.dumps({
                    "type": "audio_chunk",
                    "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                    "index": i,
                }))

            tts_time = time.time() - tts_start
            print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

            if not interrupted.is_set():
                await ws.send_text(json.dumps({
                    "type": "audio_end",
                    "tts_time": round(tts_time, 2),
                }))

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()
        try:
            await asyncio.get_event_loop().run_in_executor(
                engine_executor, lambda: conversation.__exit__(None, None, None)
            )
        except Exception as e:
            print(f"Error exiting conversation: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
