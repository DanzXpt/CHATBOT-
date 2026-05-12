import streamlit as st
import google.generativeai as genai
from PIL import Image
import urllib.parse
import requests
import io
from pypdf import PdfReader
from docx import Document
from streamlit_mic_recorder import mic_recorder
from duckduckgo_search import DDGS
import base64
import json
import edge_tts
import asyncio
import re
from streamlit_paste_button import paste_image_button
from io import BytesIO
from dotenv import load_dotenv
import os

# =========================
# LOAD API KEY
# =========================
load_dotenv()

api_key = None

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

if not api_key:
    st.error("GEMINI_API_KEY tidak ditemukan. Cek file .env atau Streamlit Secrets.")
    st.stop()

# =========================
# SETUP GEMINI
# =========================
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-3.1-flash-lite")


# =========================
# FUNCTIONS
# =========================
def search_web(query, max_results=3):
    if query is None:
        return ""

    query = str(query)

    if not query.strip():
        return ""

    results_text = ""

    try:
        with DDGS() as ddgs:
            search_query = f'"{query}" spesifikasi harga'
            results = list(ddgs.text(keywords=search_query, max_results=max_results))

            if not results:
                return ""

            for i, result in enumerate(results, start=1):
                title = result.get("title", "")
                body = result.get("body", "")
                link = result.get("href", "")

                results_text += f"""
[{i}] {title}

{body}

Link: {link}

"""

    except Exception as e:
        results_text = f"Search error: {e}"

    return results_text


def get_system_prompt(mode):
    if mode == "Coding":
        return """
Kamu adalah senior software engineer expert.
Fokus utama coding dan teknologi.
Selalu gunakan markdown code block untuk kode.
Berikan solusi teknis langsung.
"""
    elif mode == "Matematika":
        return """
Kamu adalah tutor matematika expert.
Jelaskan langkah demi langkah.
Gunakan rumus jika perlu.
Gunakan analogi sederhana.
"""
    elif mode == "Translator":
        return """
Kamu adalah translator profesional.
Fokus menerjemahkan teks secara natural.
Jangan memberi penjelasan tambahan kecuali diminta.
"""
    elif mode == "Ringkas":
        return """
Kamu adalah AI ultra singkat.
Jawab maksimal 3 kalimat.
Langsung ke inti.
"""
    elif mode == "Tutor":
        return """
Kamu adalah tutor ramah untuk pemula.
Jelaskan pelan-pelan.
Gunakan analogi sederhana.
Hindari jargon teknis yang tidak perlu.
"""
    else:
        return """
Kamu adalah AI assistant general-purpose.
Kamu bisa membantu coding, matematika, penjelasan konsep, translate, brainstorming, dan pertanyaan umum.
Jawab dengan bahasa Indonesia yang jelas dan rapi.
"""


def get_temperature(mode, user_temperature):
    if mode == "Coding":
        return 0.2
    elif mode == "Matematika":
        return 0.3
    elif mode == "Translator":
        return 0.2
    elif mode == "Ringkas":
        return 0.1
    elif mode == "Tutor":
        return 0.6
    else:
        return user_temperature


def clean_username(username):
    username = username.lower().strip()
    username = re.sub(r"[^a-z0-9_]", "_", username)
    return username


def load_memory(username):
    memory_file = f"memory_{username}.json"

    if os.path.exists(memory_file):
        with open(memory_file, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def save_memory(username, memory_data):
    memory_file = f"memory_{username}.json"

    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, ensure_ascii=False, indent=2)


def load_chat_history(username):
    history_file = f"chat_{username}.json"

    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)

    return []


def save_chat_history(username, messages):
    history_file = f"chat_{username}.json"

    safe_messages = [msg for msg in messages if msg.get("type") == "text"]

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(safe_messages, f, ensure_ascii=False, indent=2)


def build_conversation(max_history):
    conversation = ""

    text_messages = [
        msg for msg in st.session_state.messages if msg.get("type") == "text"
    ]

    for msg in text_messages[-max_history:]:
        conversation += f"{msg['role']}: {msg['content']}\n"

    return conversation


def generate_image(prompt_gambar):
    encoded_prompt = urllib.parse.quote(prompt_gambar)

    image_url = (
        f"https://image.pollinations.ai/prompt/"
        f"{encoded_prompt}?width=512&height=512&nologo=true"
    )

    img_response = requests.get(image_url, timeout=60)

    if img_response.status_code != 200:
        raise Exception("Gagal membuat gambar dari server.")

    content_type = img_response.headers.get("Content-Type", "")

    if "image" not in content_type:
        raise Exception("Response server bukan gambar. Coba prompt lain.")

    return img_response.content


def make_chat_text():
    chat_text = ""

    for msg in st.session_state.messages:
        if msg.get("type") == "text":
            chat_text += f"{msg['role'].upper()}: {msg['content']}\n\n"

    return chat_text


def build_translate_prompt(target_lang, text):
    language_map = {
        "id": "Bahasa Indonesia",
        "en": "English",
        "jp": "Japanese",
        "kr": "Korean",
        "cn": "Chinese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ar": "Arabic",
    }

    target_language = language_map.get(target_lang.lower(), target_lang)

    return f"""
Terjemahkan teks berikut ke {target_language}.

Jangan beri penjelasan tambahan.
Hanya tampilkan hasil terjemahan.

Teks:
{text}
"""


def read_uploaded_file(uploaded_file):
    file_text = ""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)

        file_text += f"\n\n--- Isi dari {uploaded_file.name} ---\n"

        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                file_text += text[:3000] + "\n"

    elif file_name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")

        file_text += f"\n\n--- Isi dari {uploaded_file.name} ---\n"
        file_text += text[:3000] + "\n"

    elif file_name.endswith(".docx"):
        doc = Document(uploaded_file)

        file_text += f"\n\n--- Isi dari {uploaded_file.name} ---\n"

        for para in doc.paragraphs:
            if para.text.strip():
                file_text += para.text[:3000] + "\n"

    return file_text


def process_image_upload(image_file, caption="Gambar diupload"):
    image = Image.open(image_file)

    st.image(image, caption=caption, use_container_width=True)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    image_bytes = img_byte_arr.getvalue()

    st.session_state.uploaded_image = {
        "mime_type": "image/png",
        "data": image_bytes,
    }


# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "internet_mode" not in st.session_state:
    st.session_state.internet_mode = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "file_context" not in st.session_state:
    st.session_state.file_context = ""

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    mode = st.selectbox(
        "Mode AI",
        ["General", "Coding", "Matematika", "Translator", "Ringkas", "Tutor"],
    )

    st.success(f"Mode aktif: {mode}")

    internet_mode = st.checkbox(
        "🌐 Internet Search",
        value=st.session_state.internet_mode,
    )

    st.session_state.internet_mode = internet_mode

    voice_output = st.checkbox("🔊 Voice Output")

    temperature = st.slider(
        "Kreativitas Jawaban",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
    )

    max_history = st.slider(
        "Batas Riwayat Chat",
        min_value=5,
        max_value=30,
        value=10,
    )

    final_temperature = get_temperature(mode, temperature)

    st.caption(f"Temperature aktif: {final_temperature}")

    st.divider()
    st.write("Command:")
    st.code(
        """
/gambar motor merah futuristik
/tr en halo apa kabar
/tr jp saya suka anime
/edit ubah background jadi cyberpunk
"""
    )
    st.divider()
    st.caption("Tips: upload gambar lalu tanya: 'jelaskan gambar ini'.")

generation_config = genai.types.GenerationConfig(temperature=final_temperature)

# =========================
# TITLE + LOGIN
# =========================
st.title("🤖 AI Chatbot")
st.write("Chatbot AI pakai Python + Gemini + Streamlit. by Ahdan Hype")
st.caption(
    "Bisa chat, analisis gambar, generate gambar, upload banyak file, voice, dan memory."
)

st.subheader("👤 Login User")

username_input = st.text_input("Masukkan username", placeholder="contoh: danz")

if not username_input:
    st.warning("Masukkan username dulu untuk mulai chat.")
    st.stop()

username = clean_username(username_input)
st.success(f"Login sebagai: {username}")

# =========================
# LOAD MEMORY + HISTORY PER USER
# =========================
memory_data = load_memory(username)

if st.session_state.current_user != username:
    st.session_state.current_user = username
    st.session_state.messages = load_chat_history(username)

# =========================
# ACTION BUTTONS
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("Hapus Chat"):
        st.session_state.messages = []
        st.session_state.uploaded_image = None
        st.session_state.file_context = ""

        history_file = f"chat_{username}.json"

        if os.path.exists(history_file):
            os.remove(history_file)

        st.rerun()

with col2:
    chat_text = make_chat_text()

    st.download_button(
        label="Download Chat",
        data=chat_text if chat_text else "Belum ada chat.",
        file_name=f"riwayat_chat_{username}.txt",
        mime="text/plain",
    )

# =========================
# TAMPILKAN CHAT LAMA
# =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "image_bytes":
            st.image(
                BytesIO(message["content"]),
                caption=message.get("caption", "Generated Image"),
                use_container_width=True,
            )
        else:
            st.markdown(message["content"])

# =========================
# VOICE INPUT
# =========================
st.subheader("🎤 Voice AI")

audio = mic_recorder(
    start_prompt="🎙️ Mulai Rekam",
    stop_prompt="⏹️ Stop Rekam",
    just_once=True,
    key="recorder",
)

voice_text = ""

if audio:
    try:
        with st.spinner("Mengubah suara jadi teks..."):
            audio_bytes = audio["bytes"]

            audio_file = {"mime_type": "audio/wav", "data": audio_bytes}

            response = model.generate_content(
                [
                    """
Transkrip audio ini ke teks bahasa Indonesia.
Hanya tuliskan hasil transkrip tanpa penjelasan tambahan.
""",
                    audio_file,
                ]
            )

            voice_text = response.text

            st.success("Voice berhasil diubah ke teks.")
            st.write("Hasil suara:", voice_text)

    except Exception as e:
        st.error(f"Gagal memproses voice: {e}")

# =========================
# CHAT TOOLS DEKAT INPUT
# =========================
st.divider()
st.caption("📎 Upload / paste file atau gambar di sini, jadi tidak perlu scroll ke atas. Akhirnya UX-nya berhenti menyiksa.")

tool_col1, tool_col2 = st.columns([1, 1])

with tool_col1:
    uploaded_bottom_file = st.file_uploader(
        "📎 Upload",
        type=["jpg", "jpeg", "png", "pdf", "txt", "docx"],
        label_visibility="collapsed",
        key="bottom_uploader",
    )

with tool_col2:
    try:
        paste_result = paste_image_button(label="📋 Paste Gambar", key="paste_img")

        if paste_result.image_data is not None:
            image = paste_result.image_data

            st.image(image, caption="Gambar hasil paste", use_container_width=True)

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            image_bytes = img_byte_arr.getvalue()

            st.session_state.uploaded_image = {
                "mime_type": "image/png",
                "data": image_bytes,
            }

            st.success("Gambar dari clipboard berhasil ditempel.")

    except Exception:
        st.caption("Paste image tidak tersedia di environment ini.")

if uploaded_bottom_file is not None:
    try:
        file_name = uploaded_bottom_file.name.lower()

        if file_name.endswith((".jpg", ".jpeg", ".png")):
            process_image_upload(uploaded_bottom_file, caption="Gambar diupload dari bawah")
            st.success("Gambar berhasil diupload.")

        elif file_name.endswith((".pdf", ".txt", ".docx")):
            new_text = read_uploaded_file(uploaded_bottom_file)

            if new_text.strip():
                st.session_state.file_context = f"\nISI FILE YANG DIUPLOAD:\n{new_text}\n"
                st.success(f"{uploaded_bottom_file.name} berhasil dibaca.")
            else:
                st.warning("File terbaca, tapi teksnya kosong.")

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

file_context = st.session_state.file_context

# =========================
# INPUT USER
# =========================
user_input = st.chat_input("Tulis pesan...", key="main_chat_input")

final_input = ""

if user_input:
    final_input = user_input

if voice_text:
    final_input = voice_text

if final_input:
    bot_reply = None

    st.session_state.messages.append(
        {"role": "user", "content": final_input, "type": "text"}
    )
    save_chat_history(username, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(final_input)

    # =========================
    # MODE TRANSLATE
    # =========================
    if final_input.lower().startswith("/tr"):
        parts = final_input.split(" ", 2)

        with st.chat_message("assistant"):
            if len(parts) < 3:
                bot_reply = (
                    "Format salah.\n\n"
                    "Contoh:\n"
                    "/tr en halo apa kabar\n"
                    "/tr jp saya suka anime"
                )

                st.markdown(bot_reply)

            else:
                target_lang = parts[1]
                translate_text = parts[2]

                try:
                    translate_prompt = build_translate_prompt(
                        target_lang,
                        translate_text,
                    )

                    response = model.generate_content(
                        translate_prompt,
                        generation_config=generation_config,
                    )

                    bot_reply = response.text

                    st.markdown(bot_reply)

                except Exception as e:
                    bot_reply = f"Gagal translate: {e}"
                    st.markdown(bot_reply)

    # =========================
    # MODE GENERATE GAMBAR
    # =========================
    elif final_input.lower().startswith("/gambar"):
        prompt_gambar = final_input.replace("/gambar", "", 1).strip()

        with st.chat_message("assistant"):
            if not prompt_gambar:
                bot_reply = (
                    "Tulis prompt gambar. Contoh: `/gambar motor merah futuristik`"
                )
                st.markdown(bot_reply)

            else:
                try:
                    image_bytes = generate_image(prompt_gambar)

                    st.image(
                        BytesIO(image_bytes),
                        caption=prompt_gambar,
                        use_container_width=True,
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": image_bytes,
                            "caption": prompt_gambar,
                            "type": "image_bytes",
                        }
                    )

                except Exception as e:
                    bot_reply = f"Terjadi error saat membuat gambar: {e}"
                    st.markdown(bot_reply)

    # =========================
    # MODE CHAT AI
    # =========================
    else:
        conversation = build_conversation(max_history)
        system_prompt = get_system_prompt(mode)

        web_context = ""

        if st.session_state.internet_mode and final_input:
            with st.spinner("Mencari informasi di internet..."):
                web_context = search_web(final_input)

        with st.chat_message("assistant"):
            with st.spinner("Bot sedang berpikir..."):
                try:
                    prompt = f"""
{system_prompt}

Gunakan HASIL PENCARIAN INTERNET jika tersedia.

Jika HASIL PENCARIAN INTERNET kosong, umum, tidak spesifik, atau tidak relevan:
- jangan membuat fakta
- jangan menebak
- jangan menyimpulkan produk palsu atau tidak ada
- katakan bahwa hasil pencarian belum cukup spesifik atau belum terverifikasi

Jika hasil pencarian hanya menampilkan halaman umum seperti homepage brand, toko umum, atau kategori produk:
- jangan menyimpulkan produk tidak ada
- jangan membuat klaim pasti
- katakan bahwa hasil pencarian belum menemukan halaman spesifik produk tersebut

Jika informasi internet tidak jelas, tidak lengkap, atau saling bertentangan:
- jangan membuat klaim pasti
- jangan mengarang fakta
- katakan bahwa data belum jelas atau belum terverifikasi

Untuk info produk terbaru seperti HP, laptop, gadget, atau teknologi:
- prioritaskan informasi terbaru
- prioritaskan website resmi brand
- jangan bilang produk tidak ada jika data masih terbatas
- gunakan bahasa hati-hati seperti "belum terverifikasi", "hasil pencarian belum cukup spesifik", atau "informasi masih terbatas"

Dilarang menyatakan produk fiktif, palsu, atau tidak pernah dirilis hanya karena hasil pencarian tidak menemukannya.

Jangan bilang kamu tidak punya akses internet jika hasil search sudah diberikan.

MODE AKTIF:
{mode}

MEMORY USER:
{memory_data}

Riwayat percakapan:
{conversation}

HASIL PENCARIAN INTERNET:
{web_context}

{file_context}

Pertanyaan user:
{final_input}
"""

                    if st.session_state.uploaded_image is not None:
                        response = model.generate_content(
                            [prompt, st.session_state.uploaded_image],
                            generation_config=generation_config,
                        )
                    else:
                        response = model.generate_content(
                            prompt,
                            generation_config=generation_config,
                        )

                    bot_reply = response.text

                except Exception as e:
                    bot_reply = f"Terjadi error: {e}"

                if "nama saya" in final_input.lower():
                    try:
                        name = final_input.lower().replace("nama saya", "").strip()

                        if name:
                            memory_data["nama"] = name
                            save_memory(username, memory_data)

                    except Exception:
                        pass

                st.markdown(bot_reply)

    # =========================
    # VOICE OUTPUT
    # =========================
    if voice_output and bot_reply:
        try:
            async def generate_voice():
                communicate = edge_tts.Communicate(
                    bot_reply,
                    voice="id-ID-ArdiNeural",
                )
                await communicate.save("response.mp3")

            asyncio.run(generate_voice())

            with open("response.mp3", "rb") as audio_file:
                audio_bytes = audio_file.read()

            audio_base64 = base64.b64encode(audio_bytes).decode()

            audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """

            st.markdown(audio_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Gagal membuat voice output: {e}")

    # =========================
    # SAVE CHAT HISTORY
    # =========================
    if bot_reply:
        st.session_state.messages.append(
            {"role": "assistant", "content": bot_reply, "type": "text"}
        )
        save_chat_history(username, st.session_state.messages)