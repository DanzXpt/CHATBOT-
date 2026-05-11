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

def search_web(query, max_results=5):
    results_text = ""

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)

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


def build_conversation(max_history):
    conversation = ""

    text_messages = [
        msg for msg in st.session_state.messages
        if msg.get("type") == "text"
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


# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    mode = st.selectbox(
        "Mode AI",
        ["General", "Coding", "Matematika", "Translator", "Ringkas", "Tutor"]
    )

    st.success(f"Mode aktif: {mode}")
    internet_mode = st.toggle("🌐 Internet Search", value=False)

    temperature = st.slider(
        "Kreativitas Jawaban",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    max_history = st.slider(
        "Batas Riwayat Chat",
        min_value=5,
        max_value=30,
        value=10
    )

    final_temperature = get_temperature(mode, temperature)
    st.caption(f"Temperature aktif: {final_temperature}")

    st.divider()
    st.write("Command:")
    st.code("/gambar motor merah futuristik")
    st.divider()
    st.caption("Tips: upload gambar lalu tanya: 'jelaskan gambar ini'.")

generation_config = genai.types.GenerationConfig(temperature=final_temperature)

# =========================
# TITLE
# =========================
st.title("🤖 AI Chatbot")
st.write("Chatbot AI pakai Python + Gemini + Streamlit. by Ahdan Hype")
st.caption("Bisa chat, analisis gambar, generate gambar, dan upload banyak file.")

# =========================
# UPLOAD GAMBAR
# =========================
uploaded_image_file = st.file_uploader(
    "Upload gambar jika ingin ditanyakan ke AI",
    type=["jpg", "jpeg", "png"],
    key="image_uploader"
)

if uploaded_image_file is not None:
    try:
        image = Image.open(uploaded_image_file)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        image_bytes = img_byte_arr.getvalue()

        st.session_state.uploaded_image = {
            "mime_type": "image/png",
            "data": image_bytes,
        }

    except Exception:
        st.error("File gambar tidak valid.")
        st.session_state.uploaded_image = None

# =========================
# MULTI FILE UPLOAD
# =========================
uploaded_files = st.file_uploader(
    "Upload File (PDF, TXT, DOCX)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    key="file_uploader"
)

all_file_text = ""

if uploaded_files:
    for uploaded_doc in uploaded_files:
        try:
            file_name = uploaded_doc.name.lower()

            if file_name.endswith(".pdf"):
                pdf_reader = PdfReader(uploaded_doc)

                all_file_text += f"\n\n--- Isi dari {uploaded_doc.name} ---\n"

                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        all_file_text += text + "\n"

            elif file_name.endswith(".txt"):
                text = uploaded_doc.read().decode("utf-8")

                all_file_text += f"\n\n--- Isi dari {uploaded_doc.name} ---\n"
                all_file_text += text + "\n"

            elif file_name.endswith(".docx"):
                doc = Document(uploaded_doc)

                all_file_text += f"\n\n--- Isi dari {uploaded_doc.name} ---\n"

                for para in doc.paragraphs:
                    if para.text.strip():
                        all_file_text += para.text + "\n"

            st.success(f"{uploaded_doc.name} berhasil dibaca.")

        except Exception as e:
            st.error(f"Gagal membaca {uploaded_doc.name}: {e}")

if all_file_text.strip():
    file_context = f"\nISI FILE YANG DIUPLOAD:\n{all_file_text}\n"
else:
    file_context = ""

# =========================
# ACTION BUTTONS
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("Hapus Chat"):
        st.session_state.messages = []
        st.session_state.uploaded_image = None
        st.rerun()

with col2:
    chat_text = make_chat_text()

    st.download_button(
        label="Download Chat",
        data=chat_text if chat_text else "Belum ada chat.",
        file_name="riwayat_chat.txt",
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
    key="recorder"
)

voice_text = ""

if audio:
    try:
        with st.spinner("Mengubah suara jadi teks..."):

            audio_bytes = audio["bytes"]

            audio_file = {
                "mime_type": "audio/wav",
                "data": audio_bytes
            }

            response = model.generate_content([
                """
Transkrip audio ini ke teks bahasa Indonesia.
Hanya tuliskan hasil transkrip tanpa penjelasan tambahan.
""",
                audio_file
            ])

            voice_text = response.text

            st.success("Voice berhasil diubah ke teks.")
            st.write("Hasil suara:", voice_text)

    except Exception as e:
        st.error(f"Gagal memproses voice: {e}")

# =========================
# INPUT USER
# =========================
user_input = st.chat_input("Tulis pesan...")

final_input = user_input

if voice_text:
    final_input = voice_text

if final_input:
    st.session_state.messages.append(
        {"role": "user", "content": final_input, "type": "text"}
    )

    with st.chat_message("user"):
        st.markdown(final_input)

    # =========================
    # MODE GENERATE GAMBAR
    # =========================
    if final_input.lower().startswith("/gambar"):
        prompt_gambar = final_input.replace("/gambar", "", 1).strip()

        with st.chat_message("assistant"):
            if not prompt_gambar:
                bot_reply = (
                    "Tulis deskripsi gambar setelah `/gambar`.\n\n"
                    "Contoh: `/gambar kucing cyberpunk di kota futuristik`"
                )

                st.markdown(bot_reply)

                st.session_state.messages.append(
                    {"role": "assistant", "content": bot_reply, "type": "text"}
                )

            else:
                with st.spinner("Sedang membuat gambar..."):
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

                        st.session_state.messages.append(
                            {"role": "assistant", "content": bot_reply, "type": "text"}
                        )

    # =========================
    # MODE CHAT / ANALISIS GAMBAR / FILE
    # =========================
else:
    conversation = build_conversation(max_history)
    system_prompt = get_system_prompt(mode)

    with st.chat_message("assistant"):
        with st.spinner("Bot sedang berpikir..."):
            try:

                # =========================
                # INTERNET SEARCH
                # =========================
                web_context = ""

                if internet_mode:
                    with st.spinner("Mencari informasi di internet..."):
                        web_context = search_web(final_input)
                        st.write(web_context)

                # =========================
                # PROMPT
                # =========================
                prompt = f"""
{system_prompt}

MODE AKTIF:
{mode}

Riwayat percakapan:
{conversation}

HASIL PENCARIAN INTERNET:
{web_context}

{file_context}

Pertanyaan user:
{final_input}
"""

                # =========================
                # IMAGE MODE
                # =========================
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

                # =========================
                # RESPONSE
                # =========================
                try:
                    bot_reply = response.text

                except Exception:
                    bot_reply = (
                        "AI gagal membaca response. "
                        "Kemungkinan quota Gemini habis atau model tidak mendukung input ini."
                    )

            except Exception as e:

                error_text = str(e)

                if "429" in error_text or "quota" in error_text.lower():

                    bot_reply = (
                        "Quota API habis atau kena limit. "
                        "Coba tunggu sebentar atau cek billing/quota Gemini."
                    )

                elif "API_KEY" in error_text or "api key" in error_text.lower():

                    bot_reply = (
                        "API key bermasalah. "
                        "Cek file `.env` atau Streamlit Secrets."
                    )

                elif "not found" in error_text.lower():

                    bot_reply = (
                        "Model Gemini tidak ditemukan. "
                        "Pakai model yang tersedia di akun kamu."
                    )

                else:

                    bot_reply = f"Terjadi error: {e}"

            st.markdown(bot_reply)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": bot_reply,
            "type": "text"
        }
    )