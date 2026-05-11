import streamlit as st
import google.generativeai as genai
from PIL import Image
import urllib.parse
import requests
from io import BytesIO
from dotenv import load_dotenv
import os

# =========================
# LOAD API KEY
# =========================
load_dotenv()

api_key = None

# Untuk Streamlit Cloud
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

# Untuk lokal dari file .env
if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

if not api_key:
    st.error("GEMINI_API_KEY tidak ditemukan. Cek file .env kamu.")
    st.stop()

# =========================
# SETUP GEMINI
# =========================
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-3.1-flash-lite")


# =========================
# FUNCTIONS
# =========================
def get_system_prompt(mode):
    if mode == "Coding":
        return """
Kamu adalah senior software engineer expert.

ATURAN WAJIB:
- Fokus utama coding dan teknologi.
- Selalu gunakan markdown code block untuk kode.
- Berikan solusi teknis langsung.
- Jelaskan error dan cara memperbaikinya.
- Jika user meminta program, langsung buatkan contoh kode.
- Jika user bertanya di luar coding, arahkan ke sudut pandang teknologi.
- Jawaban harus teknis, jelas, dan tidak terlalu panjang.
"""

    elif mode == "Matematika":
        return """
Kamu adalah tutor matematika expert.

ATURAN WAJIB:
- Jelaskan langkah demi langkah.
- Gunakan rumus jika perlu.
- Fokus pada proses perhitungan.
- Gunakan analogi sederhana.
- Jangan membahas topik di luar matematika kecuali perlu.
- Jelaskan seperti guru privat.
"""

    elif mode == "Translator":
        return """
Kamu adalah translator profesional.

ATURAN WAJIB:
- Fokus menerjemahkan teks.
- Jangan memberi penjelasan tambahan kecuali diminta.
- Gunakan bahasa alami dan grammar yang benar.
- Pertahankan makna asli.
- Jika bahasa tujuan tidak jelas, terjemahkan ke bahasa Indonesia.
"""

    elif mode == "Ringkas":
        return """
Kamu adalah AI ultra singkat.

ATURAN WAJIB:
- Jawab maksimal 3 kalimat.
- Langsung ke inti.
- Jangan bertele-tele.
- Jangan menambahkan penjelasan panjang.
"""

    elif mode == "Tutor":
        return """
Kamu adalah tutor ramah untuk pemula.

ATURAN WAJIB:
- Jelaskan pelan-pelan.
- Gunakan analogi sederhana.
- Hindari jargon teknis yang tidak perlu.
- Ajarkan seperti ke anak SMA.
- Fokus membuat user benar-benar paham.
"""

    else:
        return """
Kamu adalah AI assistant general-purpose.

Kamu bisa membantu:
- coding
- matematika
- penjelasan konsep
- translate
- brainstorming
- pertanyaan umum

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


# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    mode = st.selectbox(
        "Mode AI", ["General", "Coding", "Matematika", "Translator", "Ringkas", "Tutor"]
    )

    st.success(f"Mode aktif: {mode}")

    temperature = st.slider(
        "Kreativitas Jawaban", min_value=0.1, max_value=1.0, value=0.7, step=0.1
    )

    max_history = st.slider("Batas Riwayat Chat", min_value=5, max_value=30, value=10)

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
st.caption("Bisa chat, analisis gambar, generate gambar, dan mode khusus.")

# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader(
    "Upload gambar jika ingin ditanyakan ke AI",
    type=["jpg", "jpeg", "png"]
)

# simpan image di session state
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        st.session_state.uploaded_image = image

        st.image(
            image,
            caption="Gambar yang diupload",
            use_container_width=True
        )

    except Exception:
        st.error("File gambar tidak valid.")
        st.session_state.uploaded_image = None

# =========================
# ACTION BUTTONS
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("Hapus Chat"):
        st.session_state.messages = []
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
# INPUT USER
# =========================
user_input = st.chat_input("Tulis pesan...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "type": "text"}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # =========================
    # MODE GENERATE GAMBAR
    # =========================
    if user_input.lower().startswith("/gambar"):
        prompt_gambar = user_input.replace("/gambar", "", 1).strip()

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
    # MODE CHAT / ANALISIS GAMBAR
    # =========================
    else:
        conversation = build_conversation(max_history)
        system_prompt = get_system_prompt(mode)

        with st.chat_message("assistant"):
            with st.spinner("Bot sedang berpikir..."):
                try:
                    if st.session_state.uploaded_image is not None:
                        prompt = f"""
{system_prompt}

Kamu juga bisa memahami gambar yang diupload user.

MODE AKTIF:
{mode}

Riwayat percakapan:
{conversation}

Pertanyaan user:
{user_input}
"""

                        response = model.generate_content(
                            [prompt, st.session_state.uploaded_image], generation_config=generation_config
                        )

                    else:
                        prompt = f"""
{system_prompt}

MODE AKTIF:
{mode}

Riwayat percakapan:
{conversation}

Pertanyaan user:
{user_input}
"""

                        response = model.generate_content(
                            prompt, generation_config=generation_config
                        )

                    bot_reply = response.text

                except Exception as e:
                    error_text = str(e)

                    if "429" in error_text or "quota" in error_text.lower():
                        bot_reply = (
                            "Quota API habis atau kena limit. "
                            "Coba tunggu sebentar atau cek billing/quota Gemini."
                        )
                    elif "API_KEY" in error_text or "api key" in error_text.lower():
                        bot_reply = (
                            "API key bermasalah. Cek file `.env` dan pastikan "
                            "GEMINI_API_KEY benar."
                        )
                    else:
                        bot_reply = f"Terjadi error: {e}"

                st.markdown(bot_reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": bot_reply, "type": "text"}
        )
