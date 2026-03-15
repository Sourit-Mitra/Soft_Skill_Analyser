# 🎙️ AI-Based Soft Skills Analyzer

A professional audio analysis tool that evaluates vocal patterns, communication style, and semantic content using AI. It provides an executive summary, detailed scoring across key metrics, and actionable feedback.

## 🚀 Features

- **Audio Transcription**: Powered by OpenAI's Whisper ().
- **Vocal Analysis**: Extracts pitch variation, pause counts, and active speech duration using Librosa.
- **Semantic Feedback**: Deep analysis of communication style using Cerebras LLM (Llama 3.1 8B/70B).
- **Interactive Dashboard**: Built with Streamlit for a premium user experience.
- **Rich Visualizations**: Radar charts and metric cards for clear performance tracking.

## 🛠️ Setup Instructions

### 1. Requirements
- Python 3.9+
- FFmpeg (Handled automatically by the app via `imageio-ffmpeg` if not found in system PATH)

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone <your-repo-url>
cd soft-skills-analyzer
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory (use `.env.example` as a template):
```env
CEREBUS_API_KEY=your_cerebras_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 📊 Evaluation Metrics

- **Clarity & Articulation**: Analysis of speech segments and transcription quality.
- **Engagement (Pitch)**: Evaluation of vocal health and dynamic range.
- **Pacing & Fluency**: Detection of excessive pauses and word per minute counts.
- **Confidence**: Overall score based on vocal stability and semantic tone.

## 🛡️ Privacy
The application processes audio locally for transcription and analysis. Only the transcript text is sent to the LLM (Cerebras) for high-level semantic feedback.

---
Built with ❤️ using Streamlit, Whisper, and Cerebras.
