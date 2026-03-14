import os
import tempfile
import streamlit as st
from src.utils import setup_ffmpeg
from src.scoring_engine import ScoringEngine
from src.visual_components import get_radar_chart

# Initialize FFmpeg setup
setup_ffmpeg()

st.set_page_config(page_title="Soft Skills Analyzer", layout="wide")

@st.cache_resource
def get_scoring_engine():
    return ScoringEngine()

st.title("🎙️ AI-Based Soft Skills Analyzer")
st.markdown("Upload your audio recording to receive professional analysis on your communication style.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Save uploaded file temporarily — preserve original extension so
    # whisper and librosa can correctly detect and decode the format
    file_extension = os.path.splitext(uploaded_file.name)[1]
    temp_audio_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_audio_path = tmp_file.name

        with st.spinner("Analyzing vocal patterns and communication style..."):
            try:
                engine = get_scoring_engine()
                report = engine.generate_report(temp_audio_path)
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"Error processing audio: {e}")
                st.expander("Show internal traceback (Debug Mode)").code(error_details)
                st.stop()
                
            # --- TOP ROW (CARDS) ---
            st.markdown("### Executive Summary")
            cols = st.columns(4)
            
            with cols[0]:
                with st.container(border=True):
                    st.metric("Overall Score", f"{report['scores']['overall']}/100")
            
            with cols[1]:
                with st.container(border=True):
                    st.metric("Pacing", f"{report['raw_metrics']['wpm']} WPM")
                    
            with cols[2]:
                with st.container(border=True):
                    st.metric("Clarity", f"{report['scores']['clarity']}/100")
                    
            with cols[3]:
                with st.container(border=True):
                    st.metric("Engagement", f"{report['scores']['engagement']}/100")

            # --- MIDDLE ROW ---
            st.markdown("### Detailed Analysis")
            col_chart, col_feedback = st.columns([6, 4])
            
            with col_chart:
                st.plotly_chart(get_radar_chart(report['scores']), use_container_width=True)
                
            with col_feedback:
                st.markdown("#### Actionable Feedback")
                # Dynamic pacing feedback
                wpm = report['raw_metrics']['wpm']
                if wpm > 180:
                    st.write("⚠️ **Pacing**: Try slowing down your delivery. Your pacing is quite fast.")
                elif wpm < 120:
                    st.write("⚠️ **Pacing**: You could increase your speech rate for better engagement.")
                else:
                    st.write("✅ **Pacing**: Excellent speaking rate!")
                    
                st.write(f"ℹ️ **Filler Words**: Detected {report['raw_metrics']['filler_count']} instances.")
                st.write(f"ℹ️ **Pauses**: Detected {report['raw_metrics']['pauses']} significant pauses.")
                
                st.markdown("#### Deep Semantic Analysis (LLM)")
                st.write(report['semantic_feedback'])
                
            # --- BOTTOM ROW: TIMESTAMPED TRANSCRIPT ---
            st.markdown("### Annotated Transcript")
            
            def highlight_word(word):
                fillers = ["um", "uh", "like"]
                clean_word = word.lower().strip(".,?!")
                if clean_word in fillers:
                    return f"<mark style='background-color:#fecaca; color:#991b1b;'>{word}</mark>"
                return word

            def format_timestamp(seconds):
                mins = int(seconds // 60)
                secs = int(seconds % 60)
                return f"[{mins:02d}:{secs:02d}]"

            transcript_html = ""
            for segment in report['segments']:
                start_time = format_timestamp(segment['start'])
                transcript_html += f"<b>{start_time}</b> "
                
                if 'words' in segment:
                    for word_data in segment['words']:
                        transcript_html += highlight_word(word_data['word'].strip()) + " "
                else:
                    words = segment['text'].split()
                    highlighted_words = [highlight_word(w) for w in words]
                    transcript_html += " ".join(highlighted_words) + " "
                    
                transcript_html += "<br>"

            st.markdown(transcript_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"General Error: {e}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
