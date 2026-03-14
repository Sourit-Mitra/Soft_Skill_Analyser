import re
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
class TextAnalyzer:
    def __init__(self):
        self.filler_words = {
            "um", "uh", "uhh", "umm",
            "like",
            "actually",
            "basically",
            "so",
            "right",
            "well",
            "yeah",
        }
        
        self.filler_phrases = [
            "you know",
            "i mean",
            "kind of",
            "sort of",
            "you know what i mean",
            "i guess",
            "i think",
            "i mean like",
            "you know like",
        ]
        
    def analyze(self, text):
        words = re.findall(r"\b\w+\b", text.lower())
        total_words = len(words)
        unique_words = len(set(words))
        
        filler_count = sum(1 for word in words if word in self.filler_words)
        
        text_lower = text.lower()
        phrase_count = 0
        for phrase in self.filler_phrases:
            occurrences = text_lower.count(phrase)
            phrase_count += occurrences
            
        filler_count += phrase_count
        
        vocab_richness = (unique_words / total_words) if total_words > 0 else 0
        
        return {
            "total_words": total_words,
            "unique_words": unique_words,
            "filler_count": filler_count,
            "vocab_richness": vocab_richness,
            "phrase_fillers": phrase_count
        }

    def semantic_analysis(self, text):
        """Uses LLM API call for deep semantic structure and style analysis."""
        # Try to get API key from Streamlit Secrets FIRST (Cloud), then Environment (Local)
        _api_key = (
            st.secrets.get("CEREBRAS_API_KEY") or  
            os.getenv("CEREBRAS_API_KEY") 
        )

        # List of free and fast Cerebras LLM models to try sequentially (fallback mechanism)
        providers = [
            {
                "name": "Cerebras Llama-3.1-8B",
                "api_key": _api_key,
                "base_url": "https://api.cerebras.ai/v1",
                "model": "llama3.1-8b"
            },
            {
                "name": "Cerebras Llama-3.3-70B",
                "api_key": _api_key,
                "base_url": "https://api.cerebras.ai/v1",
                "model": "llama-3.3-70b"
            }
        ]
            

        prompt_template = PromptTemplate(
            input_variables=["transcript"],
            template="""You are an expert communication coach. Analyze the transcript and respond with EXACTLY 3 bullet points. No introduction, no conclusion, no extra text. Just 3 bullets.

        Transcript:
        {transcript}

        Rules:
        - Start each bullet with •
        - Maximum 20 words per bullet
        - Each bullet must be one actionable improvement

        Feedback:"""
        )
        
        last_error = None
        for provider in providers:
            # Skip if API key is not set or still default placeholder
            if not provider["api_key"] or "your_" in provider["api_key"].lower():
                continue
                
            try:
                llm = ChatOpenAI(
                    temperature=0.3, 
                    model=provider["model"], 
                    api_key=provider["api_key"],
                    base_url=provider["base_url"],
                    max_retries=1
                )
                chain = prompt_template | llm
                response = chain.invoke({"transcript": text})
                return response.content
            except Exception as e:
                last_error = e
                # Print to console for debugging but continue to the next provider
                print(f"[Warning] Failed calling {provider['name']} LLM: {e}")
                continue
        
        if last_error:
            return f"Error: All available LLM models failed. Last error: {last_error}"
            
        return "No API keys provided for Cerebras (CEREBRAS_API_KEY). Skipping deep semantic analysis."


