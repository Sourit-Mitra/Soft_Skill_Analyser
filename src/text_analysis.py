import re
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class TextAnalyzer:
    def __init__(self):
        # Initialized with a simple filler word set
        self.filler_words = {"um", "uh", "uhh", "umm", "like", "actually", "basically"}
        
    def analyze(self, text):
        # Convert to lower and find all words using regex
        # This handles tokens similarly to basic NLP without the heavy spacy/pydantic dependencies
        words = re.findall(r"\b\w+\b", text.lower())
        total_words = len(words)
        unique_words = len(set(words))
        
        # Simple filler word counting
        filler_count = sum(1 for word in words if word in self.filler_words)
        
        # Add common multi-word filler phrases
        text_lower = text.lower()
        filler_count += text_lower.count("you know")
        filler_count += text_lower.count("i mean")
        
        vocab_richness = (unique_words / total_words) if total_words > 0 else 0
        
        return {
            "total_words": total_words,
            "unique_words": unique_words,
            "filler_count": filler_count,
            "vocab_richness": vocab_richness
        }

    def semantic_analysis(self, text):
        """Uses LLM API call for deep semantic structure and style analysis."""
        # List of free and fast Cerebras LLM models to try sequentially (fallback mechanism)
        providers = [
            {
                "name": "Cerebras Llama-3.1-8B",
                "api_key": os.getenv("CEREBUS_API_KEY") or os.getenv("CEREBRAS_API_KEY"),
                "base_url": "https://api.cerebras.ai/v1",
                "model": "llama3.1-8b"
            },
            {
                "name": "Cerebras Llama-3.3-70B",
                "api_key": os.getenv("CEREBUS_API_KEY") or os.getenv("CEREBRAS_API_KEY"),
                "base_url": "https://api.cerebras.ai/v1",
                "model": "llama-3.3-70b"
            }
        ]
            
        prompt_template = PromptTemplate(
            input_variables=["transcript"],
            template="Analyze the following transcript for communication structure, clarity, and professionalism. Provide 3 short, actionable bullet points of feedback.\n\nTranscript: {transcript}\n\nFeedback:"
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
            
        return "No API keys provided for Cerebras (CEREBUS_API_KEY). Skipping deep semantic analysis."
