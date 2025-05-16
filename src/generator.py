import openai
from typing import List, Dict
from .config import OPENAI_API_KEY, LLM_MODEL, MAX_TOKENS, TEMPERATURE
from .loader import is_arabic

# Initialize OpenAI client with API key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# CoRAG-style System Prompt
SYSTEM_PROMPT = """
You are the official AI assistant for the Saudi Food and Drug Authority (SFDA).
You specialize in providing accurate, concise information about foods, medications, 
treatments, and regulations based on the SFDA's official documentation.

IMPORTANT GUIDELINES:
1. Provide direct, concise answers without unnecessary explanations
2. DO NOT include citations or references to source documents in your responses
3. If information appears in both Arabic and English, prefer to answer in the language of the question
4. Balance comprehensiveness with brevity - include all essential facts but be succinct
5. For safety-critical information (medication dosages, food safety, etc.), be especially precise
6. Never guess or make up information - only provide facts found in the SFDA documentation
7. When technical terms appear in both languages, include both versions (Arabic term / English term)

Remember that your responses represent the official position of the Saudi Food and Drug Authority.
Remember that your final responses must be in the English Language (even thought prompt contains Arabic; Process it in Arabic but respond in English).
"""

def answer(query: str, contexts: List[Dict]) -> str:
    """Generate a concise answer using CoRAG-style generation."""
    
    # Detect query language
    query_lang = "ar" if is_arabic(query) else "en"
    
    # Format the context for the prompt, prioritizing matching language
    matching_lang_contexts = [c for c in contexts if c.get("language") == query_lang]
    other_lang_contexts = [c for c in contexts if c.get("language") != query_lang]
    
    # Prioritize contexts in the query language, then add others
    prioritized_contexts = matching_lang_contexts + other_lang_contexts
    
    # Format contexts
    context_str = "\n\n".join(
        f"Content from {c.get('source_file', 'document')}, Page {c.get('page', 'N/A')}:\n{c['text']}" 
        for c in prioritized_contexts
    )
    
    # Generate the answer
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here is information from the SFDA documentation:\n\n{context_str}\n\nQuestion: {query}\n\nPlease provide a direct, concise answer without referencing the sources."}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        
        response = completion.choices[0].message.content.strip()
        
        # Filter out any source references that might have been generated
        response = response.replace("[Source:", "").replace("[ID:", "")
        
        # Remove any markdown citations like [1], [2], etc.
        import re
        response = re.sub(r'\[\d+\]', '', response)
        
        return response
        
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        return "I apologize, but I encountered an error while generating your answer. Please try again."