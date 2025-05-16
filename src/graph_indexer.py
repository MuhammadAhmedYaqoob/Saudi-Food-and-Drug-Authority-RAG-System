import networkx as nx
import spacy
import pathlib
import pickle
import json
import re
from tqdm import tqdm
import openai
from .config import OPENAI_API_KEY, GRAPH_DIR, LLM_MODEL, SUPPORTED_LANGUAGES
from .loader import load_multiple_pdfs, is_arabic

# Initialize OpenAI client with API key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Load SpaCy models for English
NLP_EN = spacy.load("en_core_web_sm")

# Try to load Arabic model if available, otherwise use English
try:
    NLP_AR = spacy.load("xx_ent_wiki_sm")  # MultiLanguage model with Arabic support
    ARABIC_NLP_AVAILABLE = True
except:
    print("[WARNING] Arabic SpaCy model not available. Using English model for all text.")
    NLP_AR = NLP_EN
    ARABIC_NLP_AVAILABLE = False

def extract_domain_entities(sample_texts):
    """
    Use OpenAI to dynamically extract food and pharmaceutical entity 
    categories from sample document content in both English and Arabic
    """
    # Default fallback categories
    default_entities = {
        "drug": ["medication", "medicine", "pill", "tablet", "capsule", "injection", "dose", "دواء", "علاج", "حبوب"],
        "condition": ["disease", "syndrome", "disorder", "illness", "condition", "symptom", "مرض", "متلازمة", "اضطراب"],
        "body_part": ["organ", "system", "tissue", "body", "anatomical", "عضو", "جهاز", "نسيج", "جسم"],
        "dosage": ["mg", "mcg", "ml", "units", "dose", "dosage", "ملغ", "جرعة"],
        "side_effect": ["effect", "reaction", "adverse", "toxicity", "أثر جانبي", "تفاعل"],
        "food": ["food", "nutrition", "ingredient", "edible", "غذاء", "تغذية", "مكون"],
        "regulation": ["regulation", "standard", "law", "compliance", "قانون", "معيار", "امتثال"]
    }
    
    try:
        # Prepare English and Arabic samples
        en_samples = "\n\n".join([text for text, lang in sample_texts if lang == "en"])
        ar_samples = "\n\n".join([text for text, lang in sample_texts if lang == "ar"])
        
        # Combined prompt with both languages
        prompt = f"""
        You are tasked with creating a knowledge graph for the Saudi Food and Drug Authority (SFDA).
        
        Analyze the following sample texts from SFDA documents in both English and Arabic and identify important entity categories 
        and related keywords that should be extracted for the knowledge graph.
        
        English sample text:
        {en_samples[:1000] if en_samples else "No English sample available"}
        
        Arabic sample text:
        {ar_samples[:1000] if ar_samples else "No Arabic sample available"}
        
        Create a comprehensive JSON object with entity categories as keys and lists of related keywords in BOTH English and Arabic as values.
        Focus on food and pharmaceutical-specific entities like drugs, foods, conditions, dosages, regulations, etc.
        
        Format your response as a valid JSON object only, with no other text.
        Example format: {{"drug": ["medication", "pill", "دواء", "علاج"], "food": ["nutrition", "غذاء", "تغذية"]}}
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a bilingual (English/Arabic) expert in food and pharmaceutical domains. Extract entity categories and keywords from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        
        # Extract and parse JSON from response
        content = response.choices[0].message.content.strip()
        
        # Try to find JSON in the response
        import re
        json_match = re.search(r'({.*})', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
            
        # Parse the JSON
        entity_categories = json.loads(content)
        
        print(f"[INFO] Dynamically extracted {len(entity_categories)} entity categories")
        
        # Combine with default entities
        for category, keywords in default_entities.items():
            if category not in entity_categories:
                entity_categories[category] = keywords
            else:
                # Add any missing keywords from defaults
                entity_categories[category] = list(set(entity_categories[category] + keywords))
        
        return entity_categories
        
    except Exception as e:
        print(f"[WARNING] Failed to dynamically extract entities using LLM: {e}")
        print("[INFO] Using fallback entity categories")
        return default_entities

def extract_translation_pairs(entity_categories):
    """Extract potential Arabic-English translation pairs from entity categories"""
    translation_pairs = {}
    
    # For each category, try to pair Arabic and English terms
    for category, terms in entity_categories.items():
        arabic_terms = [term for term in terms if is_arabic(term)]
        english_terms = [term for term in terms if not is_arabic(term) and term.isalpha()]
        
        # Simple heuristic: match by position if possible
        for i in range(min(len(arabic_terms), len(english_terms))):
            translation_pairs[arabic_terms[i]] = english_terms[i]
            translation_pairs[english_terms[i]] = arabic_terms[i]
    
    return translation_pairs

def process_text_with_nlp(text, language):
    """Process text with appropriate NLP model based on language"""
    if language == "ar" and ARABIC_NLP_AVAILABLE:
        return NLP_AR(text)
    return NLP_EN(text)

def build_graph() -> pathlib.Path:
    """
    Build a heterogeneous knowledge graph from PDF content:
    - Document nodes (id:chunk_id)
    - Entity nodes (normalized named entities)
    - Cross-language connections
    - Cross-document relationships
    """
    # Ensure directory exists
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create graph
    G = nx.MultiDiGraph(name="sfda_knowledge_graph")
    
    # Load document chunks from all PDFs
    doc_chunks = load_multiple_pdfs()
    if not doc_chunks:
        print("[ERROR] No document chunks found to build graph")
        return None
    
    # Create a sample text for entity extraction (mixture of English and Arabic)
    sample_texts = []
    en_samples = [chunk for chunk in doc_chunks if chunk.get("language") == "en"][:5]
    ar_samples = [chunk for chunk in doc_chunks if chunk.get("language") == "ar"][:5]
    
    for chunk in en_samples + ar_samples:
        sample_texts.append((chunk["text"], chunk["language"]))
    
    # Dynamically extract entity categories using LLM
    entity_categories = extract_domain_entities(sample_texts)
    
    # Extract potential translation pairs to link Arabic and English
    translation_pairs = extract_translation_pairs(entity_categories)
    
    # Log the extracted categories
    print("[INFO] Using the following entity categories:")
    for category, keywords in entity_categories.items():
        en_keywords = [k for k in keywords if not is_arabic(k)][:3]
        ar_keywords = [k for k in keywords if is_arabic(k)][:3]
        print(f"  - {category}: (EN) {', '.join(en_keywords)}..., (AR) {', '.join(ar_keywords)}...")
    
    # Process each document chunk
    for doc in tqdm(doc_chunks, desc="Building Knowledge Graph"):
        # Create document node with source file tracking
        doc_node = f"doc::{doc['id']}"
        G.add_node(
            doc_node, 
            type="document", 
            text=doc["text"], 
            source_file=doc["source_file"],
            page=doc["page"],
            is_table=doc["is_table"],
            section=doc["section"],
            language=doc["language"]
        )
        
        # Process text with appropriate NLP
        nlp_doc = process_text_with_nlp(doc["text"], doc["language"])
        
        # Extract named entities
        entities = {ent.text.strip(): ent.label_ for ent in nlp_doc.ents}
        
        # Add custom entity detection based on entity categories
        for category, keywords in entity_categories.items():
            for keyword in keywords:
                if keyword.lower() in doc["text"].lower():
                    # Find context around keyword
                    pattern = r"(?:[^\.\n]+\b" + re.escape(keyword) + r"\b[^\.\n]+)"
                    matches = re.findall(pattern, doc["text"], re.IGNORECASE)
                    for match in matches:
                        # Extract potential terms around keywords
                        context_doc = process_text_with_nlp(match, doc["language"])
                        for token in context_doc:
                            if token.is_alpha and (token.is_title or is_arabic(token.text)):
                                entities[token.text] = category
        
        # Add entities and their connections to the graph
        doc_entities = set()  # Track entities in this document
        
        for ent_text, ent_label in entities.items():
            if not ent_text or len(ent_text) < 2:  # Skip very short entities
                continue
                
            ent_node = f"ent::{ent_text.lower()}"
            
            # Add entity node if new
            if not G.has_node(ent_node):
                # Determine language of entity
                ent_lang = "ar" if is_arabic(ent_text) else "en"
                G.add_node(ent_node, type="entity", label=ent_label, language=ent_lang)
            
            # Add relationship from document to entity
            G.add_edge(doc_node, ent_node, role="contains")
            doc_entities.add(ent_node)
            
            # Add translation edges if available
            if ent_text.lower() in translation_pairs:
                trans_text = translation_pairs[ent_text.lower()]
                trans_node = f"ent::{trans_text.lower()}"
                
                if not G.has_node(trans_node):
                    trans_lang = "ar" if is_arabic(trans_text) else "en"
                    G.add_node(trans_node, type="entity", label=ent_label, language=trans_lang)
                
                # Add bidirectional translation edges
                G.add_edge(ent_node, trans_node, role="translates_to")
                G.add_edge(trans_node, ent_node, role="translates_to")
        
        # Add co-occurrence edges between entities in this document
        ent_nodes = list(doc_entities)
        for i in range(len(ent_nodes)):
            for j in range(i + 1, len(ent_nodes)):
                G.add_edge(ent_nodes[i], ent_nodes[j], role="cooccur")
                
        # For tables, add special relationships
        if doc["is_table"]:
            lines = doc["text"].split("\n")
            if len(lines) > 1:
                # Assume first line contains headers
                header_text = lines[0]
                header_doc = process_text_with_nlp(header_text, doc["language"])
                
                header_entities = []
                for token in header_doc:
                    if token.is_alpha and len(token.text) > 2:
                        header_entities.append(token.text.lower())
                
                # Connect headers with content
                for header in header_entities:
                    header_node = f"ent::{header}"
                    if header_node in G:
                        for line in lines[1:]:
                            content_doc = process_text_with_nlp(line, doc["language"])
                            for ent in content_doc.ents:
                                content_node = f"ent::{ent.text.lower()}"
                                if content_node in G and content_node != header_node:
                                    G.add_edge(header_node, content_node, role="table_relation")
    
    # Add cross-document relationships using entity co-occurrence
    print("[INFO] Building cross-document relationships...")
    entity_to_docs = {}
    
    # Map entities to documents
    for node, attrs in G.nodes(data=True):
        if attrs.get("type") == "document":
            for _, ent, _ in G.out_edges(node, data=True):
                if ent.startswith("ent::"):
                    if ent not in entity_to_docs:
                        entity_to_docs[ent] = []
                    entity_to_docs[ent].append(node)
    
    # Connect documents that share significant entities
    doc_relationships = 0
    for ent, docs in entity_to_docs.items():
        if len(docs) > 1:  # Entity appears in multiple docs
            for i in range(len(docs)):
                for j in range(i+1, len(docs)):
                    # Add relationship between documents
                    G.add_edge(docs[i], docs[j], role="related_via", entity=ent)
                    doc_relationships += 1
    
    print(f"[INFO] Added {doc_relationships} cross-document relationships")

    # Save the graph
    out_file = GRAPH_DIR / "sfda_knowledge_graph.pkl"
    with out_file.open("wb") as f:
        pickle.dump(G, f)
    
    print(f"[INFO] Knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return out_file