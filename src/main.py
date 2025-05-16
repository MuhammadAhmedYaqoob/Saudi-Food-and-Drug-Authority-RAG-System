import argparse
from .graph_indexer import build_graph
from .retriever import build_faiss, query_semantic
from .generator import answer

def cli():
    p = argparse.ArgumentParser("SFDA-RAG")
    p.add_argument("--index", action="store_true", help="Build graph + FAISS indices")
    p.add_argument("--query", type=str, help="Ask a question")
    p.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    args = p.parse_args()

    if args.index:
        print("🔄 Building knowledge graph...")
        build_graph()
        print("🔄 Building semantic index...")
        build_faiss()
        print("✅ All indices built successfully.")

    if args.query:
        print(f"🔎 Searching: {args.query}")
        hits = query_semantic(args.query, k=args.k)
        
        if not hits:
            print("❌ No relevant information found.")
            return
            
        print(f"📄 Found {len(hits)} relevant documents")
        response = answer(args.query, hits)
        print("\n🇸🇦 Answer:")
        print(response)

if __name__ == "__main__":
    cli()