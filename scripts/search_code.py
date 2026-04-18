import sys
import json
import chromadb

def search_code(query: str, k: int = 5):
    try:
        client = chromadb.PersistentClient(path="./.cocoindex_db")
        collection = client.get_collection("clippedai_context")
        
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        
        output = []
        if results and "documents" in results and results["documents"]:
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                metadata = results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] else {}
                output.append({
                    "score": results["distances"][0][i] if "distances" in results and results["distances"] else None,
                    "filepath": metadata.get("filepath", "Unknown"),
                    "content": doc
                })
        
        print(json.dumps(output, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_code.py '<query>' [k]")
        sys.exit(1)
        
    query_text = sys.argv[1]
    k_val = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    search_code(query_text, k_val)
