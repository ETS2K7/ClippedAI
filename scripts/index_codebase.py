import sys
import json
import chromadb
import cocoindex
from cocoindex.sources import LocalFile
from cocoindex.functions import SplitRecursively, SentenceTransformerEmbed, DetectProgrammingLanguage
import cocoindex.targets.chromadb as chromadb_target

@cocoindex.transform_flow()
def code_to_embedding(text: cocoindex.DataSlice[str]) -> cocoindex.DataSlice[list[float]]:
    """Native central pipeline embedding transformation"""
    return text.transform(
        SentenceTransformerEmbed(model="sentence-transformers/all-MiniLM-L6-v2")
    )

@cocoindex.flow_def(name="CodebaseSemanticIndex")
def index_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    data_scope["files"] = flow_builder.add_source(LocalFile(
        path=".",
        included_patterns=["*.py", "*.ts", "*.tsx", "*.md", "*.prisma"],
        excluded_patterns=["**/node_modules", "**/.*", "scripts/cocoindex_env", "**.db", "dist", "out", "output", ".next"]
    ))
    
    collector = data_scope.add_collector("vector_export")
    
    with data_scope["files"].row() as file:
        # Dynamically classify programming languages explicitly to enable AST mapping
        file["language"] = file["filename"].transform(DetectProgrammingLanguage())
        
        # Split Recursively now acts as a Tree-Sitter interpreter based on dynamic language
        file["chunks"] = file["content"].transform(
            SplitRecursively(), 
            language=file["language"],
            chunk_size=1000,
            min_chunk_size=300,
            chunk_overlap=300
        )
        
        with file["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].call(code_to_embedding)
            
            # Persisting Location and Boundary metadata along with semantic texts
            collector.collect(
                filepath=file["filename"], 
                content=chunk["text"], 
                embedding=chunk["embedding"],
                start=chunk["start"],
                end=chunk["end"]
            )
            
    collector.export(
        "chromadb_export",
        chromadb_target.ChromaDB(
            collection_name="clippedai_context",
            path="./.cocoindex_db"
        ),
        primary_key_fields=["content"]
    )

TOP_K = 5

@index_flow.query_handler(
    result_fields=cocoindex.QueryHandlerResultFields(
        embedding=["embedding"]
    )
)
def search(query: str, k: int = TOP_K) -> cocoindex.QueryOutput:
    # 1. Translate string query precisely mapping the primary ingestion embedding model
    query_vector = code_to_embedding.eval(query)
    
    # 2. Extract context from Chroma nodes natively
    client = chromadb.PersistentClient(path="./.cocoindex_db")
    try:
        collection = client.get_collection("clippedai_context")
    except Exception:
        print("[]")
        return cocoindex.QueryOutput(
            query_info=cocoindex.QueryInfo(
                embedding=query_vector,
                similarity_metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            ),
            results=[]
        )
    
    # 3. Retrieve similarities via direct vector bindings matching
    results = collection.query(
        query_embeddings=[query_vector],
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
    
    return cocoindex.QueryOutput(
        query_info=cocoindex.QueryInfo(
            embedding=query_vector,
            similarity_metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
        ),
        results=output
    )

if __name__ == "__main__":
    cocoindex.init()
    if len(sys.argv) > 1:
        # Interpret exactly as search CLI query immediately
        search_query = sys.argv[1]
        search(search_query)
        sys.exit(0)
    else:
        print("This file statically defines the primary CocoIndex flow.")
        print("1. Update index natively: COCOINDEX_DATABASE_URL=\"sqlite:////Users/ebelthomasseiko/ClippedAI/.cocoindex_metadata.db\" scripts/cocoindex_env/bin/cocoindex update scripts/index_codebase.py")
        print("2. Search directly: scripts/cocoindex_env/bin/python scripts/index_codebase.py 'query string'")
