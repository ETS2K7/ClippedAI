import cocoindex
from cocoindex.sources import LocalFile
from cocoindex.functions import SplitRecursively, SentenceTransformerEmbed
import cocoindex.targets.chromadb as chromadb_target

@cocoindex.flow_def(name="CodebaseSemanticIndex")
def index_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    data_scope["backend_files"] = flow_builder.add_source(LocalFile(path="backend/src"))
    data_scope["frontend_files"] = flow_builder.add_source(LocalFile(path="frontend/src"))
    
    collector = data_scope.add_collector("vector_export")
    
    with data_scope["backend_files"].row() as file:
        file["chunks"] = file["content"].transform(
            SplitRecursively(), chunk_size=800
        )
        
        with file["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].transform(
                SentenceTransformerEmbed(model="all-MiniLM-L6-v2")
            )
            
            collector.collect(
                filepath=file["filename"], 
                content=chunk["text"], 
                embedding=chunk["embedding"]
            )

    with data_scope["frontend_files"].row() as file:
        file["chunks"] = file["content"].transform(
            SplitRecursively(), chunk_size=800
        )
        
        with file["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].transform(
                SentenceTransformerEmbed(model="all-MiniLM-L6-v2")
            )
            
            collector.collect(
                filepath=file["filename"], 
                content=chunk["text"], 
                embedding=chunk["embedding"]
            )
            
    collector.export(
        "chromadb_export",
        chromadb_target.ChromaDB(
            collection_name="clippedai_context",
            path="./.cocoindex_db"
        ),
        primary_key_fields=["content"]
    )

if __name__ == "__main__":
    print("This file defines a CocoIndex flow.")
    print("To run the indexer, use: scripts/cocoindex_env/bin/cocoindex update scripts/index_codebase.py")
