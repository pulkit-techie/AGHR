"""
AGHR System — Knowledge Graph Builder (Phase 3.3–3.4) [Neo4j Version]
======================================================
Builds and manages knowledge graph using Neo4j (AuraDB).
Supports multi-hop traversal and semantic graph routing.
"""

from typing import Dict, List, Optional
from loguru import logger
from neo4j import GraphDatabase, exceptions

class KnowledgeGraphBuilder:
    """
    Neo4j-backed Knowledge Graph.
    Connects to AuraDB and executes Cypher queries.
    """

    def __init__(self, uri: str = None, username: str = None, password: str = None):
        self.driver = None
        if uri and username and password:
            self.connect(uri, username, password)

    def connect(self, uri: str, username: str, password: str) -> None:
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j AuraDB: {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def add_triple(self, entity_1: str, relation: str, entity_2: str,
                   e1_type: str = "ENTITY", e2_type: str = "ENTITY",
                   metadata: Dict = None) -> None:
        """Add a single (entity_1)-[relation]->(entity_2) triple to Neo4j."""
        if not self.driver:
            return

        query = f"""
        MERGE (n1:Entity {{normalized: toLower($e1)}})
        ON CREATE SET n1.text = $e1, n1.type = $e1_type, n1.count = 1
        ON MATCH SET n1.count = n1.count + 1

        MERGE (n2:Entity {{normalized: toLower($e2)}})
        ON CREATE SET n2.text = $e2, n2.type = $e2_type, n2.count = 1
        ON MATCH SET n2.count = n2.count + 1

        MERGE (n1)-[r:{relation.upper()}]->(n2)
        ON CREATE SET r.weight = 1
        ON MATCH SET r.weight = r.weight + 1
        """
        
        try:
            with self.driver.session() as session:
                session.run(query, e1=entity_1, e1_type=e1_type,
                            e2=entity_2, e2_type=e2_type)
        except Exception as e:
            logger.warning(f"Neo4j add_triple failed: {e}")

    def build_from_triples(self, triples: List[Dict]) -> None:
        """Batch insert triples."""
        for t in triples:
            self.add_triple(t["entity_1"], t["relation"], t["entity_2"])

    def build_from_relation_results(self, relation_results: List[Dict]) -> None:
        """Build graph from relation extraction results using high-performance batch insertion."""
        if not self.driver:
            return

        all_triples = []
        for result in relation_results:
            for rel in result.get("relations", []):
                all_triples.append({
                    "e1": rel["entity_1"],
                    "e1_type": rel.get("entity_1_type", "ENTITY"),
                    "rel": rel["relation"].upper(),
                    "e2": rel["entity_2"],
                    "e2_type": rel.get("entity_2_type", "ENTITY")
                })

        if not all_triples:
            return

        # Batch insertion query using UNWIND
        # Note: Dynamic relationship types require APOC or multiple passes. 
        # For simplicity and compatibility, we'll group by relationship type and batch.
        
        rel_groups = {}
        for t in all_triples:
            r_type = t["rel"]
            if r_type not in rel_groups:
                rel_groups[r_type] = []
            rel_groups[r_type].append(t)

        try:
            with self.driver.session() as session:
                for r_type, triples in rel_groups.items():
                    query = f"""
                    UNWIND $batch AS row
                    MERGE (n1:Entity {{normalized: toLower(row.e1)}})
                    ON CREATE SET n1.text = row.e1, n1.type = row.e1_type, n1.count = 1
                    ON MATCH SET n1.count = n1.count + 1

                    MERGE (n2:Entity {{normalized: toLower(row.e2)}})
                    ON CREATE SET n2.text = row.e2, n2.type = row.e2_type, n2.count = 1
                    ON MATCH SET n2.count = n2.count + 1

                    MERGE (n1)-[r:{r_type}]->(n2)
                    ON CREATE SET r.weight = 1
                    ON MATCH SET r.weight = r.weight + 1
                    """
                    session.run(query, batch=triples)
            logger.info(f"Pushed {len(all_triples)} relations to Neo4j in batches.")
        except Exception as e:
            logger.warning(f"Neo4j batch insert failed: {e}")

    def get_neighbors(self, entity_text: str, max_hops: int = 1) -> Dict:
        """Get neighbors up to max_hops away from Neo4j."""
        if not self.driver:
            return {"nodes": [], "edges": [], "paths": []}

        # Query all paths up to max_hops
        query = """
        MATCH path = (start:Entity)-[*1..%d]-(end:Entity)
        WHERE start.normalized CONTAINS toLower($entity) OR toLower($entity) CONTAINS start.normalized
        RETURN [node in nodes(path) | node.text] AS node_texts,
               [rel in relationships(path) | type(rel)] AS rel_types
        LIMIT 50
        """ % max_hops

        paths = []
        edges = []
        
        try:
            with self.driver.session() as session:
                result = session.run(query, entity=entity_text)
                for record in result:
                    node_texts = record["node_texts"]
                    rel_types = record["rel_types"]
                    
                    if not node_texts or len(node_texts) < 2:
                        continue
                        
                    # Format string representation as natural language
                    sentences = []
                    for i in range(len(rel_types)):
                        rel_str = rel_types[i].lower().replace('_', ' ')
                        sentences.append(f"{node_texts[i]} {rel_str} {node_texts[i+1]}")
                    
                    paths.append(". ".join(sentences) + ".")
                    
                    # Store edges for Context Fusion
                    for i in range(len(rel_types)):
                        edges.append({
                            "source_text": node_texts[i],
                            "relations": [rel_types[i]],
                            "target_text": node_texts[i+1]
                        })
        except Exception as e:
            logger.warning(f"Neo4j query failed: {e}")

        return {"nodes": [], "edges": edges, "paths": paths}

    def get_subgraph_context(self, entities: List[str], max_hops: int = 2) -> str:
        """Get textual context from Neo4j for the given entities."""
        context_parts = []
        seen = set()

        for entity in entities:
            result = self.get_neighbors(entity, max_hops=max_hops)
            for path in result["paths"]:
                if path not in seen:
                    seen.add(path)
                    context_parts.append(path)

        return "\n".join(context_parts) if context_parts else "No graph context available."

    def get_stats(self) -> Dict:
        """Get node and relationship counts from Neo4j."""
        if not self.driver:
            return {"nodes": 0, "edges": 0}
            
        stats = {"nodes": 0, "edges": 0}
        try:
            with self.driver.session() as session:
                res1 = session.run("MATCH (n) RETURN count(n) as c")
                stats["nodes"] = res1.single()["c"]
                
                res2 = session.run("MATCH ()-[r]->() RETURN count(r) as c")
                stats["edges"] = res2.single()["c"]
        except:
            pass
        return stats

    def save(self, filepath: str) -> None:
        """Export a snapshot of the graph from Neo4j to a local JSON file for UI visualization."""
        if not self.driver:
            logger.warning("Cannot save: No Neo4j driver connected.")
            return
            
        try:
            with self.driver.session() as session:
                # Query all nodes
                nodes_result = session.run("MATCH (n:Entity) RETURN n.text AS id, n.type AS type")
                nodes = [{"id": r["id"], "label": r["id"], "type": r["type"]} for r in nodes_result]
                
                # Query all relationships
                edges_result = session.run("MATCH (n1)-[r]->(n2) RETURN n1.text AS source, type(r) AS label, n2.text AS target")
                edges = [{"source": r["source"], "label": r["label"], "target": r["target"]} for r in edges_result]
                
                import json
                import os
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump({"nodes": nodes, "edges": edges}, f, indent=2)
                logger.info(f"Exported {len(nodes)} nodes and {len(edges)} edges to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export graph snapshot: {e}")

    def load(self, filepath: str) -> None:
        """Load a graph snapshot from a local JSON file (useful as a fallback)."""
        import json
        from pathlib import Path
        
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Graph snapshot file not found: {filepath}")
            return
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # We don't store nodes/edges in memory because Neo4j is the source of truth,
                # but if we're in offline mode, we might want to store them.
                # For now, we'll just log that we loaded it.
                logger.info(f"Loaded graph snapshot with {len(data.get('nodes', []))} nodes and {len(data.get('edges', []))} edges.")
                # If we want offline support, we'd need to implement a local query engine here.
                # But since we have Neo4j, we use Neo4j for querying.
        except Exception as e:
            logger.error(f"Failed to load graph snapshot: {e}")
