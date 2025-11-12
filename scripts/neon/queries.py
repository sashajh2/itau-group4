"""
Database query functions for Neon PostgreSQL.

This module provides functions to query segments and embeddings from Neon.
"""

from typing import List, Optional
import numpy as np
import psycopg2

from utils.config_loader import load_config


def connect_neon():
    """Connect to Neon Postgres database."""
    cfg = load_config()
    dsn = cfg["database"]["postgres"]["neon_database_url"]
    return psycopg2.connect(dsn)


def get_all_video_ids(
    conn: psycopg2.extensions.connection, 
    created_at_filter: Optional[str] = None,
    source: Optional[str] = None,
    require_all_embeddings: bool = True
) -> List[str]:
    """
    Get all unique video_ids with optional filters.
    
    Args:
        conn: Database connection
        created_at_filter: Optional filter by created_at >= this date
        source: Optional filter by source (e.g., 'ShareVeo3')
        require_all_embeddings: If True, only return video_ids that have segments with
                               all 3 embedding types (openl3, hubert, senet)
    
    Returns:
        List of unique video_ids
    """
    where_clauses = []
    params = []
    
    if require_all_embeddings:
        # Join with all 3 embedding tables to ensure segments have all embeddings
        from_clause = """
            FROM segments s
            INNER JOIN embeddings_audio_openl3 e1 ON s.segment_id = e1.segment_id
            INNER JOIN embeddings_audio_hubert e2 ON s.segment_id = e2.segment_id
            INNER JOIN embeddings_video_senet e3 ON s.segment_id = e3.segment_id
        """
        prefix = "s."
    else:
        from_clause = "FROM segments"
        prefix = ""
    
    if created_at_filter:
        where_clauses.append(f"{prefix}created_at >= %s")
        params.append(created_at_filter)
    
    if source:
        where_clauses.append(f"{prefix}source = %s")
        params.append(source)
    
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    with conn.cursor() as cur:
        if require_all_embeddings:
            cur.execute(f"""
                SELECT DISTINCT {prefix}video_id
                {from_clause}
                {where_sql}
                ORDER BY {prefix}video_id
            """, tuple(params) if params else None)
        else:
            cur.execute(f"""
                SELECT DISTINCT video_id
                {from_clause}
                {where_sql}
                ORDER BY video_id
            """, tuple(params) if params else None)
        return [row[0] for row in cur.fetchall()]


def analyze_video_id_distribution(conn: psycopg2.extensions.connection, created_at_filter: str):
    """
    Analyze video_id distribution across created_at batches to explain discrepancies.
    
    Returns:
        Dictionary with statistics about video_id distribution
    """
    with conn.cursor() as cur:
        # Count total videos per created_at
        cur.execute("""
            SELECT 
                created_at,
                COUNT(DISTINCT video_id) as num_videos
            FROM segments
            WHERE created_at >= %s
            GROUP BY created_at
            ORDER BY created_at
        """, (created_at_filter,))
        per_batch = {row[0]: row[1] for row in cur.fetchall()}
        
        # Count unique video_ids overall
        cur.execute("""
            SELECT COUNT(DISTINCT video_id)
            FROM segments
            WHERE created_at >= %s
        """, (created_at_filter,))
        unique_total = cur.fetchone()[0]
        
        # Find video_ids that appear in multiple batches
        cur.execute("""
            SELECT 
                video_id,
                COUNT(DISTINCT created_at) as num_batches
            FROM segments
            WHERE created_at >= %s
            GROUP BY video_id
            HAVING COUNT(DISTINCT created_at) > 1
            ORDER BY num_batches DESC, video_id
        """, (created_at_filter,))
        duplicates = cur.fetchall()
        
        return {
            "per_batch": per_batch,
            "unique_total": unique_total,
            "sum_per_batch": sum(per_batch.values()),
            "duplicates": duplicates,
        }


def fetch_video_data(
    conn: psycopg2.extensions.connection,
    video_id: str,
    created_at_filter: Optional[str] = None,
    source: Optional[str] = None,
    deduplicate: bool = True,
) -> List[dict]:
    """
    Fetch all segments and embeddings for a given video_id.
    
    Args:
        conn: Database connection
        video_id: The video_id to fetch
        created_at_filter: Optional filter for created_at >= this date
        source: Optional filter by source (e.g., 'ShareVeo3')
        deduplicate: If True, deduplicate by segment_id (keep first occurrence)
    
    Returns:
        List of dictionaries with segment data and embeddings
    """
    where_clauses = ["s.video_id = %s"]
    params = [video_id]
    
    if created_at_filter:
        where_clauses.append("s.created_at >= %s")
        params.append(created_at_filter)
    
    if source:
        where_clauses.append("s.source = %s")
        params.append(source)
    
    where_sql = "WHERE " + " AND ".join(where_clauses)
    
    query = f"""
        SELECT 
            s.segment_id,
            s.video_path,
            s.start_time,
            s.audio_label,
            s.video_label,
            s.duration,
            s.created_at,
            s.source,
            e1.embedding::float4[] AS openl3_embedding,
            e2.embedding::float4[] AS hubert_embedding,
            e3.embedding::float4[] AS senet_embedding
        FROM segments s
        LEFT JOIN embeddings_audio_openl3 e1 ON s.segment_id = e1.segment_id
        LEFT JOIN embeddings_audio_hubert e2 ON s.segment_id = e2.segment_id
        LEFT JOIN embeddings_video_senet e3 ON s.segment_id = e3.segment_id
        {where_sql}
        ORDER BY s.created_at, s.video_path, s.start_time
    """
    
    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        
        results = []
        seen_segment_ids = set()
        duplicates_skipped = 0
        
        for row in rows:
            row_dict = dict(zip(columns, row))
            segment_id = row_dict["segment_id"]
            
            # Deduplicate by segment_id if requested
            if deduplicate:
                if segment_id in seen_segment_ids:
                    duplicates_skipped += 1
                    continue
                seen_segment_ids.add(segment_id)
            
            # Only include segments that have all 3 embeddings
            has_all_embeddings = (
                row_dict["openl3_embedding"] is not None
                and row_dict["hubert_embedding"] is not None
                and row_dict["senet_embedding"] is not None
            )
            
            if not has_all_embeddings:
                continue  # Skip segments without all 3 embeddings
            
            # Convert embedding lists to numpy arrays
            for emb_type in ["openl3_embedding", "hubert_embedding", "senet_embedding"]:
                row_dict[emb_type] = np.array(row_dict[emb_type], dtype=np.float32)
            results.append(row_dict)
        
        if duplicates_skipped > 0:
            print(f"  ⚠️  {video_id}: Skipped {duplicates_skipped} duplicate segment_ids")
        
        return results

