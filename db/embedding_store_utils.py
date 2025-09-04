import sqlite3

def insert_segment(db_path, segment_dict):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO segments (
                segment_id, source, video_id, video_path,
                start_time, duration, video_label, audio_label, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            segment_dict["segment_id"],
            segment_dict["source"],
            segment_dict["video_id"],
            segment_dict["video_path"],
            segment_dict["start_time"],
            segment_dict["duration"],
            segment_dict["video_label"],
            segment_dict["audio_label"],
            segment_dict.get("created_at")  # Can be None; DB will default it
        ))
        conn.commit()


def get_segments_by_video_id(db_path, video_id):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM segments WHERE video_id = ?
        """, (video_id,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]


def insert_embedding(db_path, embedding_dict):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO embeddings (
                embedding_id, segment_id, mode, noise, model_name,
                embedding_type, reducer_id, contraster_id,
                embedding_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            embedding_dict["embedding_id"],
            embedding_dict["segment_id"],
            embedding_dict["mode"],
            embedding_dict["noise"],
            embedding_dict["model_name"],
            embedding_dict["embedding_type"],
            embedding_dict["reducer_id"],
            embedding_dict["contraster_id"],
            embedding_dict["embedding_path"],
            embedding_dict.get("created_at")
        ))
        conn.commit()


def get_embeddings_by_segment(db_path, segment_id):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM embeddings WHERE segment_id = ?
        """, (segment_id,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]


def get_segments_by_created_at(db_path, created_at):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM segments
            WHERE created_at == ?
        """, (created_at,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
