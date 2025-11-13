-- SQL Queries for Analyzing Duplicate Video IDs Across Batches
-- Run these in your Neon database client to investigate duplicates

-- 1. Find video_ids that appear in multiple batches
SELECT 
    video_id,
    COUNT(DISTINCT created_at) as num_batches,
    STRING_AGG(DISTINCT created_at::text, ', ' ORDER BY created_at::text) as batch_timestamps
FROM segments
WHERE created_at >= '2025-11-01 00:00:00'
GROUP BY video_id
HAVING COUNT(DISTINCT created_at) > 1
ORDER BY num_batches DESC, video_id;

-- 2. For a specific video_id, show segment counts per batch
-- Replace 'gqpErbFnbiY/00007' with the video_id you want to analyze
SELECT 
    created_at,
    COUNT(*) as num_segments,
    COUNT(DISTINCT video_path) as num_video_paths,
    COUNT(DISTINCT segment_id) as unique_segment_ids
FROM segments
WHERE video_id = 'gqpErbFnbiY/00007'
  AND created_at >= '2025-11-01 00:00:00'
GROUP BY created_at
ORDER BY created_at;

-- 3. Check for exact duplicate segment_ids (same segment_id in multiple batches)
SELECT 
    segment_id,
    COUNT(*) as num_occurrences,
    COUNT(DISTINCT created_at) as num_batches,
    STRING_AGG(DISTINCT created_at::text, ', ' ORDER BY created_at::text) as batch_timestamps,
    COUNT(DISTINCT video_path) as num_video_paths
FROM segments
WHERE video_id = 'gqpErbFnbiY/00007'
  AND created_at >= '2025-11-01 00:00:00'
GROUP BY segment_id
HAVING COUNT(*) > 1
ORDER BY num_occurrences DESC;

-- 4. Check for exact duplicate rows (all fields same except created_at)
-- This finds rows where everything is identical except the timestamp
SELECT 
    source,
    video_path,
    start_time,
    duration,
    audio_label,
    video_label,
    audio_model,
    video_model,
    COUNT(*) as num_occurrences,
    COUNT(DISTINCT created_at) as num_batches,
    STRING_AGG(DISTINCT created_at::text, ', ' ORDER BY created_at::text) as batch_timestamps,
    STRING_AGG(DISTINCT segment_id, ', ' ORDER BY segment_id) as segment_ids
FROM segments
WHERE video_id = 'gqpErbFnbiY/00007'
  AND created_at >= '2025-11-01 00:00:00'
GROUP BY source, video_path, start_time, duration, audio_label, video_label, audio_model, video_model
HAVING COUNT(*) > 1
ORDER BY num_occurrences DESC;

-- 5. Compare video_paths between batches for a duplicate video_id
SELECT 
    created_at,
    video_path,
    COUNT(*) as num_segments
FROM segments
WHERE video_id = 'gqpErbFnbiY/00007'
  AND created_at >= '2025-11-01 00:00:00'
GROUP BY created_at, video_path
ORDER BY created_at, video_path;

-- 6. Check segment_id overlap between two specific batches
-- Replace the timestamps with actual batch timestamps
WITH batch1 AS (
    SELECT segment_id, video_path, start_time, duration, audio_label, video_label
    FROM segments
    WHERE video_id = 'gqpErbFnbiY/00007'
      AND created_at = '2025-11-05 17:31:18.485225+00'
),
batch2 AS (
    SELECT segment_id, video_path, start_time, duration, audio_label, video_label
    FROM segments
    WHERE video_id = 'gqpErbFnbiY/00007'
      AND created_at = '2025-11-11 03:07:09.919316+00'
)
SELECT 
    'In batch1 only' as location,
    COUNT(*) as count
FROM batch1
WHERE segment_id NOT IN (SELECT segment_id FROM batch2)
UNION ALL
SELECT 
    'In batch2 only' as location,
    COUNT(*) as count
FROM batch2
WHERE segment_id NOT IN (SELECT segment_id FROM batch1)
UNION ALL
SELECT 
    'In both batches' as location,
    COUNT(*) as count
FROM batch1
WHERE segment_id IN (SELECT segment_id FROM batch2);

-- 7. Show all segments for a duplicate video_id with their batch info
SELECT 
    segment_id,
    created_at,
    video_path,
    start_time,
    duration,
    audio_label,
    video_label,
    audio_model,
    video_model
FROM segments
WHERE video_id = 'gqpErbFnbiY/00007'
  AND created_at >= '2025-11-01 00:00:00'
ORDER BY created_at, video_path, start_time;

-- 8. Count total segments vs unique segment_ids for duplicate video_ids
SELECT 
    video_id,
    COUNT(*) as total_segments,
    COUNT(DISTINCT segment_id) as unique_segment_ids,
    COUNT(*) - COUNT(DISTINCT segment_id) as duplicate_count,
    COUNT(DISTINCT created_at) as num_batches
FROM segments
WHERE created_at >= '2025-11-01 00:00:00'
  AND video_id IN (
      SELECT video_id
      FROM segments
      WHERE created_at >= '2025-11-01 00:00:00'
      GROUP BY video_id
      HAVING COUNT(DISTINCT created_at) > 1
  )
GROUP BY video_id
ORDER BY duplicate_count DESC, video_id;

