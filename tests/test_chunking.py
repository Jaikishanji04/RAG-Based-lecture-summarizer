from core.chunking import chunk_transcript


def test_transcript_chunks_keep_timestamp_metadata() -> None:
    segments = [
        {"start": 0, "end": 2, "text": "The first topic introduces retrieval."},
        {"start": 2, "end": 4, "text": "It connects documents to a question."},
    ]
    chunks = chunk_transcript(segments, "lecture-1", target_words=1, overlap_words=0)
    assert chunks
    assert chunks[0].lecture_id == "lecture-1"
    assert chunks[0].start_time == 0.0
    assert chunks[0].slide_no is None
