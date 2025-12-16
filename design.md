# Berlin Media Archive RAG System - Design Summary

## System Overview

The Berlin Media Archive RAG system is a production-grade retrieval system that enables researchers to search across 50+ years of archived content including audio interviews, text documents, and potentially video footage. The system uses Google's embedding models to convert content into searchable vectors stored in Pinecone, then employs Gemini LLM to generate answers with strict source citations. Every response includes precise timestamps for audio (e.g., "04:20") or page numbers for documents, ensuring researchers can verify and reference original sources.

## Scaling to Production Volume

To handle 1,000 hours of audio and 10+ million tokens of text, the architecture shifts from real-time processing to distributed batch processing. Instead of processing files sequentially, the system would use 16-32 parallel workers to ingest content over 10 days, with checkpointing to resume from failures. The vector database would be organized into namespaces (by year/content type) for faster queries, and a multi-layer Redis cache would reduce API calls by 70%. This approach reduces monthly costs from $99 to approximately $65-70 while handling 100,000 queries per month, with costs scaling efficiently as usage grows (cost per query drops from $0.00099 to $0.00034 at 10M queries/month).

## Cost Analysis

The initial ingestion costs about $126 (using Groq for transcription at $72, Google embeddings at $1.69, and GPU-based speaker diarization at $52.60). Monthly operating costs total approximately $99, broken down as: $24 for LLM generation (Gemini Flash), $20 for vector storage (Pinecone serverless), $36 for infrastructure (app server, Redis cache, S3 storage), and $3.65 for monthly content updates. The largest ongoing cost is the LLM API for generating attributed answers, though aggressive caching can reduce this by 70%. Storage is relatively inexpensive due to efficient vector compression and the use of S3 Glacier for archival content.

## Video Expansion Strategy

Adding video search requires extracting keyframes (approximately 1.8 million frames from 1,000 hours), then using CLIP (OpenAI's image-text model) to create embeddings that work in a shared space—meaning text queries like "Berlin TV Tower" can directly find relevant video frames. The system would augment each keyframe with object detection tags (identifying buildings, people, towers), scene classification (indoor interview vs. outdoor landmark), and OCR to extract visible text. Video and audio chunks are linked bi-directionally, so a visual search result can show both the frame and what was being said at that moment. This adds approximately $220/month in storage costs and $3,739 in one-time processing, but enables powerful cross-modal queries like "Show me footage where someone discusses urban planning while pointing at a map."

## Production Readiness

The system is designed with production standards including comprehensive error handling (graceful degradation when APIs fail), observability (detailed logging of which chunks were retrieved for each query), and automated testing. The architecture supports incremental scaling: starting with the current MVP, then adding speaker filtering, hybrid search for precise date queries, and eventually video capabilities. Each component can be optimized independently, and the use of serverless infrastructure (Pinecone, managed Redis, container hosting) minimizes operational overhead while maintaining 99.9% uptime requirements for serving 500 concurrent researchers.
