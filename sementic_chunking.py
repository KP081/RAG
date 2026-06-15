from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = SentenceTransformer("BAAI/bge-m3")

document = '''
# Authentication Guide

## OAuthAuthentication

Authentication is the process of verifying the identity of a user before granting access to an application or API. One common authentication method is OAuthAuthentication. OAuthAuthentication allows users to log in through trusted providers such as Google, GitHub, or Microsoft without sharing their passwords directly with the application. This approach improves security and simplifies the login experience for users.

## Rate Limiting

Rate limiting is an important mechanism used to protect APIs from excessive requests. It restricts the number of requests that a client can send within a specific time period. For example, an API may allow only 100 requests per minute for a single user. If the limit is exceeded, the API returns a response indicating that the client should wait before sending more requests. Rate limiting helps maintain system stability and prevents abuse.

## Error Handling

Error handling ensures that applications respond gracefully when something goes wrong. Authentication systems should provide clear and meaningful error messages for situations such as invalid credentials, expired access tokens, or insufficient permissions. Proper error handling helps developers identify issues quickly and improves the overall user experience by providing understandable feedback.

## Webhooks

Webhooks enable real-time communication between systems. Instead of repeatedly checking for updates, an application can register a webhook URL and receive notifications automatically when specific events occur. For example, a payment service may send a webhook notification when a transaction is completed, or an authentication service may notify another system when a user successfully logs in. Webhooks reduce unnecessary API requests and allow systems to react immediately to important events.
'''

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=['\n\n', '\n', '. ', ' ']
)

recursive_chunks = recursive_splitter.split_text(document)

print(f'\nRecursive Chunks: {len(recursive_chunks)}')
for i, chunk in enumerate(recursive_chunks):
    print(f'\\n--- Chunk {i+1} ({len(chunk)} chars) ---')
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    
    
    

semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90 # split at 90th percentile dissimilarity
)

semantic_chunks = semantic_chunker.split_text(document)

print(f'\nSemantic Chunks: {len(semantic_chunks)}')
for i, chunk in enumerate(semantic_chunks):
    print(f'\\n--- Chunk {i+1} ({len(chunk)} chars) ---')
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)