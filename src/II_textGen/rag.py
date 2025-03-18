import os
import json
import numpy as np
import faiss
from typing import List, Dict, Optional, Union, Callable, Any, Tuple
from dataclasses import dataclass
import logging

# Configure logging to suppress FAISS messages
logging.basicConfig(level=logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

# Import without printing warnings
try:
    import faiss
except ImportError:
    print("Warning: FAISS not installed. Vector search functionality will be limited.")
    faiss = None

# Ensure we can easily import our OpenAI wrapper
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "I_integrations")))
try:
    from openai_API import OpenAIAPI
except ImportError:
    print("Warning: Could not import OpenAIAPI. Make sure openai_API.py is in the I_integrations directory.")
    OpenAIAPI = None

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TextSplitter:
    """
    Splits text into chunks using different strategies.
    Handles both basic splitting and semantic-aware splitting.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable = len,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.semantic_markers = [
            # Headers
            "# ", "## ", "### ", "####",
            # Document sections
            "Chapter ", "Section ", "Introduction", "Conclusion",
            # Transition markers
            "First,", "Second,", "Finally,", "In conclusion",
            "For example,", "However,", "Moreover," 
        ]
    
    def split_text(self, text: str, use_semantic: bool = True) -> List[TextChunk]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: The text to split
            use_semantic: Whether to use semantic-aware splitting
            
        Returns:
            List of TextChunk objects
        """
        if use_semantic:
            return self._split_semantic(text)
        
        # Try each separator in order
        for sep_idx, separator in enumerate(self.separators):
            chunks = []
            
            if separator == "":
                # Last resort: character splitting
                current_chunk = ""
                for char in text:
                    if self.length_function(current_chunk + char) <= self.chunk_size:
                        current_chunk += char
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = char
                if current_chunk:
                    chunks.append(current_chunk)
            else:
                # Try splitting by this separator
                segments = text.split(separator)
                current_chunk = []
                current_length = 0
                
                for segment in segments:
                    segment_len = self.length_function(segment)
                    
                    # If segment is too long for a chunk, try a smaller separator
                    if segment_len > self.chunk_size and sep_idx < len(self.separators) - 1:
                        if current_chunk:
                            chunks.append(separator.join(current_chunk))
                        
                        # Recursively split this segment
                        subsplitter = TextSplitter(
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.chunk_overlap,
                            separators=self.separators[sep_idx+1:]
                        )
                        sub_chunks = subsplitter.split_text(segment, use_semantic=False)
                        chunks.extend([chunk.text for chunk in sub_chunks])
                        
                        current_chunk = []
                        current_length = 0
                        continue
                    
                    # Add to current chunk if it fits
                    if current_length + segment_len + len(separator) <= self.chunk_size:
                        current_chunk.append(segment)
                        current_length += segment_len + len(separator)
                    else:
                        # Save current chunk and start a new one
                        if current_chunk:
                            chunks.append(separator.join(current_chunk))
                        current_chunk = [segment]
                        current_length = segment_len
                
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
            
            # If we have valid chunks with this separator
            if chunks:
                # Add overlap between chunks
                if self.chunk_overlap > 0 and len(chunks) > 1:
                    overlapped = []
                    for i, chunk in enumerate(chunks):
                        if i > 0:
                            words = chunks[i-1].split()
                            overlap = " ".join(words[-min(self.chunk_overlap, len(words)):])
                            chunk = overlap + separator + chunk
                        overlapped.append(chunk)
                    chunks = overlapped
                
                # Create TextChunks with metadata
                return [
                    TextChunk(
                        text=chunk,
                        metadata={
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "separator": separator
                        }
                    ) for i, chunk in enumerate(chunks)
                ]
        
        # Fallback empty list if all splitters failed
        return []
    
    def _split_semantic(self, text: str) -> List[TextChunk]:
        """
        Split text with awareness of semantic boundaries.
        Prioritizes keeping related content together.
        """
        # First check for major section markers
        sections = []
        current_section = []
        section_start = 0
        
        lines = text.split("\n")
        for i, line in enumerate(lines):
            # Check if line starts with any semantic marker
            if any(line.strip().startswith(marker) for marker in self.semantic_markers):
                if current_section:
                    sections.append("\n".join(current_section))
                    current_section = []
                section_start = i
            current_section.append(line)
            
        # Add final section
        if current_section:
            sections.append("\n".join(current_section))
        
        # If no sections were found or sections are too large, fall back to regular splitting
        if not sections or any(self.length_function(s) > self.chunk_size * 2 for s in sections):
            return self.split_text(text, use_semantic=False)
        
        # Process each section
        chunks = []
        for section in sections:
            # If section fits in a chunk, keep it whole
            if self.length_function(section) <= self.chunk_size:
                chunks.append(
                    TextChunk(
                        text=section,
                        metadata={"semantic_section": True}
                    )
                )
            else:
                # Split large sections using regular method
                section_chunks = self.split_text(section, use_semantic=False)
                # Mark these as belonging to the same semantic section
                for i, chunk in enumerate(section_chunks):
                    chunk.metadata["semantic_section"] = True
                    chunk.metadata["section_part"] = i
                    chunk.metadata["section_total"] = len(section_chunks)
                chunks.extend(section_chunks)
        
        # Set overall indices
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks

class RAG:
    """
    Retrieval-Augmented Generation system with FAISS vector search.
    Integrates with OpenAI API for embeddings and completions.
    """
    
    def __init__(
        self, 
        openai_api: Optional[Any] = None,
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        use_semantic_splitting: bool = True
    ):
        """
        Initialize the RAG system.
        
        Args:
            openai_api: An instance of OpenAIAPI (will create one if None)
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
            use_semantic_splitting: Whether to use semantic-aware text splitting
        """
        # Initialize OpenAI API if not provided
        if openai_api is None:
            if OpenAIAPI is not None:
                self.openai = OpenAIAPI()
            else:
                raise ImportError("OpenAIAPI not available and no API instance provided")
        else:
            self.openai = openai_api
            
        self.splitter = TextSplitter(chunk_size, chunk_overlap)
        self.use_semantic = use_semantic_splitting
        
        # FAISS index (initialized when first used)
        self.index = None
        self.chunks = []
        self.embedding_dimension = None
        
    def add_documents(
        self,
        documents: Union[str, List[str]],
        document_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> List[TextChunk]:
        """
        Process documents into chunks and add to the vector store.
        
        Args:
            documents: Text document(s) to process
            document_ids: Optional IDs for each document 
            metadata: Optional metadata for each document
            
        Returns:
            List of processed text chunks
        """
        if isinstance(documents, str):
            documents = [documents]
            
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]
            
        if metadata is None:
            metadata = [{} for _ in documents]
            
        # Ensure lists are the same length
        if len(documents) != len(document_ids) or len(documents) != len(metadata):
            raise ValueError("documents, document_ids, and metadata must have the same length")
        
        # Split documents into chunks
        all_chunks = []
        for doc, doc_id, meta in zip(documents, document_ids, metadata):
            # Split the document
            chunks = self.splitter.split_text(doc, use_semantic=self.use_semantic)
            
            # Add document info to chunk metadata
            for chunk in chunks:
                chunk.metadata.update({
                    "document_id": doc_id,
                    **meta
                })
                
            all_chunks.extend(chunks)
            
        # Create embeddings for all chunks
        chunk_texts = [chunk.text for chunk in all_chunks]
        chunk_embeddings = self.openai.create_embeddings(chunk_texts)
        
        # Initialize FAISS index if needed
        if self.index is None:
            self.embedding_dimension = chunk_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
        
        # Add embeddings to index
        self.index.add(chunk_embeddings)
        
        # Store chunks for later retrieval
        self.chunks.extend(all_chunks)
        
        return all_chunks
    
    def search(
        self,
        query: str,
        k: int = 3
    ) -> List[Dict]:
        """
        Search for documents similar to query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of dicts with text, metadata, and similarity score
        """
        if self.index is None or not self.chunks:
            return []
            
        # Create query embedding
        query_embedding = self.openai.create_embeddings(query)
        
        # Search the index
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks) and idx >= 0:  # Ensure valid index
                chunk = self.chunks[idx]
                # Calculate cosine similarity from L2 distance
                # sim = 1 - (distance / 2)  # Approximate for normalized vectors
                results.append({
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "score": float(distance)  # L2 distance (lower is better)
                })
                
        return results
    
    def clear(self):
        """Clear all documents and reset the index."""
        self.chunks = []
        if self.embedding_dimension:
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
        else:
            self.index = None

    def ingest_documents(self, document: str, embedding_function: callable) -> None:
        """
        Ingest documents for RAG retrieval.
        
        Args:
            document: Text content to ingest
            embedding_function: Function to create embeddings
        """
        # Split document into chunks
        chunks = self._chunk_text(document)
        
        # Create embeddings for chunks
        embeddings = [embedding_function(chunk) for chunk in chunks]
        
        # Store chunks and embeddings
        self.document_chunks = chunks
        self.document_embeddings = embeddings
        
    def retrieve_context(self, document: str, query: str, embedding_function: callable, top_k: int = 5) -> str:
        """
        Retrieve relevant context based on query.
        
        Args:
            document: Original document text (not used in this simple implementation)
            query: Search query
            embedding_function: Function to create embeddings
            top_k: Number of top chunks to retrieve
            
        Returns:
            Concatenated text of top relevant chunks
        """
        # If no documents ingested, return empty string
        if not hasattr(self, 'document_chunks') or not self.document_chunks:
            return ""
        
        # Create query embedding
        query_embedding = embedding_function(query)
        
        # Simple implementation: return the first few chunks
        # (In a real implementation, we would compute similarity and return top chunks)
        return "\n\n".join(self.document_chunks[:min(top_k, len(self.document_chunks))])

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Simple implementation: split by rough character count
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Try to find a sentence break
            if end < len(text):
                for marker in ['. ', '! ', '? ', '\n\n']:
                    pos = text.rfind(marker, start, end + 20)
                    if pos > start:
                        end = pos + 2  # Include the sentence end marker
                        break
            
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else len(text)
        
        return chunks

# Example usage
if __name__ == "__main__":
    # Import OpenAI API wrapper
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "I_integrations")))
    try:
        from openai_API import OpenAIAPI
    except ImportError:
        print("Error: Could not import OpenAIAPI. Make sure openai_API.py is in the I_integrations directory.")
        exit(1)
    
    # Initialize the OpenAI API and RAG system
    openai_api = OpenAIAPI()
    rag = RAG(openai_api)
    
    # Example document about AI
    ai_doc = """# Artificial Intelligence
    
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to human or animal intelligence. 
AI applications include advanced web search engines, recommendation systems, 
language translation, autonomous driving, and creating art.

## Machine Learning

Machine learning (ML) is a field of study in artificial intelligence concerned with the development of algorithms 
that can learn from and make decisions based on data. ML algorithms build a model based on sample data to 
make predictions without being explicitly programmed to do so.

### Types of Machine Learning

1. **Supervised Learning**: The algorithm is trained on labeled data.
2. **Unsupervised Learning**: The algorithm finds patterns in unlabeled data.
3. **Reinforcement Learning**: The algorithm learns by receiving feedback from its actions.

## Deep Learning

Deep learning is a subset of machine learning that uses neural networks with multiple layers. 
These deep neural networks are capable of learning complex patterns in large amounts of data.

## Applications of AI

AI has numerous applications across various industries:

- **Healthcare**: Disease diagnosis, drug discovery
- **Finance**: Fraud detection, algorithmic trading
- **Transportation**: Autonomous vehicles, traffic prediction
- **Entertainment**: Content recommendation, game AI
- **Manufacturing**: Quality control, predictive maintenance
"""

    # Add document to RAG
    rag.add_documents(ai_doc, document_ids=["ai_overview"], metadata=[{"author": "OpenAI"}])
    
    # Query the RAG system
    query = "What are the main types of machine learning?"
    
    # Get retrieved chunks
    retrieved_chunks = rag.search(query, k=3)
    
    # Print retrieved chunks
    print("\n=== Retrieved Chunks ===\n")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"Chunk {i+1} (score: {chunk['score']:.4f}):")
        print(f"{chunk['text'][:150]}...\n")
    
    # Demonstrate how to use RAG results with OpenAI API
    # This would normally be in the textGen module
    if retrieved_chunks:
        # Format retrieved context
        context_text = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])
        prompt = f"""Answer the following question based on the provided context:

CONTEXT:
{context_text}

QUESTION:
{query}

YOUR ANSWER:"""
        
        # Generate response using OpenAI API
        system_prompt = "You are a helpful assistant that answers based on the provided context."
        response = openai_api.chat_completion(
            user_prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        print("\n=== Generated Response ===\n")
        print(response)
    else:
        print("\nNo relevant chunks found for the query.") 