#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingsManager:
    """Manages text embeddings for retrieval augmented generation (RAG)"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 academic_model_name="allenai/scibert_scivocab_uncased",
                 device=None, use_reranker=False):
        """
        Initialize the embeddings manager
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            academic_model_name (str): Name of the model for academic papers
            device (str, optional): Device to run models on ('cpu', 'cuda', etc.)
            use_reranker (bool): Whether to use reranking for search results
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize primary embedding model
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Initialize academic corpus model if requested
        self.academic_model = None
        self.academic_tokenizer = None
        if academic_model_name:
            self.academic_model = AutoModel.from_pretrained(academic_model_name)
            self.academic_tokenizer = AutoTokenizer.from_pretrained(academic_model_name)
            self.academic_model.to(self.device)
            
        # Whether to use reranking
        self.use_reranker = use_reranker
        
        # In-memory database for embeddings
        self.document_store = {
            'documents': [],
            'embeddings': [],
            'metadata': []
        }
        
        # FAISS index for fast similarity search
        self.index = None
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
    def add_document(self, text, metadata=None):
        """
        Add a document to the store
        
        Args:
            text (str): Text to embed
            metadata (dict, optional): Metadata for the document
        
        Returns:
            int: Index of the added document
        """
        # Generate embedding
        embedding = self.get_embedding(text)
        
        # Add to store
        doc_idx = len(self.document_store['documents'])
        self.document_store['documents'].append(text)
        self.document_store['embeddings'].append(embedding)
        self.document_store['metadata'].append(metadata or {})
        
        # Update index if it exists
        if self.index is not None:
            if len(self.document_store['embeddings']) == 1:
                # Initialize index with the first embedding
                self._init_index()
            else:
                # Add to existing index
                self.index.add(np.array([embedding], dtype=np.float32))
                
        return doc_idx
    
    def add_documents(self, texts, metadatas=None):
        """
        Add multiple documents to the store
        
        Args:
            texts (list): List of texts to embed
            metadatas (list, optional): List of metadata for the documents
            
        Returns:
            list: List of indices of the added documents
        """
        # Generate embeddings in batch
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Add to store
        start_idx = len(self.document_store['documents'])
        indices = list(range(start_idx, start_idx + len(texts)))
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            self.document_store['documents'].append(text)
            self.document_store['embeddings'].append(embedding)
            self.document_store['metadata'].append(metadata)
            
        # Update index
        if self.index is None:
            self._init_index()
        else:
            self.index.add(np.array(embeddings, dtype=np.float32))
            
        return indices
    
    def get_embedding(self, text):
        """
        Get embedding for a text
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def get_academic_embedding(self, text):
        """
        Get embedding for academic text
        
        Args:
            text (str): Academic text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not self.academic_model:
            return self.get_embedding(text)
            
        # Tokenize
        inputs = self.academic_tokenizer(text, return_tensors="pt", 
                                         truncation=True, max_length=512).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            outputs = self.academic_model(**inputs)
            
        # Use CLS token as embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]
    
    def search(self, query, k=5, academic=False):
        """
        Search for similar documents
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            academic (bool): Whether to use academic model for embedding
            
        Returns:
            list: List of dicts with document, score, and metadata
        """
        if not self.document_store['documents']:
            return []
            
        # Get query embedding
        if academic and self.academic_model:
            query_embedding = self.get_academic_embedding(query)
        else:
            query_embedding = self.get_embedding(query)
            
        # Initialize index if necessary
        if self.index is None:
            self._init_index()
            
        # Search in FAISS index
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query_embedding_np, k=min(k, len(self.document_store['documents'])))
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # FAISS may return -1 for not enough results
                continue
                
            results.append({
                'document': self.document_store['documents'][idx],
                'score': float(scores[0][i]),
                'metadata': self.document_store['metadata'][idx]
            })
            
        # Rerank if requested
        if self.use_reranker and len(results) > 1:
            results = self._rerank_results(query, results)
            
        return results
    
    def _init_index(self):
        """Initialize FAISS index with current embeddings"""
        embeddings = np.array(self.document_store['embeddings'], dtype=np.float32)
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        if len(embeddings) > 0:
            self.index.add(embeddings)
    
    def _rerank_results(self, query, results):
        """
        Rerank results using more expensive but accurate comparison
        
        Args:
            query (str): Original query
            results (list): Initial results
            
        Returns:
            list: Reranked results
        """
        # In a real implementation, you would use a cross-encoder model or
        # another reranking approach here
        
        # For this example, we'll just return the original results
        return results
    
    def clear(self):
        """Clear the document store and index"""
        self.document_store = {
            'documents': [],
            'embeddings': [],
            'metadata': []
        }
        self.index = None 