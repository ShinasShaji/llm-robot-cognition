import chromadb
from typing import List, Dict, Any, Optional
import uuid

from logging_utils import get_logger

# Initialize logger
logger = get_logger("MEMORY")


class Memory:
    """
    A memory class that uses ChromaDB for persistent storage and search of memories.
    """
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize the Memory class with a persistent ChromaDB client.
        
        Args:
            persist_directory (str): Directory to store the persistent database
        """
        # Create a persistent client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create a collection for memories
        self.collection = self.client.get_or_create_collection(
            name="memories",
            metadata={"description": "Collection of robot memories"}
        )
    
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a memory to the collection.
        
        Args:
            content (str): The content of the memory
            metadata (dict, optional): Additional metadata for the memory
            
        Returns:
            str: ID of the added memory
        """
        # Generate a unique ID for the memory
        memory_id = str(uuid.uuid4())
        
        # Add the memory to the collection
        self.collection.add(
            documents=[content],
            metadatas=[metadata] if metadata else [{}],
            ids=[memory_id]
        )
        
        return memory_id
    
    def search_memories(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for memories based on a query.
        
        Args:
            query (str): The search query
            n_results (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of matching memories with their content and metadata
        """
        # Perform the search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format the results
        memories = []
        if results['ids'] and results['documents']:
            for i in range(len(results['ids'][0])):
                memory = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                }
                memories.append(memory)
        
        return memories
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by its ID.
        
        Args:
            memory_id (str): The ID of the memory to retrieve
            
        Returns:
            Dict[str, Any] or None: The memory data or None if not found
        """
        try:
            result = self.collection.get(ids=[memory_id])
            if result['ids'] and result['documents']:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0] if result['metadatas'] else {}
                }
            return None
        except Exception:
            return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by its ID.
        
        Args:
            memory_id (str): The ID of the memory to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False
    
    def delete_all_memories(self) -> bool:
        """
        Delete all memories from the collection.
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Get all memory IDs first
            all_memories = self.get_all_memories()
            if all_memories:
                # Extract all IDs and delete them
                all_ids = [memory['id'] for memory in all_memories]
                self.collection.delete(ids=all_ids)

                logger.info(f"Deleted {len(all_ids)} (all) memories")
            return True
        except Exception:
            return False
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Retrieve all memories in the collection.
        
        Returns:
            List[Dict[str, Any]]: List of all memories
        """
        results = self.collection.get()
        
        memories = []
        if results['ids'] and results['documents']:
            for i in range(len(results['ids'])):
                memory = {
                    'id': results['ids'][i],
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i] if results['metadatas'] else {}
                }
                memories.append(memory)
        
        return memories