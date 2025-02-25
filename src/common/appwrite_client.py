import uuid
from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.services.databases import Databases
from appwrite.input_file import InputFile
from appwrite.query import Query
from src.common.logging_utils import get_logger
import re
logger = get_logger(__name__)
# Get ENV variables
import os 
from dotenv import load_dotenv
load_dotenv()

# import pandas as pd
client = Client()
(client
  .set_endpoint(os.getenv("API_ENDPOINT")) # Your API Endpoint
  .set_project(os.getenv("PROJECT_ID")) # Your project ID
  .set_key(os.getenv("API_KEY")) # Your secret API key
  .set_self_signed() # Use only on dev mode with a self-signed SSL cert
)

# Initialize the Databases service
databases = Databases(client)
storage   = Storage(client)
# Define your Appwrite database and collection IDs
DATABASE_ID = os.getenv("DATABASE_ID")
BUCKET_ID = os.getenv("BUCKET_ID")                         # Bucket for storing image files

def upload_document(collection_id, document_id, data):
    """Helper function to upload a document to a specified Appwrite collection."""
    try:
        # Verify document_id is not None or empty and meets Appwrite requirements
        if not document_id:
            raise ValueError("Document ID cannot be empty")
            
        # Log the document ID for debugging
        logger.info(f"Uploading document with ID: '{document_id}'")
        logger.debug(f"Parameters: database_id={DATABASE_ID}, collection_id={collection_id}, data={data}")
        
        response = databases.create_document(
            database_id=DATABASE_ID,
            collection_id=collection_id,
            document_id=document_id,  # Make sure this parameter name matches the API
            data=data
        )
        logger.info(f"Successfully uploaded document to collection '{collection_id}': {response}")
        return response
    except Exception as e:
        logger.error(f"Error uploading document to collection '{collection_id}': {e}")
        logger.error(f"Document ID: {document_id}")
        logger.error(f"Data: {data}")
        raise e

def upload_file_to_bucket(file_path):
    """
    Uploads a file to the specified Appwrite Storage bucket.
    Returns the file ID if successful, otherwise None.
    """
    try:
        # Generate a unique file ID
        file_id = str(uuid.uuid4())
        
        # Open and read the file
        with open(file_path, "rb") as file:
            file_data = file.read()
            
        # Create an InputFile object
        input_file = InputFile.from_bytes(file_data, filename=file_path.split("/")[-1])
        
        response = storage.create_file(
            bucket_id=BUCKET_ID,
            file_id=file_id,
            file=input_file
        )
        logger.info("File uploaded successfully: %s", response)
        return response.get("$id")
    except Exception as e:
        logger.error("Error uploading file: %s", e)
        return None

def delete_all_documents(collection_id):
    """
    Deletes all documents from a specified collection, handling pagination.
    Returns the number of documents deleted.
    
    Args:
        collection_id (str): The ID of the collection to clear
    
    Returns:
        int: Number of documents deleted
    """
    try:
        deleted_count = 0
        page = 1
        while True:
            # Get documents with pagination (limit=100 is max allowed by Appwrite)
            documents = databases.list_documents(
                database_id=DATABASE_ID,
                collection_id=collection_id,
                queries=[
                    Query.limit(100)
                ]
            )
            
            if not documents.get('documents'):
                break  # No more documents to delete
                
            # Delete each document in the current page
            for doc in documents.get('documents', []):
                try:
                    databases.delete_document(
                        database_id=DATABASE_ID,
                        collection_id=collection_id,
                        document_id=doc['$id']
                    )
                    deleted_count += 1
                    if deleted_count % 100 == 0:  # Log progress every 100 documents
                        logger.info(f"Deleted {deleted_count} documents so far...")
                except Exception as e:
                    logger.error(f"Error deleting document {doc['$id']}: {e}")
            
            # Don't need to increment page because we're always getting the first page
            # as documents are being deleted
            
        logger.info(f"Successfully deleted {deleted_count} documents from collection {collection_id}")
        return deleted_count
    
    except Exception as e:
        logger.error(f"Error while attempting to delete all documents: {e}")
        raise e


