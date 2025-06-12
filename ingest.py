import os
import glob
import argparse
import time # For sleep
import traceback # For detailed error printing
from datetime import datetime # Corrected import for datetime.datetime
from typing import List, Optional, Any

# GCP Libraries
from google.cloud import storage
from google.cloud.storage.blob import Blob # For parsing GCS URIs for sink
import vertexai
from vertexai import rag

# --- Global Configuration (derived from env vars and args) ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION")

# RAG Configuration Defaults
DEFAULT_EMBEDDING_MODEL_PUBLISHER_URI = "publishers/google/models/text-embedding-004"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# --- Initialization ---
def init_vertex_ai(project_id: str, location: str):
    """Initializes Vertex AI if not already done."""
    # vertexai.init() can be called multiple times, but it's good practice
    # to ensure it's called before other vertexai operations.
    # This function is similar to the one implied in user's snippets.
    try:
        current_project = vertexai.preview.global_config.project_id # Check if already initialized
        current_location = vertexai.preview.global_config.location
        if current_project == project_id and current_location == location:
            print(f"Vertex AI already initialized for project '{project_id}' in location '{location}'.")
            return
    except Exception: # Not initialized or different config
        pass

    print(f"Initializing Vertex AI for project '{project_id}' in region '{location}'...")
    try:
        vertexai.init(project=project_id, location=location)
        print("Vertex AI initialized.")
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        raise # Re-raise to stop execution if init fails

def get_storage_client(project_id: str) -> storage.Client:
    """Gets a GCS storage client."""
    print(f"Initializing Storage client for project '{project_id}'...")
    try:
        storage_client = storage.Client(project=project_id)
        print("Storage client initialized.")
        return storage_client
    except Exception as e:
        print(f"Error initializing Storage client: {e}")
        raise

def upload_files_to_gcs(
    storage_client: storage.Client,
    local_folder_path: str,
    bucket_name: str,
    gcs_upload_base_prefix: str
) -> Optional[str]:
    if not bucket_name:
        print("ERROR: Bucket name was not provided to upload_files_to_gcs function.")
        return None

    bucket = storage_client.bucket(bucket_name)
    md_files = glob.glob(os.path.join(local_folder_path, "*.md"))

    if not md_files:
        print(f"No .md files found in '{local_folder_path}'.")
        return None

    timestamp_folder = datetime.now().strftime("%Y%m%d-%H%M%S")
    path_parts = [part.strip('/') for part in [gcs_upload_base_prefix, timestamp_folder] if part.strip('/')]
    full_gcs_folder_prefix = '/'.join(path_parts)

    print(f"\nUploading .md files to gs://{bucket_name}/{full_gcs_folder_prefix}/ ...")
    uploaded_count = 0
    for local_file_path in md_files:
        filename = os.path.basename(local_file_path)
        gcs_blob_name = f"{full_gcs_folder_prefix}/{filename}"
        blob = bucket.blob(gcs_blob_name)
        try:
            blob.upload_from_filename(local_file_path)
            print(f"  Uploaded '{local_file_path}' to 'gs://{bucket_name}/{gcs_blob_name}'")
            uploaded_count += 1
        except Exception as e:
            print(f"  ERROR uploading '{local_file_path}': {e}")

    if uploaded_count > 0:
        print(f"Successfully uploaded {uploaded_count} files.")
        return f"gs://{bucket_name}/{full_gcs_folder_prefix}/" # Ensure trailing slash for folder
    else:
        print("No files were successfully uploaded to GCS.")
        return None

def find_corpus_by_display_name(display_name: str) -> Optional[str]:
    """Finds an existing RAG corpus by its display name and returns its full resource name."""
    print(f"Searching for existing corpus with display name: '{display_name}'...")
    try:
        corpora_list = rag.list_corpora()
        for corpus_item in corpora_list:
            if corpus_item.display_name == display_name:
                print(f"Found existing corpus: {corpus_item.name} (Display Name: {corpus_item.display_name})")
                return corpus_item.name # Return the full resource name
        print(f"No corpus found with display name '{display_name}'.")
        return None
    except Exception as e:
        print(f"Error listing corpora: {e}")
        traceback.print_exc()
        return None

def create_or_get_rag_corpus(
    project_id: str,
    location: str,
    corpus_display_name: str,
    corpus_description: Optional[str],
    embedding_model_uri: str
) -> Optional[str]: # Returns corpus resource name (string) or None
    """
    Creates a new Vertex AI RAG Corpus or gets an existing one.
    Returns the full resource name of the corpus (e.g., projects/.../corpora/...).
    """
    init_vertex_ai(project_id, location) # Ensure Vertex AI is initialized

    existing_corpus_name = find_corpus_by_display_name(corpus_display_name)
    if existing_corpus_name:
        print(f"Reusing existing corpus: {existing_corpus_name}")
        return existing_corpus_name

    print(f"\n--- Creating New Vertex AI RAG Corpus ---")
    print(f"Display Name: {corpus_display_name}, Embedding Model URI: {embedding_model_uri}")
    actual_description = corpus_description or f"RAG Corpus for {corpus_display_name}."

    try:
        embedding_model_config = rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model=embedding_model_uri
            )
        )
        # As per user snippet, create_corpus takes backend_config
        rag_corpus_obj = rag.create_corpus(
            display_name=corpus_display_name,
            description=actual_description,
            backend_config=rag.RagVectorDbConfig(
                rag_embedding_model_config=embedding_model_config
            ),
        )
        print(f"Successfully created RAG Corpus: {rag_corpus_obj.name}, Display Name: {rag_corpus_obj.display_name}")
        return rag_corpus_obj.name # Return the full resource name
    except Exception as e:
        print(f"Error creating RAG Corpus: {e}")
        traceback.print_exc()
        # Fallback to check if it was created just before a concurrent op or if error message indicates existence
        if "ALREADY_EXISTS" in str(e).upper() or "already exists" in str(e).lower():
            print("  Creation failed, possibly because corpus already exists. Attempting to find it again.")
            return find_corpus_by_display_name(corpus_display_name)
        return None


def import_documents_to_rag_corpus(
    project_id: str,
    location: str,
    corpus_name_full: str, # Full resource name of the corpus
    gcs_folder_uri: str,    # GCS URI of the DIRECTORY containing MD files
    chunk_size: int,
    chunk_overlap: int
) -> bool:
    init_vertex_ai(project_id, location)
    storage_client = get_storage_client(project_id)

    print(f"\n--- Importing Files to Vertex AI RAG Corpus '{corpus_name_full}' ---")
    print(f"Source GCS Folder: {gcs_folder_uri}")
    print(f"Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}")

    if not gcs_folder_uri.startswith("gs://"):
        print(f"Error: GCS folder URI '{gcs_folder_uri}' must be a GCS URI.")
        return False
    
    # Ensure gcs_folder_uri ends with a slash if it's a directory
    gcs_data_root = gcs_folder_uri.rstrip('/') + '/'

    try:
        transformation_config_for_import = rag.TransformationConfig(
             chunking_config=rag.ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        )
        
        current_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Construct results sink URI based on the input GCS folder
        parsed_gcs_root_for_sink = Blob.from_string(gcs_data_root, client=storage_client)
        sink_bucket_name = parsed_gcs_root_for_sink.bucket.name
        
        # Path within the bucket where the data resides
        sink_base_prefix = parsed_gcs_root_for_sink.name # e.g., "my_rag_data/corpus_name/timestamp/"
        if sink_base_prefix and not sink_base_prefix.endswith('/'):
            sink_base_prefix += '/'
        elif not sink_base_prefix: # Handles "gs://bucket-name/"
            sink_base_prefix = ""

        results_folder_in_gcs = "rag_import_results" # Subfolder within the data's GCS path for results
        # e.g., "my_rag_data/corpus_name/timestamp/rag_import_results"
        results_sink_path_prefix = f"{sink_base_prefix.strip('/')}/{results_folder_in_gcs}".strip('/')
        
        results_sink_filename = f"import_details_{current_timestamp}.jsonl"
        results_sink_uri = f"gs://{sink_bucket_name}/{results_sink_path_prefix}/{results_sink_filename}"
        
        print(f"Detailed import results will be saved to: {results_sink_uri}")
        print(f"IMPORTANT: Ensure GCS folder 'gs://{sink_bucket_name}/{results_sink_path_prefix}/' can be written to by the service.")

        import_op_response = rag.import_files(
            corpus_name=corpus_name_full,
            paths=[gcs_data_root], # GCS directory URI
            transformation_config=transformation_config_for_import,
            import_result_sink=results_sink_uri,
            # max_embedding_requests_per_min can be set here if needed
        )

        print(f"File import call submitted.")
        imported_count = getattr(import_op_response, 'imported_rag_files_count', 0)
        failed_count = getattr(import_op_response, 'failed_rag_files_count', 0)
        skipped_count = getattr(import_op_response, 'skipped_rag_files_count', 0)

        print(f"  Summary from immediate response:")
        print(f"    Imported source paths count: {imported_count}") # Usually 1 for a directory
        print(f"    Failed source paths count: {failed_count}")
        print(f"    Skipped source paths count: {skipped_count}")

        processing_error_msg = ""
        if hasattr(import_op_response, 'processing_error') and import_op_response.processing_error:
            processing_error_msg = str(import_op_response.processing_error)
            print(f"    Processing Error reported: {processing_error_msg}")
        
        if failed_count > 0 or skipped_count > 0 or processing_error_msg:
            print(f"    One or more issues occurred during import submission. Check details at: {results_sink_uri}")
            return False # Indicate potential failure
        elif imported_count == 0 and not processing_error_msg : # No errors but nothing imported from the path
             print(f"    The import call was submitted, but the response indicates 0 source paths were imported. "
                   f"This might mean no suitable files were found in '{gcs_data_root}' or other issues. "
                   f"Check details at: {results_sink_uri}")
             return True # Technically call succeeded, but worth noting
        else: # imported_count > 0
             print("  Import call submitted successfully. File-level status will be in the GCS sink file.")
             print(f"  Refer to: {results_sink_uri}")
             return True

    except TypeError as te:
        print(f"TypeError during Vertex AI RAG file import: {te}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Error during Vertex AI RAG file import: {e}")
        traceback.print_exc()
        return False

def retrieve_from_rag_corpus(
    project_id: str,
    location: str,
    corpus_name_full: str, # Full resource name
    query_string: str,
    top_k: int = 3
) -> Optional[Any]: # Using Any for response type due to uncertainty about rag.types
    init_vertex_ai(project_id, location)
    print(f"\n--- Retrieving from RAG Corpus '{corpus_name_full}' ---")
    print(f"Query: '{query_string}', Top K: {top_k}")
    try:
        # Instantiate a rag.Corpus object to use its retrieve_contexts method
        rag_corpus_obj = rag.Corpus(corpus_name=corpus_name_full)
        
        response = rag_corpus_obj.retrieve_contexts(
            query_texts=[query_string],
            similarity_top_k=top_k
        )

        if response and response.contexts and response.contexts[0]:
            print(f"Found {len(response.contexts[0])} relevant context(s):")
            for i, context_chunk in enumerate(response.contexts[0]):
                print(f"\n--- Context Chunk {i+1} ---")
                # Attributes like source_uri, chunk_id, distance, text_chunk should be available
                print(f"  Source URI: {getattr(context_chunk, 'source_uri', 'N/A')}")
                print(f"  Chunk ID: {getattr(context_chunk, 'chunk_id', 'N/A')}")
                distance = getattr(context_chunk, 'distance', None)
                if distance is not None:
                    print(f"  Distance: {distance:.4f}")
                text_preview = getattr(context_chunk, 'text_chunk', "")
                if len(text_preview) > 300:
                    text_preview = text_preview[:300] + "..."
                print(f"  Text: \"{text_preview}\"")
            return response
        else:
            print("No relevant contexts found for the query.")
            return None
    except Exception as e:
        print(f"Error retrieving contexts from RAG corpus: {e}")
        traceback.print_exc()
        return None

def main(args):
    """Main function to orchestrate the process."""
    print("--- Starting MD File to Vertex AI RAG Corpus Ingestion & Retrieval ---")

    if not all([GCP_PROJECT_ID, GCP_REGION, args.bucket_name]):
        print("ERROR: GCP_PROJECT, GCP_REGION environment variables and --bucket-name argument must be set/provided.")
        return

    if not os.path.isdir(args.local_folder):
        print(f"ERROR: Local MD folder '{args.local_folder}' does not exist.")
        return

    try:
        # Initialize services early
        init_vertex_ai(GCP_PROJECT_ID, GCP_REGION)
        storage_client = get_storage_client(GCP_PROJECT_ID)
    except Exception as e:
        print(f"Failed to initialize Google Cloud services: {e}. Exiting.")
        return

    gcs_upload_base_prefix = f"rag_corpus_data/{args.corpus_name.lower().replace(' ', '_')}"

    gcs_uploaded_folder_uri = upload_files_to_gcs(
        storage_client,
        args.local_folder,
        args.bucket_name,
        gcs_upload_base_prefix
    )

    corpus_description = (
        f"Documentation corpus for '{args.corpus_name}'. "
        "Contains Markdown files from local folder for RAG."
    )
    # Get or create corpus, returns the corpus resource name (string)
    corpus_resource_name = create_or_get_rag_corpus(
        GCP_PROJECT_ID,
        GCP_REGION,
        args.corpus_name,
        corpus_description,
        args.embedding_model_uri
    )

    if not corpus_resource_name:
        print("Could not create or get RAG corpus. Exiting.")
        return

    import_successful_this_run = False
    if gcs_uploaded_folder_uri:
        print("Attempting to import newly uploaded files...")
        import_successful_this_run = import_documents_to_rag_corpus(
            GCP_PROJECT_ID,
            GCP_REGION,
            corpus_resource_name,
            gcs_uploaded_folder_uri,
            args.chunk_size,
            args.chunk_overlap
        )
        if import_successful_this_run:
            print("Import process for new files submitted. Indexing may take some time.")
        else:
            print("Import process for new files encountered issues or was skipped.")
    else:
        print("No new files uploaded in this run. Skipping document import for this batch.")

    if args.query:
        # Wait regardless of import_successful_this_run because indexing of *any* data takes time
        # If the corpus was just created, it needs indexing. If files were just added, they need indexing.
        # If querying an old corpus, this wait might be less critical but doesn't hurt much.
        print("Waiting for potential indexing to complete before querying...")
        time.sleep(30) # Increased wait time as indexing can be slow

        retrieve_from_rag_corpus(
            GCP_PROJECT_ID,
            GCP_REGION,
            corpus_resource_name,
            args.query,
            args.top_k
        )
    else:
        print("\nNo query provided. Skipping retrieval step.")

    print("\n--- RAG Corpus Process Complete ---")
    print(f"RAG Corpus Name (Display): {args.corpus_name}")
    print(f"RAG Corpus Resource Name: {corpus_resource_name}")
    if gcs_uploaded_folder_uri:
      print(f"Files from this run were uploaded to GCS folder: {gcs_uploaded_folder_uri}")
    print("Verify corpus and import status in Google Cloud Console (Vertex AI -> RAG Management).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload MD files, ingest into Vertex AI RAG Corpus, and optionally query.")
    parser.add_argument("--local-folder", type=str, default="./your_md_files_folder", help="Path to local folder with .md files.")
    parser.add_argument("--corpus-name", type=str, default="MyMarkdownDocCorpusWithRAG", help="Display name for RAG Corpus.")
    parser.add_argument("--bucket-name", type=str, required=True, help="GCS bucket name for uploads and import results sink.")
    parser.add_argument("--embedding-model-uri", type=str, default=DEFAULT_EMBEDDING_MODEL_PUBLISHER_URI, help="Publisher URI for the embedding model.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for RAG import.")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap for RAG import.")
    parser.add_argument("--query", type=str, help="Optional: Query string for retrieval.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of contexts to retrieve.")

    parsed_args = parser.parse_args()

    if not all([GCP_PROJECT_ID, GCP_REGION]):
        print("CRITICAL ERROR: Environment variables GCP_PROJECT and GCP_REGION must be set.")
    else:
        main(parsed_args)