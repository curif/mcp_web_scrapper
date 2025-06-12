
# Vertex AI RAG Corpus Ingestion and Querying Tool

This Python script automates the process of uploading local Markdown (`.md`) files to Google Cloud Storage (GCS), ingesting them into a Vertex AI RAG (Retrieval Augmented Generation) Corpus, and optionally querying the corpus for relevant information.

## Features

*   **Upload Local Files:** Uploads all `.md` files from a specified local directory to a designated GCS bucket.
*   **Corpus Management:**
    *   Creates a new Vertex AI RAG Corpus if one with the specified display name doesn't exist.
    *   Retrieves and uses an existing RAG Corpus if found.
*   **Document Ingestion:** Imports the uploaded Markdown files from GCS into the RAG Corpus.
    *   Configurable chunking parameters (chunk size, overlap).
    *   Saves detailed import results to a `.jsonl` file in GCS.
*   **Retrieval/Querying:** Optionally queries the populated RAG Corpus with a given text string to retrieve relevant document chunks.
*   **Configurable:** Uses environment variables for GCP project/region and command-line arguments for operational parameters.

## Prerequisites

1.  **Google Cloud Project:**
    *   A Google Cloud Project with the Vertex AI API and Cloud Storage API enabled.
    *   Billing enabled for the project.
2.  **Authentication:**
    *   The `gcloud` CLI installed and authenticated. Run `gcloud auth application-default login` to set up Application Default Credentials (ADC) for local development.
    *   The authenticated user/service account needs the following IAM roles (or equivalent custom roles) on your project:
        *   `Vertex AI User` (or `roles/aiplatform.user`) - For creating and managing RAG corpora.
        *   `Storage Object Admin` (or `roles/storage.objectAdmin`) - For uploading files to GCS and for the RAG service to write import results.
        *   `Storage Object Creator` (for uploads) and `Storage Object Viewer` (for RAG service to read) can also work if more granular permissions are desired.
3.  **Python Environment:**
    *   Python 3.8 or higher.
    *   Required Python libraries. Install them using pip:
        ```bash
        pip install google-cloud-aiplatform google-cloud-storage
        # Ensure vertexai[rag] specific dependencies are met.
        # google-cloud-aiplatform should bring in vertexai.
        # If you encounter issues with rag module, try:
        # pip install "google-cloud-aiplatform>=1.38.1" # or a later version supporting the RAG features used
        ```
4.  **Local Markdown Files:** A local directory containing the `.md` files you want to ingest.
5.  **GCS Bucket:** An existing Google Cloud Storage bucket where the script can upload files and store import results.

## Configuration

### Environment Variables

The script requires the following environment variables to be set:

*   `GCP_PROJECT`: Your Google Cloud Project ID.
*   `GCP_REGION`: The Google Cloud region where the RAG Corpus will be created (e.g., `us-central1`).

Set them in your shell before running the script:
```bash
export GCP_PROJECT="your-gcp-project-id"
export GCP_REGION="your-gcp-region"
```

### Command-Line Arguments

The script accepts the following command-line arguments:

| Argument                 | Required | Default Value                       | Description                                                                 |
| ------------------------ | -------- | ----------------------------------- | --------------------------------------------------------------------------- |
| `--local-folder`         | No       | `./your_md_files_folder`            | Path to the local directory containing `.md` files.                         |
| `--corpus-name`          | No       | `MyMarkdownDocCorpusWithRAG`        | Display name for the Vertex AI RAG Corpus.                                  |
| `--bucket-name`          | Yes      | N/A                                 | Name of the GCS bucket for uploads and import results sink.                 |
| `--embedding-model-uri`  | No       | `publishers/google/models/text-embedding-004` | Publisher URI for the embedding model to be used by the RAG Corpus.       |
| `--chunk-size`           | No       | `1024`                              | Target chunk size (in tokens) for document processing during import.        |
| `--chunk-overlap`        | No       | `200`                               | Token overlap between adjacent chunks during import.                        |
| `--query`                | No       | N/A                                 | Optional: A query string to retrieve relevant contexts from the corpus.     |
| `--top-k`                | No       | `3`                                 | Number of top relevant contexts to retrieve if a query is provided.         |

## Usage

1.  **Set Environment Variables:**
    ```bash
    export GCP_PROJECT="your-gcp-project-id"
    export GCP_REGION="us-central1" # e.g., us-central1
    ```

2.  **Prepare Local Files:**
    Ensure your Markdown files are in a local directory. For example, if your files are in `./my_documentation_files/`.

3.  **Run the Script:**

    Replace placeholders with your actual values.

    *   **To upload files and ingest them into a new/existing corpus:**
        ```bash
        python your_script_name.py \
            --local-folder ./my_documentation_files \
            --corpus-name "MyWebAppDocumentation" \
            --bucket-name "my-gcs-bucket-for-rag" \
            --embedding-model-uri "publishers/google/models/text-embedding-004"
        ```

    *   **To upload, ingest, AND query the corpus:**
        ```bash
        python your_script_name.py \
            --local-folder ./my_documentation_files \
            --corpus-name "MyWebAppDocumentation" \
            --bucket-name "my-gcs-bucket-for-rag" \
            --query "How do I reset my password?" \
            --top-k 5
        ```

    *   **To only query an existing corpus (assuming it's already populated):**
        You can provide an empty or non-existent `--local-folder` (the script will note no files found for upload). The script will still attempt to find/use the corpus by `--corpus-name` for querying.
        ```bash
        python your_script_name.py \
            --local-folder ./dummy_empty_folder \
            --corpus-name "MyWebAppDocumentation" \
            --bucket-name "my-gcs-bucket-for-rag" \
            --query "What are the API rate limits?"
        ```
### Examples
```bash
python3 ingest.py --bucket-name curif-agent-docs-adk --local-folder ./output/ --corpus-name "Agent ADK Documentation" ^C
```
## Script Workflow

1.  **Initialization:**
    *   Initializes Vertex AI SDK and Google Cloud Storage client using `GCP_PROJECT` and `GCP_REGION`.
2.  **Upload to GCS (if local files exist):**
    *   Scans the `--local-folder` for `.md` files.
    *   If files are found, they are uploaded to a timestamped sub-directory within `gs://<bucket-name>/rag_corpus_data/<corpus-name>/`.
3.  **Corpus Creation/Retrieval:**
    *   Checks if a RAG Corpus with the specified `--corpus-name` already exists in the project/region.
    *   If it exists, the script uses it.
    *   If not, a new RAG Corpus is created using the provided display name, description, and `--embedding-model-uri`.
4.  **Document Import (if files were uploaded):**
    *   If files were successfully uploaded to GCS in step 2, the `rag.import_files()` function is called.
    *   It points to the GCS directory containing the uploaded `.md` files.
    *   `--chunk-size` and `--chunk-overlap` are used for configuring document chunking.
    *   A GCS URI for `import_result_sink` is generated (e.g., `gs://<bucket-name>/rag_corpus_data/<corpus-name>/<timestamp_upload_folder>/rag_import_results/import_details_<timestamp>.jsonl`), where detailed results of the import operation will be stored.
    *   The import process is initiated. This process itself is asynchronous on the backend, but the script waits for the initial submission response.
5.  **Querying (if `--query` is provided):**
    *   If a `--query` string is given, the script waits for a short period (to allow for potential indexing completion).
    *   It then uses the `retrieve_contexts()` method on the RAG Corpus object to find the `--top-k` most relevant document chunks based on the query.
    *   The retrieved contexts (source URI, chunk ID, distance, text) are printed to the console.

## Output

*   The script prints progress and status messages to the console.
*   If querying, retrieved contexts are displayed.
*   Detailed import results are saved to a `.jsonl` file in your GCS bucket, in a subfolder like `rag_corpus_data/<corpus_name>/<timestamp_upload_folder>/rag_import_results/`.

## Troubleshooting

*   **Authentication Errors:** Ensure `gcloud auth application-default login` has been run and the correct project is selected (`gcloud config set project YOUR_PROJECT_ID`). Verify IAM permissions.
*   **API Not Enabled:** Make sure "Vertex AI API" and "Cloud Storage API" are enabled in the Google Cloud Console for your project.
*   **Bucket Permissions:** The service account used by Vertex AI for RAG operations (usually a Google-managed service agent, or the one running the script if using ADC) needs permissions to read from the GCS source bucket and write to the GCS sink location for import results.
*   **"No .md files found":** Double-check the path provided to `--local-folder` and ensure it contains Markdown files.
*   **Import Failures:** Check the `import_details_<timestamp>.jsonl` file in GCS for detailed error messages from the RAG import service. Also, check the Vertex AI RAG management UI in the Google Cloud Console for operation statuses.
*   **Module Not Found (e.g., `vertexai.rag`):** Ensure you have a recent version of `google-cloud-aiplatform`. The RAG module is relatively new.
    ```bash
    pip install --upgrade google-cloud-aiplatform
    ```
*   **Corpus Creation "Already Exists":** The script attempts to handle this by finding and reusing the existing corpus. If issues persist, verify the corpus display name.
*   **Slow Indexing:** After importing new files, it can take some time (minutes) for the documents to be fully indexed and available for retrieval. The script includes a short `time.sleep()` before querying, but for very large datasets, more robust status checking or a longer wait might be needed.

```

**Remember to:**

1.  Replace `your_script_name.py` with the actual name of your Python script in the usage examples.
2.  Review and adjust any default values or descriptions to perfectly match your script's final behavior if you've made further modifications.
3.  Ensure the "Prerequisites" section accurately reflects the library versions that work well with your script. The `vertexai.rag` module's API has seen changes, so specific versions of `google-cloud-aiplatform` might be relevant.