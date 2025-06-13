# code_query.py
# pip install --upgrade --user google-cloud-aiplatform google-genai vertexai
from google import genai

import argparse
import os
from google.genai.types import Tool, Retrieval, VertexRagStore, GenerateContentConfig

import vertexai
from vertexai import rag # For RAG specific operations
import traceback
import logging
import asyncio # Added for MCP tool

import uvicorn

from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS
from mcp.server.sse import SseServerTransport

# from mcp.server.sse import SseServerTransport # Not used in this refactoring
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route, Mount
# Create an MCP server instance with an identifier ("wiki")

mcp = FastMCP("codebase_natural_language_query")

# --- Default Configurations (can be overridden by args) ---
DEFAULT_PROJECT_ID = os.getenv("GCP_PROJECT")
DEFAULT_LOCATION = os.getenv("GCP_REGION")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "gemini-1.5-flash-001")
DEFAULT_SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 5)) # Ensure int
DEFAULT_VECTOR_DISTANCE_THRESHOLD = float(os.getenv("VECTOR_DISTANCE_THRESHOLD", 0.5)) # Ensure float
DEFAULT_MCP_PORT = int(os.getenv("MCP_PORT", 8001)) # Ensure int

# --- Global variables for MCP mode resources ---
# These will be initialized if --run-mcp is used
mcp_genai_client = None
mcp_rag_corpus_resource_name = None
# args will be parsed in __main__ and accessible globally in this module
args = None 

def find_rag_corpus_by_display_name(project_id: str, location: str, display_name: str):
    """Finds an existing RAG corpus by its display name."""
    print(f"Searching for RAG Corpus with display name: '{display_name}'...")
    try:
        # vertexai.init() called by the main execution flow or MCP initialization
        corpora_list = vertexai.rag.list_corpora()
        found_corpus = None
        for corpus_item in corpora_list:
            if corpus_item.display_name == display_name:
                print(f"Found RAG Corpus: {corpus_item.name} (Display Name: {corpus_item.display_name})")
                found_corpus = corpus_item
                break
        
        if not found_corpus:
            print(f"Error: RAG Corpus with display name '{display_name}' not found in project '{project_id}' and location '{location}'.")
            print("Available RAG Corpora:")
            if corpora_list:
                for corpus_item in corpora_list:
                    print(f"  - Display Name: {corpus_item.display_name}, Resource Name: {corpus_item.name}")
            else:
                print("  No RAG Corpora found.")
        return found_corpus
    except Exception as e:
        print(f"Error during RAG corpus search: {e}")
        traceback.print_exc() # Print traceback for better debugging
        raise

def perform_rag_retrieval(
    corpus_resource_name: str,
    query_text: str,
    top_k: int,
    distance_threshold: float
) -> list[dict]:
    """
    Performs a retrieval query against the RAG corpus using vertexai.rag.
    Returns a list of context dictionaries or an empty list if error/no results.
    """
    print(f"Performing RAG retrieval for query: '{query_text}' on corpus: {corpus_resource_name}")
    try:
        rag_retrieval_config_obj = rag.RagRetrievalConfig(
            top_k=top_k,
            filter=rag.Filter(vector_distance_threshold=distance_threshold)
        )

        response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=corpus_resource_name,
                )
            ],
            text=query_text,
            rag_retrieval_config=rag_retrieval_config_obj
        )

        results = []
        if response.contexts:
            for ctx in response.contexts:
                source_name_val = ""
                if ctx.document_metadata and isinstance(ctx.document_metadata, dict):
                    source_name_val = ctx.document_metadata.get("title", "") or \
                                      ctx.document_metadata.get("original_filename", "") or \
                                      ctx.document_metadata.get("source_uri", "")
                elif hasattr(ctx, 'source_uri'):
                     source_name_val = ctx.source_uri

                result_item = {
                    "source_uri": ctx.source_uri if hasattr(ctx, "source_uri") else "",
                    "source_name": source_name_val,
                    "text": ctx.text if hasattr(ctx, "text") else "",
                    "distance": ctx.distance if hasattr(ctx, "distance") else 0.0,
                }
                results.append(result_item)
        
        if not results:
            print(f"No results found in corpus for query: '{query_text}'")
            return []

        print(f"Retrieved {len(results)} contexts from RAG.")
        return results

    except Exception as e:
        error_msg = f"Error during RAG retrieval: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return []

def _get_response_for_query(
    client: genai.Client,
    rag_corpus_resource_name: str,
    model_id: str,
    similarity_top_k: int,
    vector_distance_threshold: float,
    user_query_text: str
) -> str:
    """
    Core logic to get an LLM response augmented by RAG for a single query.
    This function preserves the original genai/vertexai interaction patterns.
    """
    augmented_prompt = (
        f"User Question: {user_query_text}\n\n"
        "Based on the provided context (if any) and your general knowledge, "
        "answer the user's question about the codebase. If no specific context was found, answer based on your general knowledge."
    )
    
    # This print statement is for console feedback, similar to original behavior
    print(f"Generating LLM response for query: '{user_query_text}'...")
    try:
        rag_retrieval_tool = Tool(
            retrieval=Retrieval(
                vertex_rag_store=VertexRagStore(
                    rag_corpora=[rag_corpus_resource_name],
                    similarity_top_k=similarity_top_k,
                    vector_distance_threshold=vector_distance_threshold,
                )
            )
        )

        response = client.models.generate_content(
                        model=model_id,
                        contents=augmented_prompt,
                        config=GenerateContentConfig(tools=[rag_retrieval_tool]),
        )

        response_text = ""
        try:
            response_text = response.text
        except ValueError: 
            if response.candidates and response.candidates[0].content.parts:
                response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, "text"))
            else:
                # This error will be caught by the caller
                raise ValueError(f"Could not extract text from response. Finish Reason: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
        
        return response_text

    except Exception as e:
        # Log details for debugging, then re-raise for caller to handle
        # (e.g., print in CLI, return McpError in MCP tool)
        print(f"Error during LLM generation for query '{user_query_text}': {e}")
        traceback.print_exc()
        raise

def run_query_interface(
    project_id: str,
    location: str,
    corpus_display_name: str,
    model_id: str,
    similarity_top_k: int,
    vector_distance_threshold: float
):
    print("Initializing Vertex AI for Query Interface...")
    try:
        vertexai.init(project=project_id, location=location)
        client = genai.Client(vertexai=True, project=project_id, location=location)
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        traceback.print_exc()
        return

    try:
        target_corpus = find_rag_corpus_by_display_name(project_id, location, corpus_display_name)
    except Exception: # Error already printed by find_rag_corpus_by_display_name
        return 

    if not target_corpus:
        return # Error already printed
    rag_corpus_resource_name = target_corpus.name

    print("\n--- RAG Query Interface ---")
    print(f"Using LLM: {model_id}")
    print(f"Querying RAG Corpus: {corpus_display_name} ({rag_corpus_resource_name})")
    print(f"Retrieval params: top_k={similarity_top_k}, distance_threshold={vector_distance_threshold}")
    print("Type your questions below. Type 'quit' or 'exit' to end.")

    while True:
        user_query_text = input("\nAsk a question: ")
        if user_query_text.lower() in ["quit", "exit"]:
            print("Exiting query interface.")
            break
        if not user_query_text.strip():
            continue

        try:
            response_str = _get_response_for_query(
                client=client,
                rag_corpus_resource_name=rag_corpus_resource_name,
                model_id=model_id,
                similarity_top_k=similarity_top_k,
                vector_distance_threshold=vector_distance_threshold,
                user_query_text=user_query_text
            )
            print("\nResponse:")
            print(response_str)

        except Exception as e:
            # _get_response_for_query already prints detailed error and traceback
            print(f"Failed to get response for query: '{user_query_text}'. See details above.")
        print("-" * 30)

def list_existing_rag_corpora_action(project_id: str, location: str):
    if not project_id or not location:
        print("Error: Project ID and Location must be provided to list corpora.")
        return
    print(f"Listing RAG Corpora in project '{project_id}', location '{location}'...")
    try:
        vertexai.init(project=project_id, location=location) # Initialize for this action
        corpora_list = vertexai.rag.list_corpora()
        if corpora_list:
            print("Available RAG Corpora:")
            for corpus_item in corpora_list: 
                print(f"  - Display Name: {corpus_item.display_name}")
                print(f"    Resource Name: {corpus_item.name}")
                print(f"    Description: {corpus_item.description if hasattr(corpus_item, 'description') else 'N/A'}")
                embed_model_info = "N/A"
                if hasattr(corpus_item, 'rag_embedding_model_config') and corpus_item.rag_embedding_model_config:
                    if hasattr(corpus_item.rag_embedding_model_config, 'vertex_prediction_endpoint') and \
                       corpus_item.rag_embedding_model_config.vertex_prediction_endpoint and \
                       hasattr(corpus_item.rag_embedding_model_config.vertex_prediction_endpoint, 'publisher_model') and \
                       corpus_item.rag_embedding_model_config.vertex_prediction_endpoint.publisher_model:
                        embed_model_info = corpus_item.rag_embedding_model_config.vertex_prediction_endpoint.publisher_model
                    elif hasattr(corpus_item.rag_embedding_model_config, 'custom_embedding_model') and \
                         corpus_item.rag_embedding_model_config.custom_embedding_model:
                        embed_model_info = f"Custom: {corpus_item.rag_embedding_model_config.custom_embedding_model}"
                print(f"    Embedding Model: {embed_model_info}")
                print("-" * 20)
        else:
            print("No RAG Corpora found in this project/location.")
    except Exception as e:
        print(f"Error listing RAG corpora: {e}")
        traceback.print_exc()

# --- MCP Tool and Server Functions ---

def initialize_mcp_resources(project_id: str, location: str, corpus_display_name: str, model_id: str):
    """Initializes resources needed for MCP mode (client, corpus)."""
    global mcp_genai_client, mcp_rag_corpus_resource_name, args
    
    if mcp_genai_client and mcp_rag_corpus_resource_name:
        logging.info("MCP resources already initialized.")
        return True

    logging.info("Initializing Vertex AI and GenAI Client for MCP...")
    try:
        vertexai.init(project=project_id, location=location)
        mcp_genai_client = genai.Client(vertexai=True, project=project_id, location=location)
    except Exception as e:
        logging.error(f"MCP: Error initializing Vertex AI or GenAI Client: {e}", exc_info=True)
        mcp_genai_client = None
        return False

    logging.info(f"MCP: Finding RAG corpus '{corpus_display_name}' for model '{model_id}'...")
    try:
        target_corpus = find_rag_corpus_by_display_name(project_id, location, corpus_display_name)
        if not target_corpus:
            # find_rag_corpus_by_display_name already prints detailed error
            logging.error(f"MCP: RAG Corpus '{corpus_display_name}' not found. Cannot start tool.")
            mcp_rag_corpus_resource_name = None
            return False
        mcp_rag_corpus_resource_name = target_corpus.name
        logging.info(f"MCP: Using RAG Corpus: {corpus_display_name} ({mcp_rag_corpus_resource_name})")
        return True
    except Exception as e: # Catch exception from find_rag_corpus_by_display_name
        logging.error(f"MCP: Error finding RAG corpus '{corpus_display_name}': {e}", exc_info=True)
        mcp_rag_corpus_resource_name = None
        return False

@mcp.tool()
async def query_codebase_llm(query: str) -> str:
    """
    Queries the codebase using a natural language.

    Receive questions about functions or the codebase in general and respond in natural 
    language.

    It is not possible to ask about lists, for example "give me a list of all the programs", 
    the result will be incomplete in that case. Same for functions, classes, variables, etc.

    It is useful to understand the codebase.
    
    :param query: The natural language query about the codebase.
    :return: The answer as a string.
    """
    global args # Access global args for model_id, top_k, distance_threshold
    global mcp_genai_client, mcp_rag_corpus_resource_name # Access initialized resources

    if not mcp_genai_client or not mcp_rag_corpus_resource_name:
        logging.error("MCP Tool 'query_codebase_llm': GenAI client or RAG corpus not initialized. Check server startup logs.")
        raise McpError(
            error_code=INTERNAL_ERROR,
            message="RAG/LLM resources not available. Server configuration issue.",
            data=ErrorData(error_type="ConfigurationError", details="Essential resources for the tool are not initialized.")
        )

    # Ensure args is available (should be parsed by __main__)
    if args is None:
        logging.error("MCP Tool 'query_codebase_llm': Configuration (args) not available.")
        raise McpError(
            error_code=INTERNAL_ERROR,
            message="Server configuration not loaded.",
            data=ErrorData(error_type="ConfigurationError", details="Command-line arguments not parsed or accessible.")
        )
    
    # All necessary parameters (model_id, top_k, distance_threshold) are from 'args'
    # corpus_display_name was used during initialization to find mcp_rag_corpus_resource_name
    
    try:
        logging.info(f"MCP Tool 'query_codebase_llm' received query: '{query}'")
        # Run the synchronous _get_response_for_query in a separate thread
        # to avoid blocking the asyncio event loop.
        response_str = await asyncio.to_thread(
            _get_response_for_query,
            client=mcp_genai_client,
            rag_corpus_resource_name=mcp_rag_corpus_resource_name,
            model_id=args.model_id,
            similarity_top_k=args.top_k,
            vector_distance_threshold=args.distance_threshold,
            user_query_text=query
        )
        return response_str
    except ValueError as ve: # Specific error from _get_response_for_query for text extraction
        logging.error(f"MCP Tool: Value error during LLM response processing for query '{query}': {ve}", exc_info=True)
        raise McpError(
            error_code=INTERNAL_ERROR,
            message=f"Error processing LLM response: {str(ve)}",
            data=ErrorData(error_type="LLMResponseError", details=str(ve))
        )
    except Exception as e:
        # Catch-all for other errors from _get_response_for_query or other unexpected issues
        logging.error(f"MCP Tool: Exception during LLM generation for query '{query}': {e}", exc_info=True)
        raise McpError(
            error_code=INTERNAL_ERROR,
            message=f"Failed to generate LLM response: {str(e)}",
            data=ErrorData(error_type="LLMGenerationError", details=str(e))
        )

def run_mcp_server_command():
    global args, mcp # args for port, mcp for the app instance
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        _server = mcp._mcp_server
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (reader, writer):
            await _server.run(reader, writer, _server.create_initialization_options())

    app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )
    """Runs the Uvicorn server. Uses global 'args' for port and 'mcp' for app."""
    # Yaml file is not used in this script according to the provided code context.
    # If it were, args.yaml_file would be the reference.
    print(f"Starting MCP server on http://0.0.0.0:{args.port}...")
    logging.info(f"MCP server starting on port {args.port}. Configured with corpus '{args.corpus_display_name}', model '{args.model_id}'.")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description="Query a Vertex AI RAG Corpus with Gemini, using direct retrieval or via MCP server."
    )
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--query", action="store_true", help="Run the interactive query interface.")
    action_group.add_argument("--list-corpora", action="store_true", help="List available RAG corpora.")
    action_group.add_argument("--run-mcp", action="store_true", help="Run MCP server to expose query functionality as a tool.")

    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID, help=f"GCP Project ID. Env: GCP_PROJECT_ID. Default: {DEFAULT_PROJECT_ID or '(env var)'}")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help=f"GCP Region. Env: GCP_REGION. Default: {DEFAULT_LOCATION or '(env var)'}")
    parser.add_argument("--corpus-display-name", help="Display name of the RAG Corpus. Required for --query and --run-mcp.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help=f"LLM model ID for generation. Default: {DEFAULT_MODEL_ID}")
    parser.add_argument("--top-k", type=int, default=DEFAULT_SIMILARITY_TOP_K, help=f"Number of top documents to retrieve. Default: {DEFAULT_SIMILARITY_TOP_K}")
    parser.add_argument("--distance-threshold", type=float, default=DEFAULT_VECTOR_DISTANCE_THRESHOLD, help=f"Max vector distance for retrieval (lower is stricter). Default: {DEFAULT_VECTOR_DISTANCE_THRESHOLD}")
    parser.add_argument("--port", type=int, default=DEFAULT_MCP_PORT, help=f"Port for the MCP server. Default: {DEFAULT_MCP_PORT}")

    args = parser.parse_args() # args is now a global variable in this module

    project_id_to_use = args.project_id
    location_to_use = args.location

    if not project_id_to_use: 
        logging.error("Error: GCP Project ID not set. Use --project-id or GCP_PROJECT_ID env var.")
        exit(1)
    if not location_to_use: 
        logging.error("Error: GCP Location/Region not set. Use --location or GCP_REGION env var.")
        exit(1)

    # --corpus-display-name is required for both --query and --run-mcp modes
    if args.query or args.run_mcp:
        if not args.corpus_display_name:
            parser.error("--corpus-display-name is required when using --query or --run-mcp.")

    if args.list_corpora:
        list_existing_rag_corpora_action(project_id_to_use, location_to_use)
    elif args.run_mcp:
        logging.info("MCP mode selected. Initializing resources...")
        if not initialize_mcp_resources(
            project_id=project_id_to_use,
            location=location_to_use,
            corpus_display_name=args.corpus_display_name,
            model_id=args.model_id # Pass model_id for logging/potential future use in init
        ):
            logging.error("Failed to initialize critical resources for MCP server. Exiting.")
            exit(1)
        # If initialization is successful, mcp_genai_client and mcp_rag_corpus_resource_name are set.
        # The MCP tool 'query_codebase_llm' will use these and other 'args'.
        run_mcp_server_command()
    elif args.query:
        run_query_interface(
            project_id=project_id_to_use,
            location=location_to_use,
            corpus_display_name=args.corpus_display_name,
            model_id=args.model_id,
            similarity_top_k=args.top_k,
            vector_distance_threshold=args.distance_threshold
        )