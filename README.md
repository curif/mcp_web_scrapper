# mcp_web_scrapper


## GCP permission
You need to be logged to gcloud to work with `ingest.py`
You will get this error if RAG can't access the bucket

```json
{
  "error": {
    "code": 403,
    "message": "service-xx@gcp-sa-vertex-rag.iam.gserviceaccount.com does not have storage.objects.create access to the Google Cloud Storage object. Permission 'storage.objects.create' denied on resource (or it may not exist).",
    "errors": [
      {
        "message": "service-10622xx32989396@gcp-sa-vertex-rag.iam.gserviceaccount.com does not have storage.objects.create access to the Google Cloud Storage object. Permission 'storage.objects.create' denied on resource (or it may not exist).",
        "domain": "global",
        "reason": "forbidden"
      }
    ]
  }
}
```

 ```bash
 gcloud storage buckets add-iam-policy-binding gs://<bucket name> \
 --member="serviceAccount:service-xx@gcp-sa-vertex-rag.iam.gserviceaccount.com" \
 --role="roles/storage.objectAdmin"
```