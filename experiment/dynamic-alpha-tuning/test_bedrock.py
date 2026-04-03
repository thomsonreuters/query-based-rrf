import json
import asyncio
import boto3


MODEL_ID = "qwen.qwen3-32b-v1:0"
AWS_PROFILE = "a204383-ml-workspace-practicallawqw7t-prod-use1"

session = boto3.Session(profile_name=AWS_PROFILE)
client = session.client("bedrock-runtime", region_name="us-east-1")

request_body = {
    "messages": [
        {"role": "user", "content": "Reply with just: hello"},
    ],
    "max_tokens": 32,
}

try:
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )
    response_body = json.loads(response["body"].read())
    print(json.dumps(response_body, indent=2))
    output_text = response_body["choices"][0]["message"]["content"]
    print("\nModel Response:", output_text)
except Exception as e:
    print(f"Error: {e}")
