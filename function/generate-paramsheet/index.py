import csv
import logging
import os
import re
from typing import Any, Dict

import boto3
import botocore

logger = logging.getLogger()

s3 = boto3.client("s3")
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime", region_name=os.environ["BEDROCK_REGION"]
)


# ハンドラー関数
def lambda_handler(event: Dict[Any, Any], context: Any) -> Dict[str, Any]:
    try:
        bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
        object_key = event["Records"][0]["s3"]["object"]["key"]

        tmp_csv = f"/tmp/{object_key}.csv"
        template_download(bucket_name, object_key, tmp_csv)

        row_content = request_bedrock(tmp_csv)
        csv_content = format_csv(row_content)

        paramsheet_name = csv_validate(csv_content, object_key)

        with open(tmp_csv, "wt") as file:
            file.write(csv_content)

        paramsheet_upload(paramsheet_name, tmp_csv)

        return {"statusCode": 200, "body": "Processing completed successfully"}
    except Exception as e:
        logger.error("Unhandled exception: %s", str(e))
        return {"statusCode": 500, "body": "Internal Server Error"}


# テンプレートダウンロード関数
def template_download(bucket_name: str, file_name: str, tmp_file_path: str) -> None:
    try:
        s3.download_file(bucket_name, file_name, tmp_file_path)
        logger.info("Downloaded template: %s", file_name)
    except botocore.exceptions.ClientError as e:
        logger.error("Bucket Download Error: %s", e)
        raise e


# Bedrockへのメッセージリクエスト関数
def request_bedrock(tmp_template_path: str) -> Any:
    model_id = os.environ["MODEL_ID"]
    prompt_path = os.environ["PROMPT_PATH"]

    with open(tmp_template_path, "r") as file:
        yaml_content = file.read()

    system_prompt = """
    \n必ず回答はcsvファイルの内容とし、回答の先頭は"```csv"、末尾は"```"としてください。
    \n補足を付与したい場合は、csvファイルにコメントとして記載してください。
    """

    with open(prompt_path, "rt") as csv_file:

        complement_prompt = csv_file.read()
    logger.info("Complement Prompt: %s", complement_prompt)

    content_text = f"""
    \n\nHuman:
    \n<質問>
    \n提示されたサンプルパラメータシートの形式を参考にしながら、提示されたCloudFormationテンプレートを反映するパラメータシート(CSV形式)を作成してください。
    \nパラメータシートには、以下の条件全てを満たすようにしてください:
    \n- 必要なすべてのリソースとそれらの設定を含める
    \n- テンプレートに基づきリソース間の依存関係や参照を適切に記載する
    \n\n<サンプルパラメータシート>
    ```csv
    {complement_prompt}
    ```
    \n\n<CloudFormationテンプレート>
    \n```yaml
    \n{yaml_content}
    \n```
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"text": content_text},
            ],
        }
    ]

    system = [
        {
            "text": system_prompt,
        }
    ]

    inference_config = {
        "maxTokens": 4096,
        "temperature": 0,
    }

    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            messages=messages,
            system=system,
            inferenceConfig=inference_config,
        )

        row_content = response["output"]["message"]["content"][0]["text"]
        logger.info("Content: %s", row_content)

        return row_content
    except botocore.exceptions.ClientError as e:
        logger.error("Bedrock Request Error: %s", e)
        raise e


# CSVフォーマット関数
def format_csv(row_content: str) -> str:
    match = re.search(r"```csv\n(.*?)\n```", row_content, re.DOTALL)
    if match:
        response_text = match.group(1)
        logger.info("Paramsheet: %s", response_text)
        return response_text
    else:
        logger.error("CSV Format Error: %s", row_content)
        raise Exception("CSV Format Error")


# CSVバリデーション関数
def csv_validate(csv_content: str, object_key: str) -> str:
    try:
        csv_reader = csv.reader(csv_content.splitlines())
        for row in csv_reader:
            pass
        logger.info("CSV Validation: Success")
        csv_file_name = f"{object_key}_normally.csv"
        return csv_file_name
    except csv.Error:
        logger.warning("CSV Validation Error")
        csv_file_name = f"{object_key}_error.csv"
        return csv_file_name


# パラメータシートアップロード関数
def paramsheet_upload(file_name: str, tmp_file: str) -> None:
    output_s3_bucket = os.environ["OUTPUT_BUCKET"]
    try:
        s3.upload_file(tmp_file, output_s3_bucket, file_name)
        logger.info("Uploaded file: %s", file_name)
    except botocore.exceptions.ClientError as e:
        logger.error("Bucket Upload Error: %s", e)
        raise e
