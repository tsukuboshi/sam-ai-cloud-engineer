import base64
import datetime
import json
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
cfn = boto3.client("cloudformation")


# ハンドラー関数
def lambda_handler(event: Dict[Any, Any], context: Any) -> Dict[str, Any]:
    try:
        input_bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
        input_diagram_name = event["Records"][0]["s3"]["object"]["key"]

        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        tmp_image_path = f"/tmp/{current_time}.png"
        image_download(input_bucket_name, input_diagram_name, tmp_image_path)

        row_content = request_bedrock(tmp_image_path)
        yaml_content = format_yaml(row_content)

        template_name = cfn_validate(yaml_content, current_time)
        tmp_yaml = f"/tmp/{template_name}"

        with open(tmp_yaml, "w") as file:
            file.write(yaml_content)

        template_upload(template_name, tmp_yaml)

        return {"statusCode": 200, "body": "Processing completed successfully"}
    except Exception as e:
        logger.error("Unhandled exception: %s", str(e))
        return {"statusCode": 500, "body": "Internal Server Error"}


# 構成図ダウンロード関数
def image_download(bucket_name: str, file_name: str, tmp_file_path: str) -> None:
    try:
        s3.download_file(bucket_name, file_name, tmp_file_path)
        logger.info("Downloaded image: %s", file_name)
    except botocore.exceptions.ClientError as e:
        logger.error("Bucket Download Error: %s", e)
        raise e


# Bedrockへのメッセージリクエスト関数
def request_bedrock(tmp_image_path: str) -> Any:
    model_id = os.environ["MODEL_ID"]

    system_prompt = """
    \n必ず回答はyamlファイルの内容とし、回答の先頭は"```yaml"、末尾は"```"としてください。
    \n補足を付与したい場合は、yamlファイルにコメントとして記載してください。
    """

    next_token = None
    type_summaries = []
    while True:
        response = cfn.list_types(
            Visibility="PUBLIC", **({"NextToken": next_token} if next_token else {})
        )
        type_summaries.extend(response["TypeSummaries"])
        next_token = response.get("NextToken")
        if not next_token:
            break
    type_list = [
        summary["TypeName"]
        for summary in type_summaries
        if summary["TypeName"].startswith("AWS::")
    ]
    logger.info("Resource Type: %s", type_list)

    content_text = f"""
    \n\nHuman:
    \n<質問>
    \n入力されたAWS構成図の詳細情報に基づき、その構成をデプロイするためのCloudFormationテンプレート(YAML形式)を作成してください。
    \nテンプレートには、以下の条件全てを満たすようにしてください:
    \n- 必要なすべてのリソースとそれらの設定を含める
    \n- 構成図に基づきリソース間の依存関係や参照を適切に定義する
    \n- 提示されたリソースタイプ以外のリソースタイプを使用しない
    \n\n<リソースタイプ>
    \n{type_list}
    """

    with open(tmp_image_path, "rb") as image_file:
        content_image = image_file.read()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {
                            "bytes": content_image,
                        },
                    },
                },
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


# YAMLフォーマット関数
def format_yaml(row_content: str) -> str:
    match = re.search(r"```yaml\n(.*?)\n```", row_content, re.DOTALL)
    if match:
        response_text = match.group(1)
        logger.info("Template: %s", response_text)
        return response_text
    else:
        logger.error("YAML Format Error: %s", row_content)
        raise Exception("YAML Format Error")


# CloudFormationバリデーション関数
def cfn_validate(yaml_content: str, current_time: str) -> str:
    try:
        cfn_res = cfn.validate_template(
            TemplateBody=yaml_content,
        )
        logger.info("CloudFormation: %s", cfn_res)
        file_name = f"{current_time}_normally.yaml"
        return file_name
    except botocore.exceptions.ClientError as e:
        logger.warning("CloudFormation Validation Error: %s", e)
        file_name = f"{current_time}_error.yaml"
        return file_name


# テンプレートアップロード関数
def template_upload(file_name: str, tmp_file: str) -> None:
    output_s3_bucket = os.environ["OUTPUT_BUCKET"]
    try:
        s3.upload_file(tmp_file, output_s3_bucket, file_name)
        logger.info("Uploaded file: %s", file_name)
    except botocore.exceptions.ClientError as e:
        logger.error("Bucket Upload Error: %s", e)
        raise e
