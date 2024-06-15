import datetime
import logging
import os
import re
from typing import Any, Dict, List

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

        yaml_content = generate_yaml(tmp_image_path)

        template_name = cfn_validate(yaml_content, input_diagram_name, current_time)
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


# YAMLテンプレート内容生成関数
def generate_yaml(tmp_image_path: str) -> Any:
    model_id = os.environ["MODEL_ID"]
    logger.info("Model ID: %s", model_id)

    max_token = int(os.environ["MAX_TOKEN"])
    ninety_percent_token = int(max_token / 10 * 9)

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

    system_prompt = f"""
    \n回答は以下の条件全てを満たすようにしてください:
    \n- 必ず回答で出力するCloudFormationテンプレート(yaml形式)の先頭は"```yaml"、末尾は"```"とする。
    \n- 必要に応じて補足を付与したい場合は、回答で出力するCloudFormationテンプレート内に#を付けてコメントとして記載する。
    \n- もし回答が{ninety_percent_token}トークンを超えたら、{max_token}トークンに達するまでに一旦回答を分割し、ユーザーが「続き」と入力したら続きの回答を作成する。
    \n- 回答で出力するCloudFormationテンプレートには、以下のリソースタイプ以外のリソースタイプは使用しない。
    \n\n<リソースタイプ>
    \n{type_list}
    """

    logger.info("System Prompt: %s", system_prompt)

    first_content_text = f"""
    \n\nHuman:
    \n入力されたAWS構成図の詳細情報に基づき、その構成をデプロイするためのCloudFormationテンプレート(YAML形式)を作成してください。
    \nテンプレートは、以下の条件全てを満たすようにしてください:
    \n- 構成図に記載されている全リソースの設定を必ず含める
    \n- 構成図に基づきリソース間の依存関係や参照を適切に定義する
    \n- 構成図に記載されていないリソースは最低限必要なものを除いて追加しない
    """

    logger.info("First Message: %s", first_content_text)

    messages: List[Dict[str, Any]] = []

    try:
        first_res_message = request_bedrock(
            model_id,
            messages,
            first_content_text,
            system_prompt,
            tmp_image_path,
        )

        first_row_content = first_res_message["content"][0]["text"]
        first_yaml_content = format_yaml(first_row_content)

        logger.info(f"First YAML: %s", first_yaml_content)

        first_yaml_length = len(first_yaml_content)
        eighty_percent_first_yaml_length = int(first_yaml_length * 0.8)

        messages.append(first_res_message)

        yaml_content = first_yaml_content

        max_count = int(os.environ["MAX_COUNT"])

        next_content_text = "続き"

        for count in range(max_count):
            next_res_message = request_bedrock(
                model_id, messages, next_content_text, system_prompt
            )

            next_row_content = next_res_message["content"][0]["text"]
            next_yaml_content = format_yaml(next_row_content)

            logger.info(f"Next YAML {count}: %s", next_yaml_content)
            yaml_content += "\n" + next_yaml_content

            next_yaml_length = len(next_yaml_content)

            if next_yaml_length > eighty_percent_first_yaml_length:
                logger.info(
                    f"Next YAML Length {next_yaml_length} is over 80% of First YAML Length {eighty_percent_first_yaml_length}, continuing..."
                )
                messages.append(next_res_message)
            else:
                logger.info(
                    f"Next YAML Length {next_yaml_length} is under 80% of First YAML Length {eighty_percent_first_yaml_length}, breaking..."
                )
                break

        logger.info("Combined YAML: %s", yaml_content)

        return yaml_content

    except botocore.exceptions.ClientError as e:
        logger.error("Bedrock Request Error: %s", e)
        raise e


def request_bedrock(
    model_id: str,
    messages: List[Dict[str, Any]],
    content_text: str,
    system_prompt: str | None = None,
    image_path: str | None = None,
) -> Any:
    if image_path:
        with open(image_path, "rb") as image_file:
            content_image = image_file.read()
        messsage = {
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
    else:
        messsage = {
            "role": "user",
            "content": [
                {"text": content_text},
            ],
        }

    messages.append(messsage)

    if system_prompt:
        system = [
            {
                "text": system_prompt,
            }
        ]
    else:
        system = []

    inference_config = {
        "maxTokens": int(os.environ["MAX_TOKEN"]),
        "temperature": 0,
    }

    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig=inference_config,
    )

    response_message = response["output"]["message"]

    return response_message


# YAMLフォーマット関数
def format_yaml(row_content: str) -> str:
    match = re.search(r"```yaml\n(.*?)\n```", row_content, re.DOTALL)
    if match:
        response_text = match.group(1)
        logger.info("Template: %s", response_text)
        return response_text
    else:
        return ""


# CloudFormationバリデーション関数
def cfn_validate(yaml_content: str, input_diagram_name: str, current_time: str) -> str:
    try:
        cfn_res = cfn.validate_template(
            TemplateBody=yaml_content,
        )
        logger.info("CloudFormation: %s", cfn_res)
        file_name = f"{input_diagram_name}_{current_time}_normally.yaml"
        return file_name
    except botocore.exceptions.ClientError as e:
        logger.warning("CloudFormation Validation Error: %s", e)
        file_name = f"{input_diagram_name}_{current_time}_error.yaml"
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
