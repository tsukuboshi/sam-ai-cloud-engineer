import datetime
import logging
import os
import re
from typing import Any, Dict, List, Tuple

import boto3
import botocore

logger = logging.getLogger()

s3 = boto3.client("s3")
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.environ["BEDROCK_REGION"],
    config=botocore.config.Config(read_timeout=120),
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

        system_prompt = create_system_prompt()

        raw_yaml = generate_yaml(system_prompt, tmp_image_path)

        status = "notvalidated"
        cfn_res = ""
        max_review_count = int(os.environ["MAX_REVIEW_COUNT"])

        target_yaml = raw_yaml

        for review_count in range(max_review_count):
            logger.info(f"Validation Status: {status}")
            if status == "normally":
                break
            else:
                reviewed_yaml = review_yaml(
                    system_prompt, tmp_image_path, target_yaml, cfn_res
                )
                status, cfn_res = cfn_validate(reviewed_yaml)
                target_yaml = reviewed_yaml

        template_name = f"{input_diagram_name}_{current_time}_{status}.yaml"

        tmp_yaml = f"/tmp/{template_name}"

        with open(tmp_yaml, "w") as file:
            file.write(target_yaml)

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


# システムプロンプト生成関数
def create_system_prompt() -> str:
    max_token = int(os.environ["MAX_TOKEN"])
    eighty_percent_token = int(max_token / 10 * 8)

    system_prompt = f"""
    \n回答は以下の条件全てを満たすようにしてください:
    \n- 必ず回答で出力するCloudFormationテンプレート(yaml形式)の先頭は"```yaml"、末尾は"```"とする。
    \n- 必要に応じて補足を付与したい場合は、回答で出力するCloudFormationテンプレート内に#を付けてコメントとして記載する。
    \n- もし回答が{eighty_percent_token}トークンを超えたら、{max_token}トークンに達するまでに一旦回答を分割し、ユーザーが「続き」と入力したら続きの回答を作成する。
    """

    logger.info("System Prompt: %s", system_prompt)
    return system_prompt


# YAMLテンプレート生成関数
def generate_yaml(system_prompt: str, tmp_image_path: str) -> Any:

    generate_model_id = os.environ["GENERATE_MODEL_ID"]
    logger.info("Generate Model ID: %s", generate_model_id)

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

    first_generate_text = f"""
    \n\nHuman:
    \n入力されたAWS構成図に基づき、その構成をデプロイするためのCloudFormationテンプレート(YAML形式)を作成してください。
    \nテンプレートは、以下の条件全てを満たすようにしてください:
    \n- 構成図に記載されている全リソースの設定を必ず含める
    \n- 構成図に基づきリソース間の依存関係や参照を適切に定義する
    \n- 構成図に記載されていないリソースは最低限必要なものを除いて追加しない
    \n- 以下の<リソースタイプリスト>に存在しないリソースタイプは、回答で出力するCloudFormationテンプレートのリソースタイプとして使用してはいけない。
    \n\n<リソースタイプリスト>
    \n{type_list}
    """

    messages: List[Dict[str, Any]] = []

    try:
        row_yaml = output_yaml(
            generate_model_id,
            messages,
            first_generate_text,
            system_prompt,
            tmp_image_path,
        )

        return row_yaml

    except botocore.exceptions.ClientError as e:
        logger.error("Bedrock Request Error: %s", e)
        raise e


# YAMLテンプレートレビュー関数
def review_yaml(
    system_prompt: str, tmp_image_path: str, yaml_content: str, cfn_err: Any
) -> Any:
    review_model_id = os.environ["REVIEW_MODEL_ID"]
    logger.info("Review Model ID: %s", review_model_id)

    first_review_text = f"""
    \n\nHuman:
    \n入力されたAWS構成図、提示されたCloudFormationテンプレートを確認し、構成図に記載されている全てのリソースがテンプレートによって適切に作成できるかどうか確認してください。
    \nもしテンプレートに問題または不足がない場合は、提示されたテンプレートをそのまま全て出力してください。
    \nもしテンプレートに問題または不足がある場合は、該当の箇所のみ修正及び追記した上で、更新したテンプレートを全て出力してください。
    \nもしエラーメッセージが何かしら追加で提示されている場合は、エラーを解消できるように更新したテンプレートを全て出力してください。
    \n\n<CloudFormationテンプレート>
    \n{yaml_content}
    \n\n<エラーメッセージ>
    \n{cfn_err}
    """

    messages: List[Dict[str, Any]] = []

    try:
        modified_yaml = output_yaml(
            review_model_id,
            messages,
            first_review_text,
            system_prompt,
            tmp_image_path,
        )
        return modified_yaml

    except botocore.exceptions.ClientError as e:
        logger.error("Bedrock Request Error: %s", e)
        raise e


# YAML出力関数
def output_yaml(
    model_id: str,
    messages: List[Dict[str, Any]],
    first_content_text: str,
    system_prompt: str | None = None,
    tmp_image_path: str | None = None,
) -> Any:

    logger.info("First Request: %s", first_content_text)

    first_res_message = request_bedrock(
        model_id,
        messages,
        first_content_text,
        system_prompt,
        tmp_image_path,
    )

    first_row_content = first_res_message["content"][0]["text"]
    logger.info("First Response: %s", first_row_content)

    first_yaml_content = format_yaml(first_row_content)

    first_yaml_length = len(first_yaml_content)
    eighty_percent_first_yaml_length = int(first_yaml_length * 0.8)

    messages.append(first_res_message)

    yaml_content = first_yaml_content

    next_content_text = "続き"
    max_yaml_count = int(os.environ["MAX_YAML_COUNT"])

    for yaml_count in range(max_yaml_count):
        next_res_message = request_bedrock(
            model_id, messages, next_content_text, system_prompt
        )

        next_row_content = next_res_message["content"][0]["text"]
        logger.info(f"Next Response {yaml_count}: %s", next_row_content)
        next_yaml_content = format_yaml(next_row_content)

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


# Bedrockリクエスト関数
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
        return response_text
    else:
        return ""


# CloudFormationバリデーション関数
def cfn_validate(yaml_content: str) -> Tuple[Any, str]:
    try:
        cfn_res = cfn.validate_template(
            TemplateBody=yaml_content,
        )
        status = "normally"
        logger.info(f"CloudFormation is {status}: %s", cfn_res)
        return status, cfn_res
    except botocore.exceptions.ClientError as cfn_err:
        status = "error"
        logger.warning(f"CloudFormation is {status}: %s", cfn_err)
        return status, cfn_err


# テンプレートアップロード関数
def template_upload(file_name: str, tmp_file: str) -> None:
    output_s3_bucket = os.environ["OUTPUT_BUCKET"]
    try:
        s3.upload_file(tmp_file, output_s3_bucket, file_name)
        logger.info("Uploaded file: %s", file_name)
    except botocore.exceptions.ClientError as e:
        logger.error("Bucket Upload Error: %s", e)
        raise e
