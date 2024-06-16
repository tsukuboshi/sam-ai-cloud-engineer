# sam-ai-cloud-engineer

## 概要

AWS構成図の画像(png形式)Inputバケットにアップロードすると、OutputバケットにCloudFormationテンプレート(yaml形式)とパラメータシート(csv形式)が生成されるサーバレスアプリケーションをSAMで構築します。  

## 要件

- Python 3.12
- SAM CLI

## デプロイ方法

1. 事前にLambdaで使用するリージョンのBedrock Claude 3を、[モデルアクセス \- Amazon Bedrock](https://docs.aws.amazon.com/ja_jp/bedrock/latest/userguide/model-access.html#model-access-add)を参考に有効化

2. 以下コマンドで、リポジトリをクローン

```bash
git clone https://github.com/tsukuboshi/sam-ai-cloud-engineer.git
cd sam-ai-cloud-engineer
```

3. 以下コマンドで、SAMアプリをビルド

``` bash
sam build
```

4. 以下コマンドで、SAMアプリをデプロイ

``` bash
sam deploy
```

※ 以下のパラメータを上書きする場合は、デプロイ時に`--parameter-overrides`オプションを使用してください。

|名前|種類|説明|デフォルト値|
|---|---|---|---|
|BedrockRegion|String|Bedrockを呼び出すリージョン|us-west-2|
|GenerateTemplateModelId|String|テンプレート生成用のモデルID|anthropic.claude-3-opus-20240229-v1:0|
|ReviewTemplateModelId|String|テンプレートレビュー用のモデルID|anthropic.claude-3-opus-20240229-v1:0|
|GenerateParamsheetModelId|String|パラメータシート生成用のモデルID|anthropic.claude-3-opus-20240229-v1:0|
