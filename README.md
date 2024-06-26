# sam-ai-cloud-engineer

## 概要

事前に作成したAWS構成図をアップロードすると、OutputバケットにCloudFormationテンプレートとパラメータシートが生成されるサーバレスアプリケーションをSAMで構築します。  

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

## 使い方

1. Inputバケットに構成図の画像(png形式)をアップロードする
2. 一定時間待機する
3. 各々のOutputバケットにCloudFormationテンプレート(yaml形式)とパラメータシート(csv形式)が生成される
4. 生成されたCloudFormationテンプレートについては、CloudFormationスタック作成時にオブジェクトURIを指定する事でそのままデプロイ可能
