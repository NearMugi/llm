# LLM

## Docker操作

```bash
# コンテナの立ち上げ
docker-compose up -d

# コンテナに入る
docker exec -it llm-app bash

# コンテナの停止
docker-compose down

# Dockerイメージの削除
docker rmi xxx
```

## マウントしているフォルダ

./mount <-> /root/mount

## モデルをダウンロードする方法

GPUのスペックが足りないため、モデルを動かすことはできなかった。  
履歴は load_model 参照

### HaggingFace へのログイン

HaggingFace からモデルをダウンロードするときはログインが必要らしい(ログインしなくても動いていた気もする)  
モデルをダウンロードするにはHaggingFaceのトークンを登録する必要がある
参考URL https://murci.net/archives/5119

```
ダウンロードスクリプト実行時に必要になるURLがメールで通知される
Hugging FaceでSignUpする
https://huggingface.co/
Read/Write権限を選択してトークンを発行すると、メールでトークンが通知される
```

https://huggingface.co/docs/huggingface_hub/quick-start#login-command
トークンを作成した後、pythonでコマンドを直接入力した

``` bash
USER@home-workspace:~/_workInUbuntu/LLM$ pip install --upgrade huggingface_hub
・・・
USER@home-workspace:~/_workInUbuntu/LLM$ python3
Python 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from huggingface_hub import login
>>> login()

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    A token is already saved on your machine. Run `huggingface-cli whoami` to get more information or `huggingface-cli logout` if you want to log out.
    Setting a new token will erase the existing one.
    To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Token:
Add token as git credential? (Y/n) Y
Token is valid (permission: read).
Cannot authenticate through git-credential as no helper is defined on your machine.
You might have to re-authenticate when pushing to the Hugging Face Hub.
Run the following command in your terminal in case you want to set the 'store' credential helper as default.

git config --global credential.helper store

Read https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more details.
Token has not been saved to git credential helper.
Your token has been saved to /home/USER/.cache/huggingface/token
Login successful
>>>
```

