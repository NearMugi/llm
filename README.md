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

./work <-> /root/work

