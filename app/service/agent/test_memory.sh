set -ex
curl -X POST "http://localhost:8000/process-agent" -H "Content-Type: application/json" -d '{"image_url": "", "prompt": "买了牛肉，有什么午饭推荐吗"}'