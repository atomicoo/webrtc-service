set -ex
#curl -X POST "http://localhost:8000/process-agent" -H "Content-Type: application/json" -d '{"image_url": "https://test-content-public.tos-cn-shanghai.volces.com/agent/others/%E5%B7%B4%E5%8E%98%E5%B2%9B3.jpeg","prompt": "今天北京天气怎么样"}'
curl -X POST "http://localhost:8000/process-agent-s" -H "Content-Type: application/json" -d '{"image_url":"", "prompt": "介绍一下自己"}' -N
