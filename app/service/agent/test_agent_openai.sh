set -ex
curl -X POST "http://localhost:8000/process-agent-openai" \
	-H "Content-Type: application/json" \
	-d '{
		"messages": [
		{
		"role": "system",
		"content": "<yi_sys>你后面的回复都会被语音合成模型朗读出来。请不要在回复中使用除了逗号，句号，问号，感叹号之外的其他符号。文字风格尽量口语化。</yi_sys>"
		},
    		{
        	"role": "user",
        	"content": [
        		{
            	"type": "image_url",
            	"image_url": {
            	"url": "https://01-platform-public.oss-cn-beijing.aliyuncs.com/playground/image/a0459579-d63b-4d94-9153-f18c413cf919-7cd70aad-db9c-4a76-8c6a-91d976aef92a-%E6%88%AA%E5%B1%8F2024-07-26%2015.41.23.png"
            }
        	},
        	{
            	"type": "text",
            	"text": "这个东西叫什么"
        	}
        	]
    	}	
    ],
    "user_id":"10"
}'

