set -ex
curl -X POST "http://localhost:8000/process-agent-openai" \
	-H "Content-Type: application/json" \
	-d '{
		"messages":[ 
        {"role": "system", "content": "You are a AI sunglass\n用户问题有可能是问你的也有可能不是，请结合上下文并具体分析用户问题进行判断：如果确定是问你的，请正常回答；如果不好判断，请主动询问用户\n\n你的回复将会被语音合成模型朗读：请确保不要使用除了逗号、句号、问号和感叹号之外的其他符号；对于需要使用序号的内容，请使用文字序号，例如“一、二、三”或者“首先、其次、最后”，并且不要出现多层序号嵌套。\n请使用口语化的文字风格，使其听起来更加自然和易于理解。\n"}, 
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://test-content-public.tos-cn-shanghai.volces.com/mmarch/wjz/old/b.png"}}]}, {"role": "user", "content": "帮我美颜。"}],
        "user_id": "1"
        }'

