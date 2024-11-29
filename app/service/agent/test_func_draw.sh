# set -ex
# curl -X POST "http://localhost:8000/process-agent-openai" -H "Content-Type: application/json" -d '{
#   "messages": [
#       {
# 	            "role": "user",
# 		          "content": [
# 			          {
# 					            "type": "text",
# 						              "text": "画一个保时捷赛车"
# 							              }
# 								            ]
# 									        }
# 										  ]
# 									  }'
curl -X POST "http://localhost:8000/process-agent-openai" -H "Content-Type: application/json" -d '{
			"messages": 
			[{"role": "system", "content": "You are a AI sunglass\n用户问题有可能是问你的也有可能不是，请结合上下文并具体分析用户问题进行判断：如果确定是问你的，请正常回答；如果不好判断，请主动询问用户\n\n你的回复将会被语音合成模型朗读：请确保不要使用除了逗号、句号、问号和感叹号之外的其他符号；对于需要使用序号的内容，请使用文字序号，例如“一、二、三”或者“首先、其次、最后”，并且不要出现多层序号嵌套。\n请使用口语化的文字风格，使其听起来更加自然和易于理解。\n"},
			{"role": "user", "content": "是的，那个。"},
			{"role": "assistant", "content": "你是不是有点想不起来要说什么了？没关系，慢慢来，我在这里陪你。"},
			{"role": "user", "content": "帮我画一只黑色的小狗，毛茸茸的大眼睛。"}],
			"user_id": "1"
 			}'