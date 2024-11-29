set -ex
#curl -X POST "http://localhost:8000/process-agent" -H "Content-Type: application/json" -d '{"image_url": "https://test-content-public.tos-cn-shanghai.volces.com/agent/others/%E5%B7%B4%E5%8E%98%E5%B2%9B3.jpeg","prompt": "今天北京天气怎么样"}'
# curl -X POST "http://localhost:8000/process-agent" -H "Content-Type: application/json" -d '{"image_url":"", "prompt": "今天是什么日子?"}'
# curl -X POST "http://localhost:8000/process-agent-openai" -H "Content-Type: application/json" -d '{
# "messages":[ 
#         {"role": "system", "content": "现在你将扮演用户老年人的专属AI伴侣，你的名字是AI伴侣。你应该做到：（1）能够给予聊天用户温暖的陪伴；（2）你能够理解过去的[回忆]，如果它与当前问题相关，你可以结合[回忆]中的关键信息，回答问题。（3）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。(4) 当你不能直接回答问题的时候，请调用相关工具来解决问题。 <yi_sys>你后面的回复都会被语音合成模型朗读出来。请不要在回复中使用除了逗号，句号，问号，感叹号之外的其他符号。文字风格尽量口语化。</yi_sys>"}, 
#         {"role": "user", "content": [{"type": "text", "text": "\n日期2024-07-16的对话内容3为：\n老年人：AI App，早上好，有什么新闻值得关注的吗？\nAI：早上好！今天的头条是关于最近的国际旅行限制放宽，这可能影响到您未来的旅行计划。此外，当地有一个美食节将在下周举行，可能会很有趣。需要详细了解哪一个？\n\n以上是回忆的相关历史对话，可能和当前问题相关也可能不相关，如果不相关请忽略。\n回复用户当前的输入：今天是几号？"}]}],
#         "user_id": "10"
#         }'

curl -X POST "http://localhost:8000/process-agent-openai" -H "Content-Type: application/json" -d '{
        "messages":[ 
        {"role": "system", "content": "You are a AI sunglass\n用户问题有可能是问你的也有可能不是，请结合上下文并具体分析用户问题进行判断：如果确定是问你的，请正常回答；如果不好判断，请主动询问用户\n\n你的回复将会被语音合成模型朗读：请确保不要使用除了逗号、句号、问号和感叹号之外的其他符号；对于需要使用序号的内容，请使用文字序号，例如“一、二、三”或者“首先、其次、最后”，并且不要出现多层序号嵌套。\n请使用口语化的文字风格，使其听起来更加自然和易于理解。\n"}, 
        {"role": "user", "content": "我在北京，我的小米手机屏幕坏了，帮我打电话修理一下"}],
        "user_id": "10"
        }'

# curl -X POST "http://localhost:8000/process-agent-openai" -H "Content-Type: application/json" -d '{
# "messages":[ 
#         {"role": "user", "content": "今天是几号？"}],
#         "user_id": "10"
#         }'
