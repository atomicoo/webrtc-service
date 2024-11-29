# start service
# ssh -p 2222 root@172.25.0.47
# cd /ML-A800/team/mm/wangjiazhi/projs/webrtc-service/app/service/agent
# unset HTTPS_PROXY; unset HTTP_PROXY
# python agent/app.py --port 2283

set -ex

# image_enhance 整体美化
# image_url="https://test-content-public.tos-cn-shanghai.volces.com/mmarch/wjz/old/b.png"
image_url="https://test-content-public.tos-cn-shanghai.volces.com/mmarch/wjz/scenery/a.jpg"
# image_url="https://test-content-public.tos-cn-shanghai.volces.com/mmarch/wjz/old/building_light_hsze.jpg"
# image_url="https://test-content-public.tos-cn-shanghai.volces.com/mmarch/wjz/old/phone_night_street_shop.jpg"
# image_url="https://img0.baidu.com/it/u=497184200,1201678943&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=587"
# image_url="https://test-content-public.tos-cn-shanghai.volces.com/mmarch/wjz/old/n.jpg"  # old woman, dark, hall
# image_url="https://test-content-public.tos-cn-shanghai.volces.com/mmarch/wjz/old/l.jpg"  # woman, tree, haze

# prompt="帮我美化"  # failed, will call image_beautify
# prompt="帮我修图"
# prompt="优化一下"  # not good, will call image_enhance & image_enhance_clarity
prompt="增强这张图"
prompt="把这张图弄好点"

# image_enhance_clarity 让图片更清晰
# image_url="https://test-content-public.tos-cn-shanghai.volces.com/mmarch/wjz/scenery/blur_road_girl.jpeg"
# image_url="https://test-content-public.tos-cn-shanghai.volces.com/mmarch/wjz/scenery/haze_newyork.webp"
prompt="让这张图更清晰"
# prompt="我需要通透一些"

# image_enhance_brighten 让图片更亮
# image_url="https://img0.baidu.com/it/u=497184200,1201678943&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=587"
# prompt="帮我提高亮度"
# prompt="亮一点"
# prompt="这图太暗了"


# image_transfer_recolor 给图片上色
# image_url="https://test-content-public.tos-cn-shanghai.volces.com/mmarch/wjz/old/771.jpg"
# prompt="帮我上色"

# image_url="https://img0.baidu.com/it/u=854128576,2125864126&fm=253&fmt=auto&app=138&f=JPEG?w=200&h=200"  # girl, smile, headshot
image_url="https://img0.baidu.com/it/u=1775001519,3235264083&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=500"
prompt="放大这张图片"

curl -X POST "http://172.25.0.47:8000/process-agent-openai" \
	-H "Content-Type: application/json" \
	-d '{
		"messages":[ 
        {"role": "system", "content": "You are a AI sunglass\n用户问题有可能是问你的也有可能不是，请结合上下文并具体分析用户问题进行判断：如果确定是问你的，请正常回答；如果不好判断，请主动询问用户\n\n你的回复将会被语音合成模型朗读：请确保不要使用除了逗号、句号、问号和感叹号之外的其他符号；对于需要使用序号的内容，请使用文字序号，例如“一、二、三”或者“首先、其次、最后”，并且不要出现多层序号嵌套。\n请使用口语化的文字风格，使其听起来更加自然和易于理解。\n"}, 
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "'$image_url'"}}]},
        {"role": "user", "content": "'$prompt'"}],
        "user_id": "1"
        }'
