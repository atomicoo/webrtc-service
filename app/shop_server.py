import json
import time
from flask import Flask, Response, request
# import logging

from util.llm_util import process_single
from util.logger import logger

app = Flask(__name__)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


@app.route('/chat', methods=['POST'])
def chat_with_user():
    data = request.json
    user_input = data['content']
    convs = data.get('conv_hist', [])
    if user_input == '**finish**':
        logger.info(f'会话结束')
        return {'reply': '会话结束'}
    conv_text = '\n'.join([f'**客户**: {conv[0]}\n**店铺客服**: {conv[1]}' for conv in convs])
    conv_text += f'\n**客户**: {user_input}'
    prompt = f'### 店铺信息\n- 名称：家电维修小店\n- 地址：家电维修城1栋203\n- 联系电话：4006666\n- 营业时间：9:00 - 18:00\n- 店铺业务：维修各类家电\n- 维修方案：对于每种家电有“简易维修”，“全面维修”，“深入维修”3种方案，价格区间从100元到1000元，可根据实际情况和复杂程度给出方案和报价\n\n### History\n```\n{conv_text}\n```\n\n### Workflow\n- 想象你是一个店铺的客服，`店铺信息`中包含该店铺的相关信息\n- 现在有一个客户需要维修家电\n- 你需要询问客户的需求，和必要的相关信息，如地址，联系方式等等\n- 然后给出维修方案供选择，用户确认方案后，你可以结束对话然后上门维修\n- `History`是你已经和客户之间发生的对话，现在你需要想象接下来需要和客户说什么\n- 只输出下一轮对话你要说的话，尽量简短，不需要一次性把所有的话说完，不要预测客户的回复\n\n你要说的话：'
    # prompt = '回复以下query，不要超过20个字' + user_input
    response = process_single(None, prompt)
    logger.info('>>>>>>>>>>>conv_text: ' + conv_text)
    logger.info('>>>>>>>>>>>input: ' + user_input)
    logger.info('>>>>>>>>>>>resp: ' + response + '\n')
    return {'reply': response}


@app.route('/chat_stream', methods=['POST'])
def chat_with_user_stream():
    data = request.json
    user_input = data['content']
    convs = data.get('conv_hist', [])
    if user_input == '**finish**':
        logger.info(f'会话结束')
        return json.dumps({"content": "会话结束", "delta":"会话结束", "lastOne": True})
    conv_text = '\n'.join([f'**客户**: {conv[0]}\n**店铺客服**: {conv[1]}' for conv in convs])
    conv_text += f'\n**客户**: {user_input}'
    prompt = f'### 店铺信息\n- 名称：家电维修小店\n- 地址：家电维修城1栋203\n- 联系电话：4006666\n- 营业时间：9:00 - 18:00\n- 店铺业务：维修各类家电家具\n- 维修方案：对于每种家电家具都有“简易维修”，“全面维修”，“深入维修”3种方案，价格区间从100元到1000元，可根据实际情况和复杂程度给出方案和报价\n\n### Task\n- 想象你是店铺客服，`店铺信息`中包含该店铺的相关信息，模拟一段你和客户之间的电话聊天\n- 你需要询问客户的维修需求，地址，联系方式，然后给出维修方案供选择，客户给出选择后就可以结束对话\n- `Conversation`是你和客户之间已经发生的对话和已经确认的信息，请你想象接下来你要说什么，续写对话，注意要符合`Constraints`中的要求\n\n### Constraints\n- 基于`Conversation`续写对话，只输出下一轮你要说的话，尽量简短，不要预测客户的回复\n- 不要问太多无关信息，不要闲聊，尽快结束对话，客户确认方案后，你可以回复客户“已确认全部信息可以结束对话”之类的\n\n### Conversation\n{conv_text}\n**店铺客服**: '
    # prompt = '回复以下query，不要超过20个字' + user_input
    # logger.info('>>>>>>>>>>>conv_text:\n' + conv_text)
    logger.info('>>>>>>>>>>>prompt:\n' + prompt)
    
    def generate():
        final_reply = ''
        response = process_single(None, prompt, stream=True)
        last_one = False
        line_record = ''
        try:
            for chunk in response.iter_content(chunk_size=None):
                if last_one:
                    break
                text = chunk.decode('utf-8')
                delta = ''
                content = ''
                for line in text.split('\n'):
                    if last_one:
                        break
                    line_record = line
                    line = line.strip()
                    logger.info(f'******chunk line:{line}******')
                    if line.startswith('data:'):
                        line = line[5:].strip()
                    if not line:
                        continue
                    chunk_data = json.loads(line)
                    last_one = chunk_data['lastOne']
                    if not last_one:
                        delta += chunk_data['choices'][0]['delta']['content']
                    content = chunk_data.get('content', '')
                if not content and not delta:
                    continue
                final_reply += delta
                yield json.dumps({"content": content, "delta":delta, "lastOne": last_one})
            logger.info(f'>>>>>>>>>>>final_reply: {final_reply}\n')
        except Exception as e:
            logger.error(f'Error: {e}, line: {line_record}')
            yield json.dumps({"content": "error", "delta":"error", "lastOne": True})
    return Response(generate(), content_type='text/plain')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7166, processes=1)
    # chat_with_user_stream('说一句话，20个字')
