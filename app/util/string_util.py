import re


def split_sentence_at_last_punctuation(sentence, punctuation_marks="。,，!！?？;；:：\n"):
    """
    优化版：将一句话在最后一个标点处分成两部分。
    前面部分包含到最后一个标点，后面部分不包含标点。
    从后向前遍历句子以找到最后一个标点。
    """
    # punctuation_marks = "。,，!！?？;；:：\n"
    # 从后向前遍历句子，找到最后一个标点的位置
    for i in range(len(sentence) - 1, -1, -1):
        if sentence[i] in punctuation_marks:
            # 根据最后一个标点的位置，分割句子
            return sentence[:i + 1], sentence[i + 1:]
    # 如果没有找到标点，返回空字符串和原句
    return "", sentence


def utf_8_len(text: str):
    return len(text.encode("utf-8"))


def is_meaningful_string(string):
    pattern = r'[\u4e00-\u9fff]+|[a-zA-Z]+'
    return True if re.search(pattern, string) else False


class StreamingTextProcessor:
    def __init__(self):
        self.buffer = ""
        self.complete_text_record = ""
        self.raw_text_record = ""
        self.first_complete_sentence_found = False

    def maybe_split_final_piece(self):
        while utf_8_len(self.buffer) > 160:
            split_points = [m.end() for m in re.finditer(r'[!?。！？]', self.buffer) \
                            if utf_8_len(self.buffer[:m.end()]) > 120]
            if not split_points:
                break
            split_point = split_points[0]
            yield self.buffer[:split_point]
            self.buffer = self.buffer[split_point:]
        yield self.buffer[:]
        self.buffer = ""

    def filter_and_process(self, text_chunk, is_final_chunk):
        # 原始的文本记录
        self.raw_text_record += text_chunk

        text_chunk = text_chunk.replace('*', '')
        self.buffer += text_chunk

        # 定义无需TTS的文本片段的模式
        pattern = r'<notts>(.*?)</notts>'
        # pattern = '|'.join([r'<notts>(.*?)</notts>'])

        # 检查是否有匹配模式的文本片段
        match = re.search(pattern, self.buffer, re.DOTALL)
        if match is not None:
            # 移除找到的文本片段
            self.buffer = self.buffer[:match.start()] + self.buffer[match.end():]

        # 根据阶段使用不同的分句逻辑
        notts = re.search(r'<notts>', self.buffer)
        notts_start = notts.start() if notts is not None else len(self.buffer)
        if is_final_chunk:
            return self.maybe_split_final_piece()
        elif not self.first_complete_sentence_found:
            # 第一阶段：允许逗号、分号、冒号等半句标点
            split_points = [m.end() for m in re.finditer(r'[!?,;:。！？，；：\n]', self.buffer) \
                            if m.end() <= notts_start and utf_8_len(self.buffer[:m.end()]) > 15]
            if not split_points:
                return ""
            split_point = split_points[-1]
            split_points = [m.end() for m in re.finditer(r'[!?。！？\n]', self.buffer) \
                            if m.end() <= notts_start and utf_8_len(self.buffer[:m.end()]) > 15]
            if split_points:
                split_point = split_points[-1]
            complete_sentence = self.buffer[:split_point]
            self.buffer = self.buffer[split_point:]
        else:
            # 第二阶段：只允许句号、问号、叹号等整句标点
            split_points = [m.end() for m in re.finditer(r'[!?。！？\n]', self.buffer) \
                            if m.end() <= notts_start and utf_8_len(self.buffer[:m.end()]) > 120]
            if not split_points:
                return ""
            split_point = split_points[0]
            complete_sentence = self.buffer[:split_point]
            self.buffer = self.buffer[split_point:]

        # 更新完整文本记录
        self.complete_text_record += complete_sentence
        if utf_8_len(self.complete_text_record) > 30:
            self.first_complete_sentence_found = True
        return complete_sentence


if __name__ == "__main__":
    # text = "好 我来背一首李白的《蜀道难》\n\n噫吁嚱 危乎高哉 蜀道之难 难于上青天\n\n蚕丛及鱼凫 开国何茫然 尔来四万八千岁 不与秦塞通人烟\n\n西当太白有鸟道 可以横绝峨眉巅 地崩山摧壮士死 然后天梯石栈相钩连\n\n上有六龙回日之高标 下有冲波逆折之回川 黄鹤之飞尚不得过 猿猱欲度愁攀援\n\n青泥何盘盘 百步九折萦岩峦 扪参历井仰胁息 以手抚膺坐长叹\n\n问君西游何时还 畏途巉岩不可攀 但见悲鸟号古木 雄飞雌从绕林间\n\n又闻子规啼夜月 愁空山\n\n蜀道之难 难于上青天 使人听此凋朱颜\n\n连峰去天不盈尺 枯松倒挂倚绝壁 飞湍瀑流争喧豗 砯崖转石万壑雷\n\n其险也如此 嗟尔远道之人 胡为乎来哉\n\n剑阁峥嵘而崔嵬 一夫当关 万夫莫开\n\n所守或匪亲 化为狼与豺 朝避猛虎 夕避长蛇 磨牙吮血 杀人如麻\n\n锦城虽云乐 不如早还家\n\n蜀道之难 难于上青天 侧身西望长咨嗟"
    text = "<notts>\n>>>>>>Turn 0\n>>>>>>用户: 您好，我需要安排一次冰箱维修服务。地址是北京市海淀区中关村，联系人是老年人，联系电话是18800006666。请问您可以安排维修人员上门服务吗？预计到达时间是什么时候？\n>>>>>>店铺回复: 您好，我们可以安排维修人员上门服务。请问您的冰箱目前有什么问题呢？我们有“简易维修”，“全面维修”和“深入维修”三种方案，价格区间从100元到1000元不等，您可以根据实际情况选择。\n>>>>>>Turn 1\n>>>>>>用户: 冰箱目前不制冷，请安排“全面维修”方案。请告知预计到达时间。\n>>>>>>店铺回复: 好的，感谢您的选择。我们会尽快安排维修人员上门\n>>>>>>Turn 2\n>>>>>>用户: 请尽快告知预计到达时间。谢谢。\n>>>>>>店铺回复: 我们会尽快安排维修人员，并在30分钟内与您联系确认具体到达时间。请保持电话畅通。感谢您的耐心等待。\n>>>>>>Turn 3\n>>>>>>用户: 好的，谢谢您的回复。我会保持电话畅通，等待维修人员的联系。结束对话，对话的总结和关键信息为：用户老年人（联系电话：18800006666）在北京市海淀区中关村需要冰箱维修服务，选择了“全面维修”方案。维修服务将在30分钟内联系用户确认具体到达时间。</notts>我已经联系了冰箱维修服务。他们会在30分钟内与您联系确认具体到达时间。请保持电话畅通。维修人员会根据情况提供“全面维修”方案，解决冰箱不制冷的问题。"
    processor = StreamingTextProcessor()
    for i, chunk in enumerate(text, start=1):
        lastOne = (i == len(text))
        if not lastOne:
            sentence = processor.filter_and_process(chunk, is_final_chunk=False)
            if sentence != "":
                print(sentence)
        else:
            generator = processor.filter_and_process(chunk, is_final_chunk=True)
            for sentence in generator:
                if sentence != "":
                    print(sentence)