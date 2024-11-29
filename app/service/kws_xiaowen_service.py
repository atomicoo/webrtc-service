from conf.model_conf import MODEL_DICT

import os
import librosa
import math
import numpy as np
import onnxruntime as ort
import re
import yaml
import torch
from collections import defaultdict
from service.kaldi_fbank import fbank
from util.logger import logger


symbol_str = '[’!"#$%&\'()*+,-./:;<>=?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

def split_mixed_label(input_str):
    tokens = []
    s = input_str.lower()
    while len(s) > 0:
        match = re.match(r'[A-Za-z!?,<>()\']+', s)
        if match is not None:
            word = match.group(0)
        else:
            word = s[0:1]
        tokens.append(word)
        s = s.replace(word, '', 1).strip(' ')
    return tokens

def query_token_set(txt, symbol_table, lexicon_table):
    tokens_str = tuple()
    tokens_idx = tuple()

    parts = split_mixed_label(txt)
    for part in parts:
        if part == '!sil' or part == '(sil)' or part == '<sil>':
            tokens_str = tokens_str + ('!sil', )
        elif part == '<blk>' or part == '<blank>':
            tokens_str = tokens_str + ('<blk>', )
        elif part == '(noise)' or part == 'noise)' or \
                part == '(noise' or part == '<noise>':
            tokens_str = tokens_str + ('<GBG>', )
        elif part in symbol_table:
            tokens_str = tokens_str + (part, )
        elif part in lexicon_table:
            for ch in lexicon_table[part]:
                tokens_str = tokens_str + (ch, )
        else:
            # case with symbols or meaningless english letter combination
            part = re.sub(symbol_str, '', part)
            for ch in part:
                tokens_str = tokens_str + (ch, )

    for ch in tokens_str:
        if ch in symbol_table:
            tokens_idx = tokens_idx + (symbol_table[ch], )
        elif ch == '!sil':
            if 'sil' in symbol_table:
                tokens_idx = tokens_idx + (symbol_table['sil'], )
            else:
                tokens_idx = tokens_idx + (symbol_table['<blk>'], )
        elif ch == '<GBG>':
            if '<GBG>' in symbol_table:
                tokens_idx = tokens_idx + (symbol_table['<GBG>'], )
            else:
                tokens_idx = tokens_idx + (symbol_table['<blk>'], )
        else:
            if '<GBG>' in symbol_table:
                tokens_idx = tokens_idx + (symbol_table['<GBG>'], )
                logger.info(f'{ch} is not in token set, replace with <GBG>')
            else:
                tokens_idx = tokens_idx + (symbol_table['<blk>'], )
                logger.info(f'{ch} is not in token set, replace with <blk>')

    return tokens_str, tokens_idx


def read_token(token_file):
    tokens_table = {}
    with open(token_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            tokens_table[arr[0]] = int(arr[1]) - 1
    fin.close()
    return tokens_table


def read_lexicon(lexicon_file):
    lexicon_table = {}
    with open(lexicon_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().replace('\t', ' ').split()
            assert len(arr) >= 2
            lexicon_table[arr[0]] = arr[1:]
    fin.close()
    return lexicon_table


def is_sublist(main_list, check_list):
    if len(main_list) < len(check_list):
        return -1

    if len(main_list) == len(check_list):
        return 0 if main_list == check_list else -1

    for i in range(len(main_list) - len(check_list)):
        if main_list[i] == check_list[0]:
            for j in range(len(check_list)):
                if main_list[i + j] != check_list[j]:
                    break
            else:
                return i
    else:
        return -1

def ctc_prefix_beam_search(t, probs, cur_hyps, keywords_idxset, score_beam_size):
    '''

    :param t: the time in frame
    :param probs: the probability in t_th frame, (vocab_size, )
    :param cur_hyps: list of tuples. [(tuple(), (1.0, 0.0, []))]
                in tuple, 1st is prefix id, 2nd include p_blank, p_non_blank, and path nodes list.
                in path nodes list, each node is a dict of {token=idx, frame=t, prob=ps}
    :param keywords_idxset: the index of keywords in token.txt
    :param score_beam_size: the probability threshold, to filter out those frames with low probs.
    :return:
            next_hyps: the hypothesis depend on current hyp and current frame.
    '''
    # key: prefix, value (pb, pnb), default value(-inf, -inf)
    next_hyps = defaultdict(lambda: (0.0, 0.0, []))

    # 2.1 First beam prune: select topk best
    top_k_index = np.argsort(probs, axis=-1)[..., -score_beam_size:]
    top_k_probs = np.take_along_axis(probs, top_k_index, axis=-1)

    # filter prob score that is too small
    filter_probs = []
    filter_index = []
    for prob, idx in zip(top_k_probs.tolist(), top_k_index.tolist()):
        if keywords_idxset is not None:
            if prob > 0.05 and idx in keywords_idxset:
                filter_probs.append(prob)
                filter_index.append(idx)
        else:
            if prob > 0.05:
                filter_probs.append(prob)
                filter_index.append(idx)

    if len(filter_index) == 0:
        return cur_hyps

    for s in filter_index:
        ps = probs[s].item()

        for prefix, (pb, pnb, cur_nodes) in cur_hyps:
            last = prefix[-1] if len(prefix) > 0 else None
            if s == 0:  # blank
                n_pb, n_pnb, nodes = next_hyps[prefix]
                n_pb = n_pb + pb * ps + pnb * ps
                nodes = cur_nodes.copy()
                next_hyps[prefix] = (n_pb, n_pnb, nodes)
            elif s == last:
                if not math.isclose(pnb, 0.0, abs_tol=0.000001):
                    # Update *ss -> *s;
                    n_pb, n_pnb, nodes = next_hyps[prefix]
                    n_pnb = n_pnb + pnb * ps
                    nodes = cur_nodes.copy()
                    if ps > nodes[-1]['prob']:  # update frame and prob
                        nodes[-1]['prob'] = ps
                        nodes[-1]['frame'] = t
                    next_hyps[prefix] = (n_pb, n_pnb, nodes)

                if not math.isclose(pb, 0.0, abs_tol=0.000001):
                    # Update *s-s -> *ss, - is for blank
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb, nodes = next_hyps[n_prefix]
                    n_pnb = n_pnb + pb * ps
                    nodes = cur_nodes.copy()
                    nodes.append(dict(token=s, frame=t,
                                      prob=ps))  # to record token prob
                    next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
            else:
                n_prefix = prefix + (s,)
                n_pb, n_pnb, nodes = next_hyps[n_prefix]
                if nodes:
                    if ps > nodes[-1]['prob']:  # update frame and prob
                        # nodes[-1]['prob'] = ps
                        # nodes[-1]['frame'] = t
                        nodes.pop()  # to avoid change other beam which has this node.
                        nodes.append(dict(token=s, frame=t, prob=ps))
                else:
                    nodes = cur_nodes.copy()
                    nodes.append(dict(token=s, frame=t,
                                      prob=ps))  # to record token prob
                n_pnb = n_pnb + pb * ps + pnb * ps
                next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

    # 2.2 Second beam prune
    next_hyps = sorted(
        next_hyps.items(), key=lambda x: (x[1][0] + x[1][1]), reverse=True)

    return next_hyps


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        logger.info('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        logger.info('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint)
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


class KeyWordSpotter(torch.nn.Module):
    def __init__(self, ckpt_path, config_path, token_path, lexicon_path,
                 threshold, min_frames=5, max_frames=250, interval_frames=50,
                 score_beam=3, path_beam=20):
        super().__init__()
        with open(config_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        dataset_conf = configs['dataset_conf']

        # feature related
        self.sample_rate = 16000
        self.wave_remained = np.array([])
        self.num_mel_bins = dataset_conf['feature_extraction_conf']['num_mel_bins']
        self.frame_length = dataset_conf['feature_extraction_conf']['frame_length']  # in ms
        self.frame_shift = dataset_conf['feature_extraction_conf']['frame_shift']    # in ms
        self.downsampling = dataset_conf.get('frame_skip', 1)
        self.resolution = self.frame_shift / 1000   # in second
        # fsmn splice operation
        self.context_expansion = dataset_conf.get('context_expansion', False)
        self.left_context = 0
        self.right_context = 0
        if self.context_expansion:
            self.left_context = dataset_conf['context_expansion_conf']['left']
            self.right_context = dataset_conf['context_expansion_conf']['right']
        self.feature_remained = None
        self.feats_ctx_offset = 0  # after downsample, offset exist.


        # model related
        model_path = MODEL_DICT['keyword-spot-fsmn-ctc-wenwen']
        session_options = ort.SessionOptions()
        self.model = ort.InferenceSession(
            f'{model_path}/onnx/keyword_spot_fsmn_ctc_wenwen.onnx', providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )
        self.device = torch.device('cpu')
        logger.info(f'model {ckpt_path} loaded.')
        self.token_table = read_token(token_path)
        logger.info(f'tokens {token_path} with {len(self.token_table)} units loaded.')
        self.lexicon_table = read_lexicon(lexicon_path)
        logger.info(f'lexicons {lexicon_path} with {len(self.lexicon_table)} units loaded.')
        self.in_cache = np.zeros((1, 128, 11, 4), dtype=np.float32)


        # decoding and detection related
        self.score_beam = score_beam
        self.path_beam = path_beam

        self.threshold = threshold
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.interval_frames = interval_frames

        self.cur_hyps = [(tuple(), (1.0, 0.0, []))]
        self.hit_score = 1.0
        self.hit_keyword = None
        self.activated = False

        self.total_frames = 0   # frame offset, for absolute time
        self.last_active_pos = -1  # the last frame of being activated
        self.result = {}

    def set_keywords(self, keywords):
        # 4. parse keywords tokens
        assert keywords is not None, 'at least one keyword is needed, multiple keywords should be splitted with comma(,)'
        keywords_str = keywords
        keywords_list = keywords_str.strip().replace(' ', '').split(',')
        keywords_token = {}
        keywords_idxset = {0}
        keywords_strset = {'<blk>'}
        keywords_tokenmap = {'<blk>': 0}
        for keyword in keywords_list:
            strs, indexes = query_token_set(keyword, self.token_table, self.lexicon_table)
            keywords_token[keyword] = {}
            keywords_token[keyword]['token_id'] = indexes
            keywords_token[keyword]['token_str'] = ''.join('%s ' % str(i)
                                                           for i in indexes)
            [keywords_strset.add(i) for i in strs]
            [keywords_idxset.add(i) for i in indexes]
            for txt, idx in zip(strs, indexes):
                if keywords_tokenmap.get(txt, None) is None:
                    keywords_tokenmap[txt] = idx

        token_print = ''
        for txt, idx in keywords_tokenmap.items():
            token_print += f'{txt}({idx}) '
        logger.info(f'Token set is: {token_print}')
        self.keywords_idxset = keywords_idxset
        self.keywords_token = keywords_token

    def accept_wave(self, wave):
        wave = np.frombuffer(wave, dtype=np.int16)
        # here we don't divide 32768.0, because kaldi.fbank accept original input
        wave = np.append(self.wave_remained, wave)
        if wave.size < (self.frame_length * self.sample_rate / 1000) * self.right_context :
            self.wave_remained = wave
            return None
        feats = fbank(wave,
                      num_mel_bins=self.num_mel_bins,
                      frame_length=self.frame_length,
                      frame_shift=self.frame_shift,
                      sample_frequency=self.sample_rate)
        # update wave remained
        feat_len = len(feats)
        frame_shift = int(self.frame_shift / 1000 * self.sample_rate)
        self.wave_remained = wave[feat_len * frame_shift:]

        if self.context_expansion:
            assert feat_len > self.right_context, "make sure each chunk feat length is large than right context."
            # pad feats with remained feature from last chunk
            if self.feature_remained is None:  # first chunk
                # pad first frame at the beginning, replicate just support last dimension, so we do transpose.
                feats_pad = np.pad(feats.T, ((0, 0), (self.left_context, 0)), mode='edge').T # F.pad(feats.T, (self.left_context, 0), mode='replicate').T
            else:
                feats_pad = np.concatenate((self.feature_remained, feats)) # torch.cat((self.feature_remained, feats))

            ctx_frm = feats_pad.shape[0] - (self.right_context+self.right_context)
            ctx_win = (self.left_context + self.right_context + 1)
            ctx_dim = feats.shape[1] * ctx_win
            feats_ctx = np.zeros((ctx_frm, ctx_dim), dtype=np.float32) # torch.zeros(ctx_frm, ctx_dim, dtype=torch.float32)
            for i in range(ctx_frm):
                feats_ctx[i] = np.concatenate(feats_pad[i: i + ctx_win], axis=0) # torch.cat(tuple(feats_pad[i: i + ctx_win])).unsqueeze(0)

            # update feature remained, and feats
            self.feature_remained = feats[-(self.left_context+self.right_context):]
            feats = feats_ctx
        if self.downsampling > 1:
            last_remainder = 0 if self.feats_ctx_offset==0 else self.downsampling-self.feats_ctx_offset
            remainder = (feats.shape[0]+last_remainder) % self.downsampling
            feats = feats[self.feats_ctx_offset::self.downsampling, :]
            self.feats_ctx_offset = remainder if remainder == 0 else self.downsampling-remainder
        return feats

    def decode_keywords(self, t, probs):
        absolute_time = t + self.total_frames
        # search next_hyps depend on current probs and hyps.
        next_hyps = ctc_prefix_beam_search(absolute_time,
                                           probs,
                                           self.cur_hyps,
                                           self.keywords_idxset,
                                           self.score_beam)
        # update cur_hyps. note: the hyps is sort by path score(pnb+pb), not the keywords' probabilities.
        cur_hyps = next_hyps[:self.path_beam]
        self.cur_hyps = cur_hyps

    def execute_detection(self, t):
        absolute_time = t + self.total_frames
        hit_keyword = None
        start = 0
        end = 0

        # hyps for detection
        hyps = [(y[0], y[1][0] + y[1][1], y[1][2]) for y in self.cur_hyps]

        # detect keywords in decoding paths.
        for one_hyp in hyps:
            prefix_ids = one_hyp[0]
            # path_score = one_hyp[1]
            prefix_nodes = one_hyp[2]
            assert len(prefix_ids) == len(prefix_nodes)
            for word in self.keywords_token.keys():
                lab = self.keywords_token[word]['token_id']
                offset = is_sublist(prefix_ids, lab)
                if offset != -1:
                    hit_keyword = word
                    start = prefix_nodes[offset]['frame']
                    end = prefix_nodes[offset + len(lab) - 1]['frame']
                    for idx in range(offset, offset + len(lab)):
                        self.hit_score *= prefix_nodes[idx]['prob']
                    break
            if hit_keyword is not None:
                self.hit_score = math.sqrt(self.hit_score)
                break

        duration = end - start
        if hit_keyword is not None:
            if self.hit_score >= self.threshold and \
                    self.min_frames <= duration <= self.max_frames \
                    and (self.last_active_pos==-1 or end-self.last_active_pos >= self.interval_frames):
                self.activated = True
                self.last_active_pos = end
                logger.info(
                    f"Frame {absolute_time} detect {hit_keyword} from {start} to {end} frame. "
                    f"duration {duration}, score {self.hit_score}, Activated.")

            elif self.last_active_pos>0 and end-self.last_active_pos < self.interval_frames:
                logger.info(
                    f"Frame {absolute_time} detect {hit_keyword} from {start} to {end} frame. "
                    f"but interval {end-self.last_active_pos} is lower than {self.interval_frames}, Deactivated. ")

            elif self.hit_score < self.threshold:
                logger.info(
                    f"Frame {absolute_time} detect {hit_keyword} from {start} to {end} frame. "
                    f"but {self.hit_score} is lower than {self.threshold}, Deactivated. ")

            elif self.min_frames > duration or duration > self.max_frames:
                logger.info(
                    f"Frame {absolute_time} detect {hit_keyword} from {start} to {end} frame. "
                    f"but {duration} beyond range({self.min_frames}~{self.max_frames}), Deactivated. ")

        self.result = {
            "state": 1 if self.activated else 0,
            "keyword": hit_keyword if self.activated else None,
            "start": start * self.resolution if self.activated else None,
            "end": end * self.resolution if self.activated else None,
            "score": self.hit_score if self.activated else None
        }

    def forward(self, wave_chunk):
        feature = self.accept_wave(wave_chunk)
        if feature is None or feature.shape[0] < 1:
            return {}  # # the feature is not enough to get result.
        # feature = feature.detach().cpu().numpy()
        feature = np.expand_dims(feature, axis=0)   # add a batch dimension
        probs, self.in_cache = self.model.run(None, {'input': feature, 'cache': self.in_cache})
        probs = np.squeeze(probs, axis=0)  # remove batch dimension, move to cpu for ctc_prefix_beam_search
        for (t, prob) in enumerate(probs):
            t *= self.downsampling
            self.decode_keywords(t, prob)
            self.execute_detection(t)
            if self.activated:
                self.reset()
                # since a chunk include about 30 frames, once activated, we can jump the latter frames.
                # TODO: there should give another method to update result, avoiding self.result being cleared.
                break
        self.total_frames += len(probs) * self.downsampling  # update frame offset
        # For streaming kws, the cur_hyps should be reset if the time of
        # a possible keyword last over the max_frames value you set.
        # see this issue:https://github.com/duj12/kws_demo/issues/2
        if len(self.cur_hyps) > 0 and len(self.cur_hyps[0][0]) > 0:
            keyword_may_start = int(self.cur_hyps[0][1][2][0]['frame'])
            if (self.total_frames - keyword_may_start) > self.max_frames:
                self.reset()
        return self.result

    def reset(self):
        self.cur_hyps = [(tuple(), (1.0, 0.0, []))]
        self.activated = False
        self.hit_score = 1.0

    def reset_all(self):
        self.reset()
        self.wave_remained = np.array([])
        self.feature_remained = None
        self.feats_ctx_offset = 0  # after downsample, offset exist.
        self.in_cache = np.zeros((1, 128, 11, 4), dtype=np.float32)
        self.total_frames = 0   # frame offset, for absolute time
        self.last_active_pos = -1  # the last frame of being activated
        self.result = {}


class KwsXiaowenService:
    """ KWS Xiaowen Service """
    def __init__(self):
        model_path = MODEL_DICT['keyword-spot-fsmn-ctc-wenwen']
        self.keyword_spotter = KeyWordSpotter(ckpt_path=f'{model_path}/avg_30.pt',
                                              config_path=f'{model_path}/config.yaml',
                                              token_path=f'{model_path}/tokens.txt',
                                              lexicon_path=f'{model_path}/lexicon.txt',
                                              threshold=0.02,
                                              min_frames=5,
                                              max_frames=250,
                                              interval_frames=50,
                                              score_beam=3,
                                              path_beam=20)

        self.set_keywords(keywords="嗨小问,你好问问")

    def set_keywords(self, keywords):
        self.keyword_spotter.set_keywords(keywords)

    def detect_chunk(self, audio_bytes: bytes):
        return self.keyword_spotter.forward(audio_bytes)

    def detect(self, audio):
        self.keyword_spotter.reset_all()

        y, _ = librosa.load(audio, sr=16000)
        # NOTE: model supports 16k sample_rate
        wav = (y * (1 << 15)).astype("int16").tobytes()

        # We inference every 0.3 seconds, in streaming fashion.
        interval = int(0.3 * 16000) * 2
        for i in range(0, len(wav), interval):
            chunk_wav = wav[i: min(i + interval, len(wav))]
            result = self.detect_chunk(chunk_wav)
            if 'state' in result and result['state'] == 1:
                return f"Activated: Detect {result['keyword']} from {result['start']} to {result['end']} second."
        return "Deactivated."


kws_xiaowen_service = KwsXiaowenService()


if __name__ == "__main__":
    audio = r'/ML-A800/team/mm/zhouzhiyang/WORKSPACE/SERVICE/KWS_Nihao_Xiaojing/examples/kws_hixiaowen.wav'
    kws_xiaowen_service.set_keywords(keywords="嗨小问,你好问问")
    print(kws_xiaowen_service.detect(audio))
