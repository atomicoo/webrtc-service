import os

# MODEL_NAME = "yi-vl-plus"
MODEL_NAME = "gpt-4o"

MODEL_BASE_PATH = '/ML-A800/team/mm/zhouzhiyang/WORKSPACE/SERVICE/webrtc-service/hub'
MODEL_BASE_PATH = os.getenv("MODEL_WEIGHT", MODEL_BASE_PATH)
PCM_SAVE_PATH = '/ML-A800/team/mm/zhouzhiyang/WORKSPACE/SERVICE/webrtc-service/pcm'
PCM_SAVE_PATH = os.getenv("PCM_SAVE_PATH", PCM_SAVE_PATH)

MODEL_DICT = {
    'paraformer-zh': f'{MODEL_BASE_PATH}/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    'paraformer-zh-streaming': f'{MODEL_BASE_PATH}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
    'ct-punc': f'{MODEL_BASE_PATH}/punc_ct-transformer_cn-en-common-vocab471067-large',
    'fsmn-vad': f'{MODEL_BASE_PATH}/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    'sense-voice-small': f'{MODEL_BASE_PATH}/SenseVoiceSmall',
    'cosy-voice-300m-sft': f'{MODEL_BASE_PATH}/CosyVoice-300M-SFT',
    'uvr-mdx-net-inst-hq-3': f'{MODEL_BASE_PATH}/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx',
    'denoise-deepfilter-2': f'{MODEL_BASE_PATH}/DeepFilterNet2',
    'denoise-deepfilter-streaming': f'{MODEL_BASE_PATH}/speech_deepfilternet_denoise_streaming_16k',
    'sv-campplus': f'{MODEL_BASE_PATH}/speech_campplus_sv_zh-cn_16k-common',
    'mossformer2-2spk': f'{MODEL_BASE_PATH}/speech_mossformer2_separation_temporal_16k',
    'keyword-spot-fsmn-ctc-wenwen': f'{MODEL_BASE_PATH}/keyword-spot-fsmn-ctc-wenwen',
}

LLM_CONFIG = {
    "openai_api_base": "http://api.lingyiwanwu.com/v1",
    "openai_api_key": "multimodel-peter"
}