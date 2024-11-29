# WebRTC demo

## Agora RTC SDK
- [Sheng Wang](https://console.shengwang.cn/): sign up an account and create an app, then get your `APPID` and `APP CERTIFICATE`
- set config `APPID` and `APP_CERTIFICATE` in `app/conf/rtc_conf.py`

## Install for requirements
```shell
conda create -n rtc python=3.11
conda activate rtc
pip3 install -r requirements.txt -i https://mirrors.ivolces.com/pypi/simple
```

## Install related models
- install modelscope or huggingface-hub (already in `requirements.txt`)
- download related models (see `app/conf/model_conf.py` for details) from [ModelScope](https://www.modelscope.cn/models) or [HuggingFace Hub](https://huggingface.co/models)
- set config `MODEL_BASE_PATH` and `PCM_SAVE_PATH` in `app/conf/model_conf.py`
- all related models download in [here](/ML-A800/team/mm/zhouzhiyang/WORKSPACE/SERVICE/webrtc-service/hub) now, so you can just set `MODEL_BASE_PATH="/ML-A800/team/mm/zhouzhiyang/WORKSPACE/SERVICE/webrtc-service/hub"`

## start
```sh
# (optional)
# export MODEL_WEIGHT="/your/model/hub/path"
# export PCM_SAVE_PATH="/your/pcm/save/path"
export AGENT_URL="/your/agent/service/url"
export ENABLE_SPEAKER_VERIFICATION="false"  # true or false
export SPEAKER_EMBEDDING_FILE="/your/speaker/embedding/h5file"
cd app
bash start_api_service.sh 2>&1 | tee -a webrtc-demo.log
```