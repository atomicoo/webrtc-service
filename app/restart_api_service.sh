eval "$(conda shell.bash hook)"

conda activate rtc
export AGENT_PORT=8000
export MEMORY_URL=''
export CONTEXT_CHECKPOINTER_PATH=''
export LD_LIBRARY_PATH=/root/FunASR/rtc-service/app/third_party/agora_python_rtc_sdk/agora_sdk

ps -ef | grep api_service.py | grep -v 'color' | grep -v 'grep' | awk '{print $2}' | xargs kill -9

nohup python api_service.py > api_service.log 2>&1 &

nohup python service/agent/agent/app.py --port $AGENT_PORT > api_service.log 2>&1 &
#--memory_url $MEMORY_URL --context_checkpointer_path $CONTEXT_CHECKPOINTER_PATH