eval "$(conda shell.bash hook)"

conda activate rtc

export AGENT_PORT=8008
# export MEMORY_URL='http://180.184.134.254:7169'  # 公网
export MEMORY_URL='http://172.25.248.128:7169'  # VPC
export CONTEXT_CHECKPOINTER_PATH=''

ps -ef | grep service/agent/agent/app.py | grep -v 'color' | grep -v 'grep' > /dev/null
if [ $? -eq 0 ]; then
  ps -ef | grep service/agent/agent/app.py | grep -v 'color' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
fi

python service/agent/agent/app.py --port $AGENT_PORT --memory_url $MEMORY_URL
