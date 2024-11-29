eval "$(conda shell.bash hook)"

conda activate rtc

export LD_LIBRARY_PATH="third_party/agora_python_rtc_sdk/agora_sdk:$LD_LIBRARY_PATH"

ps -ef | grep api_service.py | grep -v 'color' | grep -v 'grep' > /dev/null
if [ $? -eq 0 ]; then
  ps -ef | grep api_service.py | grep -v 'color' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
fi

python api_service.py
