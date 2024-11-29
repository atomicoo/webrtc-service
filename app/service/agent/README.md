# Agent demo
python3.11 is required 

```shell
conda create -n agent python=3.11
conda activate agent
pip3 install -r requirements.txt 
```

## Getting started

```sh

WEBRTC_SERVICE_URL="YourWebrtcServiceURL"
MEMORY_URL="YourMemoryServiceURL"

python agent/app.py --port 8000 --memory_url $MEMORY_URL | tee -a agent-demo.log
```