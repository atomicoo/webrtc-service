# Build sdk
## Required OS and go version
- supported linux version: 
  - Ubuntu 18.04 LTS and above
  - CentOS 7.0 and above
- python version:
  - python 3.7 above

## Prepare C version of agora rtc sdk
- download and unzip [agora_sdk.zip](https://share.weiyun.com/lfBp0bOE)
```
unzip agora_sdk.zip
```
- make **agora_sdk** directory in the same directory with **python_wrapper**
- there should be **libagora_rtc_sdk.so** and **include_c** in **agora_sdk** directory

# Test
- run ut on linux
```
export LD_LIBRARY_PATH=/path/to/agora_sdk
python python_wrapper/example.py
```

