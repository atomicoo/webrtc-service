#coding=utf-8

# export LD_LIBRARY_PATH="third_party/agora_python_rtc_sdk/agora_sdk:$LD_LIBRARY_PATH"

import time
import os
import sys
import uuid
import hashlib
from conf.rtc_conf import APPID, APP_CERTIFICATE
from third_party.agora_python_rtc_sdk.agora_service.agora_service import AgoraServiceConfig, AgoraService, RTCConnConfig
from third_party.agora_python_rtc_sdk.agora_service.rtc_connection import *
from third_party.agora_python_rtc_sdk.agora_service.media_node_factory import *
from third_party.agora_python_rtc_sdk.agora_service.audio_pcm_data_sender import *
from third_party.agora_python_rtc_sdk.agora_service.audio_frame_observer import *
from third_party.agora_python_rtc_sdk.agora_dynamic_key.src.RtcTokenBuilder2 import *

def get_token(channel_name: str, user_id: int) -> str:
    token_expiration_in_seconds = 3600
    privilege_expiration_in_seconds = 3600
    token = RtcTokenBuilder.build_token_with_uid(APPID, APP_CERTIFICATE, channel_name, user_id, Role_Subscriber,
                                                 token_expiration_in_seconds, privilege_expiration_in_seconds)
    return token

def get_user_unique_id(username: str, max_value: int = sys.maxsize) -> int:
    hash_object = hashlib.sha256(username.encode())
    hex_digits = hash_object.hexdigest()
    unique_id = int(hex_digits, 16) % max_value
    return unique_id

# conn_observer callback
def on_connected(agora_rtc_conn, conn_info, reason):
    print("Connected:", agora_rtc_conn, conn_info, reason)

def on_disconnected(agora_rtc_conn, conn_info, reason):
    print("Disconnected:", agora_rtc_conn, conn_info, reason)

def on_connecting(agora_rtc_conn, conn_info, reason):
    print("Connecting:", agora_rtc_conn, conn_info, reason)

def on_user_joined(agora_rtc_conn, user_id):
    print("on_user_joined:", agora_rtc_conn, user_id)

def on_stream_message(local_user, user_id, stream_id, data, length):
    user_id = int(user_id.decode('utf-8'))
    message = str(data.decode('utf-8'))
    print("on_stream_message:", user_id, stream_id, message)    
    return 0

def on_user_info_updated(local_user, user_id, msg, val):
    print("on_user_info_updated:", user_id, msg, val)
    return 0


example_dir = os.path.dirname(os.path.abspath(__file__))

UID = 6666
CLIENT_USER_ID = get_user_unique_id('dagouzi', 10000)
CHANNEL = uuid.uuid3(uuid.NAMESPACE_OID, str(CLIENT_USER_ID)).hex
TOKEN = get_token(CHANNEL, UID)
print("appid:", APPID, "token:", TOKEN, "channel_id:", CHANNEL, "uid:", UID)


config = AgoraServiceConfig()
config.enable_audio_processor = 0
config.enable_audio_device = 0
# config.enable_video = 0
config.appid = APPID
config.log_path = os.path.join(example_dir, 'agorasdk.log')

agora_service = AgoraService()
agora_service.Init(config)

con_config = RTCConnConfig(
    auto_subscribe_audio=0,
    auto_subscribe_video=0,
    client_role_type=1,
    channel_profile=1,
)
con_config.pcm_observer = None

connection = agora_service.NewConnection(con_config)

conn_observer = RTCConnObserver(
    on_connected=ON_CONNECTED_CALLBACK(on_connected),
    on_disconnected=ON_CONNECTED_CALLBACK(on_disconnected),
    on_user_joined=ON_USER_JOINED_CALLBACK(on_user_joined)
)
#local userobserver
localuser_observer = RTCLocalUserObserver( 
    on_stream_message=ON_STREAM_MESSAGE_CALLBACK(on_stream_message),
    on_user_info_updated=ON_USER_INFO_UPDATED_CALLBACK(on_user_info_updated)
)

connection.RegisterObserver(conn_observer,localuser_observer)

connection.Connect(TOKEN, CHANNEL, str(UID))
stream_id,ret = connection.CreateDataStream(True, True)
print("stream_id:", stream_id, "ret:", ret)
for i in range(10):
    msg = f'message {i} for testing!'
    print("sendmsg:{} to:{}".format(msg, stream_id))
    connection.SendStreamMessage(stream_id, f"message::{msg}")
    time.sleep(2)

connection.Disconnect()
connection.Release()
print("release")
agora_service.Destroy()
print("end")