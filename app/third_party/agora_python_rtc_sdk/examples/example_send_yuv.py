#coding=utf-8


import time
import ctypes
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sdk_dir = os.path.dirname(script_dir)
if sdk_dir not in sys.path:
    sys.path.insert(0, sdk_dir)

from agora_service.agora_service import AgoraServiceConfig, AgoraService, AudioSubscriptionOptions, RTCConnConfig
from agora_service.rtc_connection import *
from agora_service.media_node_factory import *
from agora_service.audio_pcm_data_sender import *
from agora_service.audio_frame_observer import *
from agora_service.video_sender import *
from agora_service.video_frame_observer import *

# conn_observer callback
def on_connected(agora_rtc_conn, conn_info, reason):
    print("on_connected:", agora_rtc_conn, conn_info, reason)

def on_disconnected(agora_rtc_conn, conn_info, reason):
    print("on_disconnected:", agora_rtc_conn, conn_info, reason)

def on_connecting(agora_rtc_conn, conn_info, reason):
    print("on_connecting:", agora_rtc_conn, conn_info, reason)

def on_user_joined(agora_rtc_conn, user_id):
    print("on_user_joined:", agora_rtc_conn, user_id)

def on_playback_audio_frame_before_mixing(agora_local_user, channelId, uid, frame):
    # print("on_playback_audio_frame_before_mixing")#, channelId, uid)
    return 0

def on_record_audio_frame(agora_local_user ,channelId, frame):
    print("on_record_audio_frame")
    return 0

def on_playback_audio_frame(agora_local_user, channelId, frame):
    print("on_playback_audio_frame")
    return 0

def on_mixed_audio_frame(agora_local_user, channelId, frame):
    print("on_mixed_audio_frame")
    return 0

def on_ear_monitoring_audio_frame(agora_local_user, frame):
    print("on_ear_monitoring_audio_frame")
    return 0

def on_get_audio_frame_position(agora_local_user):
    print("on_get_audio_frame_position")
    return 0

def on_stream_message(local_user, user_id, stream_id, data, length):
    print("on_stream_message:", user_id, stream_id, data, length)
    return 0

def on_user_info_updated(local_user, user_id, msg, val):
    print("on_user_info_updated:", user_id, msg, val)
    return 0

def on_frame(agora_video_frame_observer2, channel_id, remote_uid, frame):
    print("on_frame")
    return 0

class Pacer:
    def __init__(self,interval):
        self.last_call_time = time.time()
        self.interval = interval

    def pace(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_call_time
        if elapsed_time < self.interval:
            time.sleep(self.interval - elapsed_time)
            # print("sleep time:", (self.interval - elapsed_time)*1000)
        self.last_call_time = time.time()


example_dir = os.path.dirname(os.path.abspath(__file__))


# 通过传参将参数传进来
# 例如： python examples/example.py {appid} {token} {channel_id} ./test_data/103_RaceHorses_416x240p30_300.yuv {userid}
appid = sys.argv[1]
token = sys.argv[2]
channel_id = sys.argv[3]
yuv_file_path = sys.argv[4]
# check argv len
if len(sys.argv) > 5:
    uid = sys.argv[5]
else:
    uid = "0"
print("appid:", appid, "token:", token, "channel_id:", channel_id, "yuv_file_path:", yuv_file_path, "uid:", uid)


config = AgoraServiceConfig()
config.enable_audio_processor = 0
config.enable_audio_device = 0
config.enable_video = 1
config.appid = appid
config.log_path = os.path.join(example_dir, 'agorasdk.log')

agora_service = AgoraService()
agora_service.Init(config)

con_config = RTCConnConfig(
    auto_subscribe_audio=1,
    auto_subscribe_video=0,
    client_role_type=1,
    channel_profile=1,
)

pcm_observer = AudioFrameObserver(
    on_record_audio_frame=ON_RECORD_AUDIO_FRAME_CALLBACK(on_record_audio_frame),
    on_playback_audio_frame=ON_PLAYBACK_AUDIO_FRAME_CALLBACK(on_playback_audio_frame),
    on_ear_monitoring_audio_frame=ON_EAR_MONITORING_AUDIO_FRAME_CALLBACK(on_ear_monitoring_audio_frame),
    on_playback_audio_frame_before_mixing=ON_PLAYBACK_AUDIO_FRAME_BEFORE_MIXING_CALLBACK(on_playback_audio_frame_before_mixing),
    on_get_audio_frame_position=ON_GET_AUDIO_FRAME_POSITION_CALLBACK(on_get_audio_frame_position),
)

con_config.pcm_observer = pcm_observer

connection = agora_service.NewConnection(con_config)

conn_observer = RTCConnObserver(
    on_connected=ON_CONNECTED_CALLBACK(on_connected),
    on_disconnected=ON_CONNECTED_CALLBACK(on_disconnected),
    on_user_joined=ON_USER_JOINED_CALLBACK(on_user_joined)
)
localuser_observer = RTCLocalUserObserver( 
    on_stream_message=ON_STREAM_MESSAGE_CALLBACK(on_stream_message),
    on_user_info_updated=ON_USER_INFO_UPDATED_CALLBACK(on_user_info_updated)
)

connection.RegisterObserver(conn_observer,localuser_observer)

connection.Connect(token, channel_id, uid)

video_sender = connection.GetVideoSender()
video_frame_observer = VideoFrameObserver2(
    on_frame=ON_FRAME_CALLBACK(on_frame)
)
# video_sender.register_video_frame_observer(video_frame_observer)

video_sender.Start()

sendinterval = 1/30
Pacer = Pacer(sendinterval)

width = 416
height = 240

def send_test():
    count = 0
    yuv_len = int(width*height*3/2)
    frame_buf = bytearray(yuv_len)            
    with open(yuv_file_path, "rb") as file:
        while True:            
            success = file.readinto(frame_buf)
            if not success:
                break
            frame = ExternalVideoFrame()
            frame.data = frame_buf
            frame.type = 1
            frame.format = 1
            frame.stride = width
            frame.height = height
            frame.timestamp = 0
            frame.metadata = "hello meta"
            ret = video_sender.SendVideoFrame(frame)        
            count += 1
            print("count,ret=",count, ret)
            Pacer.pace()

for i in range(30):
    send_test()

time.sleep(2)
video_sender.Stop()
connection.Disconnect()
connection.Release()
print("release")
agora_service.Destroy()
print("end")