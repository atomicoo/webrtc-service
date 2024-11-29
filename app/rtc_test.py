#! encoding: utf-8
import time
import traceback
from conf.rtc_conf import APPID, APP_CERTIFICATE
from third_party.agora_python_rtc_sdk.agora_service.agora_service import AgoraServiceConfig, AgoraService, RTCConnConfig
from third_party.agora_python_rtc_sdk.agora_service.rtc_connection import *
from third_party.agora_python_rtc_sdk.agora_service.media_node_factory import *
from third_party.agora_python_rtc_sdk.agora_service.audio_pcm_data_sender import *
from third_party.agora_python_rtc_sdk.agora_service.audio_frame_observer import *
from third_party.agora_python_rtc_sdk.agora_dynamic_key.src.RtcTokenBuilder2 import *
from util.logger import logger
from threadsafedict import ThreadSafeDict


connection_cache = ThreadSafeDict()
sender_cache = ThreadSafeDict()
stream_id_cache = ThreadSafeDict()
user_login_cache = ThreadSafeDict()


def on_connected(agora_rtc_conn, conn_info, reason):
    print("Connected:", agora_rtc_conn, conn_info, reason)


def on_disconnected(agora_rtc_conn, conn_info, reason):
    print("Disconnected:", agora_rtc_conn, conn_info, reason)


def on_connecting(agora_rtc_conn, conn_info, reason):
    print("Connecting:", agora_rtc_conn, conn_info, reason)


def on_user_joined(agora_rtc_conn, user_id):
    user_id = user_id.decode('utf-8')
    logger.info(f"**** User joined: {agora_rtc_conn}, {user_id}")


def on_user_left(agora_rtc_conn, user_id, reason):
    user_id = user_id.decode('utf-8')
    logger.info(f"**** User left: {agora_rtc_conn,}, {user_id}, {reason}")


def on_playback_audio_frame_before_mixing(agora_local_user, channelId, uid, frame):
    user_id = uid.decode('utf-8')
    # print("on_playback_audio_frame_before_mixing", user_id)
    return 0


def on_record_audio_frame(agora_local_user, channelId, frame):
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


# c api def
# AGORA_HANDLE agora_local_user, user_id_t user_id, int stream_id, const char* data, size_t length
# user_id: string type; steream_id: int, data: byte, lenght: int
def on_stream_message(local_user, user_id, stream_id, data, length):
    logger.info(f"on_stream_message: {user_id}, {stream_id}, {data}, {length}")
    # print("on_stream_message: userid is {user_id}, streamid is {stream_id}, data is {data}, length is {length }")
    return 0

# void (*on_user_info_updated)(AGORA_HANDLE agora_local_user, user_id_t user_id, int msg, int val);
# user_id: string type; msg: int, val: int


def on_user_info_updated(local_user, user_id, msg, val):
    logger.info(f"on useroff: {user_id}, {msg}, {val}")
    # print("on_user_info_updated: userid is: {user_id}, msg is: {msg}, val is: {val}")
    return 0


config = AgoraServiceConfig()
config.enable_audio_processor = 1
config.enable_audio_device = 0
# config.enable_video = 1
config.appid = APPID
cur_dir = os.path.dirname(os.path.abspath(__file__))
config.log_path = os.path.join(cur_dir, 'agorasdk.log')

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


def connect(rtc_channel_id, client_user_id):

    TOKEN = RtcTokenBuilder.build_token_with_uid(APPID, APP_CERTIFICATE,
                                                 rtc_channel_id, client_user_id, Role_Subscriber,
                                                 3600, 3600)
    SERVER_UID = 100

    logger.info(
        f'connecting to channel: {rtc_channel_id}, user: {client_user_id}')
    connection = agora_service.NewConnection(con_config)
    conn_observer = RTCConnObserver(
        on_connected=ON_CONNECTED_CALLBACK(on_connected),
        on_disconnected=ON_DISCONNECTED_CALLBACK(on_disconnected),
        on_user_joined=ON_USER_JOINED_CALLBACK(on_user_joined),
        on_user_left=ON_USER_LEFT_CALLBACK(on_user_left),
    )
    # local userobserver
    localuser_observer = RTCLocalUserObserver(
        on_stream_message=ON_STREAM_MESSAGE_CALLBACK(on_stream_message),
        on_user_info_updated=ON_USER_INFO_UPDATED_CALLBACK(on_user_info_updated)
    )
    connection.RegisterObserver(conn_observer, localuser_observer)

    try:
        connection.Connect(TOKEN, rtc_channel_id, str(SERVER_UID))
        steam_id, ret = connection.CreateDataStream(False, False)
        logger.info(f'create data stream: {steam_id}')
        audio_pcm_data_sender = connection.NewPcmSender()
        audio_pcm_data_sender.SetSendBufferSize(320*2000)
        audio_pcm_data_sender.Start()
        logger.info(f'create audio pcm data sender with buffer size: {320*2000}')
        connection_cache.set(str(client_user_id), connection)
        stream_id_cache.set(str(client_user_id), steam_id)
        sender_cache.set(str(client_user_id), audio_pcm_data_sender)
        user_login_cache.set(str(client_user_id), int(time.time()))

        logger.info(
            f'connected to channel: {rtc_channel_id}, user: {client_user_id}')

        while True:
            user_login = user_login_cache.get(str(client_user_id))
            connection.SendStreamMessage(steam_id, 'hello')
            if not user_login:
                logger.info('user not login, quit connection')
                break
            if int(time.time()) - user_login > 300:
                logger.info('timeout, quit connection')
                break
            time.sleep(0.1)
    except Exception:
        traceback.print_exc()  # type: ignore
    finally:
        user_login_cache.remove(str(client_user_id))

        audio_pcm_data_sender.ClearSendBuffer()
        audio_pcm_data_sender.Stop()

        connection.Disconnect()
        connection.Release()

        connection_cache.remove(str(client_user_id))
        sender_cache.remove(str(client_user_id))
        stream_id_cache.remove(str(client_user_id))


connect('test123', 12345)
# agora_service.Destroy()
