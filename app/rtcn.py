#! encoding: utf-8
import threading
import time
import ctypes
import json
import uuid
import traceback
import h5py
from copy import deepcopy
from collections import deque
from functools import partial
from conf.model_conf import PCM_SAVE_PATH, MODEL_NAME
from conf.rtc_conf import APPID, SERVER_UID, APP_CERTIFICATE
from model.rtc_session import RtcSession, SessionState
from service.vad_silero_service import VadSileroService
from service.dns_dfn2_service import dns_dfn2_service
# from service.uvr_mdx_service import uvr_mdx_service
# from service.ss_moss2_service import ss_moss2_service
from service.sv_campp_service import sv_campp_service
from service.asr_sense_service import asr_sense_service
# from service.asr_whisper_service import asr_whisper_service
from service.llm_yi_service import llm_yi_service
from service.tts_cosy_service import tts_cosy_service
from third_party.agora_python_rtc_sdk.agora_service.agora_service import AgoraServiceConfig, AgoraService, RTCConnConfig
from third_party.agora_python_rtc_sdk.agora_service.rtc_connection import *
from third_party.agora_python_rtc_sdk.agora_service.media_node_factory import *
from third_party.agora_python_rtc_sdk.agora_service.audio_pcm_data_sender import *
from third_party.agora_python_rtc_sdk.agora_service.audio_frame_observer import *
from third_party.agora_python_rtc_sdk.agora_dynamic_key.src.RtcTokenBuilder2 import *
from util.audio_util import calc_audio_volume, wave_header_chunk
from util.io_util import load_embedding_h5file, save_embedding_h5file
from util.logger import logger
from util.string_util import StreamingTextProcessor, is_meaningful_string
from threadsafedict import ThreadSafeDict


session_cache = ThreadSafeDict()
vad_event_cache = ThreadSafeDict()
user_login_cache = ThreadSafeDict()

connection_cache = ThreadSafeDict()
sender_cache = ThreadSafeDict()
stream_id_cache = ThreadSafeDict()

IMGS_TEMPLATE = {
    'type': 'image_url',
    'image_url': {'url': '/fake_image_url_to_placeholder', 'detail': None}
}

MAX_SILENCE_GAP_CHUNKS = 40  # 最大静音间隔，320毫秒（10 * (512 / 16000) * 1000 = 320）
# MAX_SILENCE_GAP_SECOND = 0.96  # 单位：秒
SPEAKER_VERIFICATION_THRESHOLD = 0.40  # 声纹验证阈值
SPEAKER_VERIFICATION_CHUNK_SIZE = 64000  # 声纹验证音频片段大小


def get_token(channel_name: str, user_id: int):
    token_expiration_in_seconds = 3600
    privilege_expiration_in_seconds = 3600
    token = RtcTokenBuilder.build_token_with_uid(APPID, APP_CERTIFICATE, channel_name, user_id, Role_Subscriber,
                                                 token_expiration_in_seconds, privilege_expiration_in_seconds)
    return token


def send_server_message(user_id: str, message: str, **kwargs):
    connection = connection_cache.get(str(user_id), None)
    if not connection:
        logger.error(f'connection not found: {user_id}')
        return

    stream_id = stream_id_cache.get(str(user_id), None)
    if stream_id is not None:
        content = {'role': 'server', 'content': message}
        content.update(kwargs)
        send_content = json.dumps(content, ensure_ascii=False)
        connection.SendStreamMessage(stream_id, send_content)
        logger.info(f'send server info: {send_content}')
    else:
        logger.error(f'stream id not found: {user_id}')


def send_audio_via_rtc(user_id: int):
    '''
    发送音频帧到 RTC
    '''
    try:

        user_session = session_cache.get(str(user_id), None)
        if not user_session:
            logger.error(f'user session not found: {user_id}')
            return
        audio_data_sender = sender_cache.get(str(user_id), None)
        if not audio_data_sender:
            logger.error(f'audio pcm data sender not found: {user_id}')
            return
        logger.info('rtc send thread start')

        cached_output_pcm = []
        is_first_synthesis = True

        frame = PcmAudioFrame()
        frame.data = bytearray()  # type: ignore
        frame.timestamp = 0
        frame.samples_per_channel = 0
        frame.bytes_per_sample = 2
        frame.number_of_channels = 1
        frame.sample_rate = 16000

        while user_session and user_login_cache.get(str(user_id), None):
            if user_session.state == SessionState.FINISHED:
                raise SystemExit('state error (finish), stop send audio.')
            if user_session.state in (SessionState.LLMING, SessionState.LLMED):
                break
            time.sleep(0.01)

        sendinterval = 0.05  # unit: s
        # packnum = int((sendinterval * 1000) / 10)  # 10ms per pack
        sessionstarttick = int(time.time() * 1000)  # round to ms
        cursendtotalpack = 0

        while user_session and user_session.state in (SessionState.LLMING,
                                                      SessionState.LLMED,
                                                      SessionState.SYNTHESIZING,
                                                      SessionState.SYNTHESIZED,
                                                      SessionState.FINISHED):
            if user_session.state == SessionState.FINISHED:
                raise SystemExit('state error (finish), stop send audio.')

            curtime = int(time.time() * 1000)
            checkinterval = curtime - sessionstarttick
            needcompensationpack = int(checkinterval / 10) - cursendtotalpack
            # logger.info(f"needcompensationpack: {needcompensationpack}")

            if needcompensationpack > 0:
                donecompensationpack = 0
                while donecompensationpack < needcompensationpack:
                    if user_session.state == SessionState.FINISHED:
                        raise SystemExit('state error (finish), stop send audio.')

                    slide = user_session.tts_audio_buffer.popleft() if len(
                        user_session.tts_audio_buffer) > 0 else None

                    if not slide and user_session.state == SessionState.SYNTHESIZED:
                        logger.info('synth audio send finished')
                        user_session.set_state(SessionState.SERVEROUTPUTED, SessionState.SYNTHESIZED)
                        break

                    if not slide:
                        # logger.info(f'no slide to send, {user_session.state}')
                        time.sleep(0.01)
                        continue

                    frame.data = slide
                    frame.samples_per_channel = len(slide) // 2
                    audio_data_sender.SendPcmData(frame)
                    # time.sleep(0.01)
                    cached_output_pcm.append(slide)
                    # logger.info(f'send {len(slide)} bytes')
                    if is_first_synthesis:
                        logger.info(
                            f' ++ first synthesis sent ++, length: {len(slide)}')  # 为了统计首包发送时间
                        is_first_synthesis = False
                    donecompensationpack += 1

                cursendtotalpack += donecompensationpack

                if user_session.state == SessionState.SERVEROUTPUTED:
                    logger.info('synth audio send finished')
                    break

            time.sleep(sendinterval)

        logger.info('rtc send thread finished')
        user_session.set_state(SessionState.FINISHED, SessionState.SERVEROUTPUTED)

        # session_cache[str(user_id)] = RtcSession(user_id=user_id)
        user_session.reset()  # reset user session

        logger.info('send finish, finish the last chunk')
    except SystemExit:
        logger.error('system exit, stop send audio.')
    except Exception:
        traceback.print_exc()  # type: ignore
        # user_session.set_state(SessionState.FINISHED)  # 清理状态
        user_session.reset()  # reset user session
    finally:
        logger.info('finish or interrupt, stop send audio.')


def synthesize_stream(user_id: int, text, is_final_chunk=False):
    '''
    通过传入的文本进行 TTS 并回传音频流
    '''
    user_session = session_cache.get(str(user_id), None)
    if not user_session:
        raise ValueError(f'user session not found: {user_id}')

    def save_synthesis(chunk: bytes, finished: bool):
        user_session = session_cache.get(str(user_id), None)
        if not user_session:
            raise ValueError(f'user session not found: {user_id}')
        if user_session.state == SessionState.FINISHED:
            raise SystemExit('state error (finish), stop synthesis.')

        # logger.info(f'put to cache {len(chunk)} bytes')
        if len(chunk) > 0:
            if len(user_session.rtc_rest_slide) > 0:
                chunk = user_session.rtc_rest_slide + chunk
                user_session.rtc_rest_slide = bytes()
            for i in range(0, len(chunk), user_session.rtc_slide_size):
                slide = chunk[i:i + user_session.rtc_slide_size]
                if len(slide) < user_session.rtc_slide_size and not is_final_chunk:
                    user_session.rtc_rest_slide = slide
                    break
                writeable_slide = bytearray(slide)

                user_session.tts_audio_buffer.append(writeable_slide)
            # chunk = chunk.ljust(user_session.rtc_slide_size, b'\x00')
            # user_session.tts_audio_buffer.append(bytearray(chunk))
        if finished:
            logger.info(
                f'server put audio finished, is_final_chunk: {is_final_chunk}, state: {user_session.state}')
        if finished and is_final_chunk:
            if user_session.state == SessionState.SYNTHESIZING:
                user_session.set_state(SessionState.SYNTHESIZED, SessionState.SYNTHESIZING)

    logger.info(f'start to synthesize: {text}')
    # 调用 TTS 服务合成音频
    tts_cosy_service.synthesis_stream_with_callback(text, save_synthesis)


def start_audio_receive_thread(user_id: int):
    user_session = session_cache.get(str(user_id), None)
    if not user_session:
        logger.error(f'user session not found: {user_id}')
        return
    # 启动异步线程进行识别
    thread = threading.Thread(
        target=core_function_thread, args=(user_id,))
    thread.setDaemon(True)
    thread.start()
    # 将线程放入用户会话缓存
    logger.info(f'cache core thread: {thread.name}')
    user_session.thread_cache.append(thread)


def start_audio_send_thread(user_id: int):
    user_session = session_cache.get(str(user_id), None)
    if not user_session:
        logger.error(f'user session not found: {user_id}')
        return
    # 启动异步线程通过 rtc 发送音频片段
    thread = threading.Thread(
        target=send_audio_via_rtc, args=(user_id,))
    thread.setDaemon(True)
    thread.start()
    # 将线程放入用户会话缓存
    logger.info(f'cache send thread: {thread.name}')
    user_session.thread_cache.append(thread)


def finish_audio_receive(user_id: int):
    logger.info('timer finish, finish the last chunk')
    user_session = session_cache.get(str(user_id), None)
    if not user_session:
        logger.error(f'user session not found: {user_id}')
        return

    user_session.set_state(SessionState.CLINETINPUTED, SessionState.RECOGNIZING)


def core_function_thread(user_id: int):
    recognise_thread_sensevoice(user_id)
    if check_whether_to_skipout(user_id):
        pass  # logger.info('skip out according to session state')
    else:
        start_llm_thread(user_id)
        start_audio_send_thread(user_id)
        synthesize_thread_cosyvoice(user_id)


def check_whether_to_skipout(user_id: int):
    user_session = session_cache.get(str(user_id), None)
    if not user_session:
        logger.error(f'user session not found: {user_id}')
        return True
    if user_session.state == SessionState.FINISHED:
        logger.info('session state is finished, skip llm and tts')
        user_session.reset()  # reset user session
        return True
    if user_session.asr_total_text.strip() == '':
        logger.info('recognised text is empty, skip llm and tts')
        user_session.reset()  # reset user session
        return True
    return False


def start_llm_thread(user_id: int):
    user_session = session_cache.get(str(user_id), None)
    if not user_session:
        logger.error(f'user session not found: {user_id}')
        return
    # 启动异步线程进行 llm
    thread = threading.Thread(
        target=llm_thread_with_datastream, args=(user_id,))
    thread.setDaemon(True)
    thread.start()
    # 将线程放入用户会话缓存
    logger.info(f'cache llm thread: {thread.name}')
    user_session.thread_cache.append(thread)


def callback_for_interrupt(user_id: int, task_name: str):
    user_session = session_cache.get(str(user_id), None)
    if not user_session:
        raise ValueError(f'user session not found: {user_id}')
    if user_session.state == SessionState.FINISHED:
        raise SystemExit(f'state error (finish), stop {task_name}.')


def llm_thread_with_datastream(user_id: int):
    '''
    通过传入的文本进行 LLM 并回传回复文本
    '''
    try:

        user_session = session_cache.get(str(user_id), None)
        if not user_session:
            logger.error(f'user session not found: {user_id}')
            return

        logger.info(f'start to llm: {user_session.asr_total_text}')

        chat_content = [{'role': 'user', 'content': user_session.asr_total_text}]

        image_prompt_cache = {'role': 'user', 'content': []}
        timestamp = int(time.time())
        while int(time.time()) - timestamp < 10:
            logger.info(f'waiting for image frames')
            if not user_session.llm_image_buffer:
                time.sleep(0.01)
                continue
            if user_session.llm_image_buffer[-1]['timestamp'] < user_session.llm_image_time:
                time.sleep(0.01)
                continue
            logger.info(f'inserting image frames')
            logger.info(f'{user_session.llm_image_buffer[-1]["timestamp"]} {user_session.llm_image_time}')
            image_prompt = deepcopy(IMGS_TEMPLATE)
            image_prompt['image_url']['url'] = user_session.llm_image_buffer[-1]['image']
            image_prompt_cache['content'].append(image_prompt)
            break
        if image_prompt_cache['content'] != []:
            chat_content.insert(-1, image_prompt_cache)

        send_server_message(user_id, 'start llm...')

        logger.info(f'llm agent with chat_content : {chat_content}')
        generator = llm_yi_service.chat_completion_trunk(
            MODEL_NAME, messages=chat_content, user_id=user_id,
            callback=partial(callback_for_interrupt, user_id, 'llm'))

        message_id = str(uuid.uuid1())

        connection = connection_cache.get(str(user_id), None)
        if not connection:
            logger.error(f'connection not found: {user_id}')
            return
        stream_id = stream_id_cache.get(str(user_id), None)

        delta_content = ''
        for index, chunk in enumerate(generator, start=0):  # type: ignore
            if user_session.state == SessionState.FINISHED:
                raise SystemExit('state error (finish), stop llm.')
            
            if index == 0:
                user_session.set_state(SessionState.LLMING, SessionState.RECOGNIZED)

            # logger.info(f'llm chunk: {chunk}')
            if not chunk.lastOne:
                delta_content += chunk.choices[0].delta.content  # 更新增量文本
                user_session.llm_total_text = str(chunk.content)  # type: ignore

            # 发送文本内容到 RTC
            if index % 5 == 0 or chunk.lastOne:
                if stream_id is not None:
                    content = {'role': 'assistant', 'content': delta_content,
                               'index': index, 'message_id': message_id}
                    send_content = json.dumps(content, ensure_ascii=False)
                    connection.SendStreamMessage(stream_id, send_content)
                    # logger.info(f'send llm content: {send_content}')
                else:
                    logger.error(f'stream id not found: {user_id}')
                delta_content = ''  # 清空增量文本
                time.sleep(0.1)  # 避免发送过快导致的消息丢失

        logger.info(f'llm total text: {user_session.llm_total_text}')
        user_session.set_state(SessionState.LLMED, SessionState.LLMING)
        logger.info('llm finish, finish the last chunk')

    except SystemExit:
        logger.error('system exit, stop llm.')
    except Exception:
        traceback.print_exc()  # type: ignore
        send_server_message(user_id, 'error llm...')
        time.sleep(0.1)
        user_session.set_state(SessionState.LLMED)  # 清理状态
    finally:
        logger.info('finish or interrupt, stop llm.')


def synthesize_thread_cosyvoice(user_id: int):
    '''
    通过 LLM 的回复文本进行 TTS
    '''
    try:

        user_session = session_cache.get(str(user_id), None)
        if not user_session:
            logger.error(f'user session not found: {user_id}')
            return

        connection = connection_cache.get(str(user_id), None)
        if not connection:
            logger.error(f'connection not found: {user_id}')
            return

        while user_session and user_login_cache.get(str(user_id), None):
            if user_session.state == SessionState.FINISHED:
                raise SystemExit('state error (finish), stop synthesis.')
            if user_session.state in (SessionState.LLMING, SessionState.LLMED):
                break
            time.sleep(0.01)

        send_server_message(user_id, 'start tts...')

        streaming_text_processor = StreamingTextProcessor()
        while user_session and user_session.state in (SessionState.LLMING,
                                                      SessionState.LLMED,
                                                      SessionState.FINISHED):
            if user_session.state == SessionState.FINISHED:
                raise SystemExit('state error (finish), stop synthesis.')

            if user_session.state == SessionState.LLMED:
                user_session.set_state(SessionState.SYNTHESIZING, SessionState.LLMED)
                break

            text_chunk = user_session.llm_total_text[len(streaming_text_processor.raw_text_record):]
            next_punc_sentence = streaming_text_processor.filter_and_process(text_chunk, is_final_chunk=False)
            if next_punc_sentence == "":
                time.sleep(0.1)
                continue
            logger.info(f'next_punc_sentenc: {next_punc_sentence}')
            synthesize_stream(user_id, next_punc_sentence, is_final_chunk=False)
            user_session.llm_synth_text += next_punc_sentence

        text_chunk = user_session.llm_total_text[len(streaming_text_processor.raw_text_record):]
        for next_punc_sentence in streaming_text_processor.filter_and_process(text_chunk, is_final_chunk=True):
            if user_session.state == SessionState.FINISHED:
                raise SystemExit('state error (finish), stop synthesis.')
            logger.info(f'next_punc_sentenc: {next_punc_sentence}')
            synthesize_stream(user_id, next_punc_sentence, is_final_chunk=True)
            user_session.llm_synth_text += next_punc_sentence

        logger.info('synth finish, finish the last chunk')
    except SystemExit:
        logger.error('system exit, stop synthesis.')
    except Exception:
        traceback.print_exc()  # type: ignore
        send_server_message(user_id, 'error tts...')
        time.sleep(0.1)
        user_session.set_state(SessionState.SYNTHESIZED)  # 清理状态
    finally:
        logger.info('finish or interrupt, stop synthesis.')


def check_audio_validity(user_id: int, bytes_chunk):
    if len(bytes_chunk) < 40000:
        raise SystemExit(f'audio invalid: audio length ({len(bytes_chunk)}) is too short.')
    bytes_chunk = dns_dfn2_service.denoise(bytes_chunk)
    mean_db, peak_db = calc_audio_volume(bytes_chunk)
    if peak_db < 70.0:
        raise SystemExit(f'audio invalid: peak volume ({peak_db}) is too low.')
    if os.getenv('ENABLE_SPEAKER_VERIFICATION', 'false').lower() == 'true':
        # 读取声纹特征文件
        speaker_embedding_file = os.getenv('SPEAKER_EMBEDDING_FILE', 'speaker_embedding.h5')
        if not os.path.exists(speaker_embedding_file):
            logger.info('create speaker feature file.')
            with h5py.File(speaker_embedding_file, 'w') as f:
                pass  # 初始化声纹特征文件
        user_embedding = load_embedding_h5file(speaker_embedding_file, str(user_id))
        # 声纹识别：CAM++ 说话人确认
        embedding = sv_campp_service.embedding(bytes_chunk)
        if user_embedding is None:  # 无声纹注册的临时解决方案
            logger.info(f'audio received: save speaker feature first time.')
            save_embedding_h5file(speaker_embedding_file, str(user_id), embedding)
            return bytes_chunk
        similarity = sv_campp_service.verify(embedding, user_embedding)
        if similarity < SPEAKER_VERIFICATION_THRESHOLD:
            logger.error(f'audio invalid: speaker verification failed ({similarity}).')
            raise SystemExit(f'audio invalid: speaker verification failed ({similarity}).')
        else:
            logger.info(f'audio received: speaker verification succeed ({similarity}).')
            if similarity > 0.75:
                update_ratio = 0.25  # 声纹特征调整比例
                user_embedding = update_ratio * embedding + (1 - update_ratio) * user_embedding
                save_embedding_h5file(speaker_embedding_file, str(user_id), user_embedding)
    return bytes_chunk


def recognise_thread_sensevoice(user_id: int):
    '''
    通过传入的音频片段进行 ASR
    '''
    try:

        user_session = session_cache.get(str(user_id), None)
        if not user_session:
            logger.error(f'user session not found: {user_id}')
            return

        connection = connection_cache.get(str(user_id), None)
        if not connection:
            logger.error(f'connection not found: {user_id}')
            return

        select_chunk = [user_session.asr_audio_buffer.popleft(
        ) for _ in range(len(user_session.asr_audio_buffer))]
        bytes_chunk = bytes(select_chunk)
        logger.info(f'recongnise with chunk {len(bytes_chunk)}')

        # NOTE: 判断音频有效性：音频长度 -> 多人声分离 -> 峰值音量 -> 声纹识别
        start = time.perf_counter()
        bytes_chunk = check_audio_validity(user_id, bytes_chunk)
        logger.info(f'elapsed preproc time: {time.perf_counter() - start}s')

        if user_session.state == SessionState.FINISHED:
            raise SystemExit('state error (finish), stop recognise.')

        send_server_message(user_id, 'start asr...')
        # user_session.llm_image_time = int(time.time())

        user_session.set_state(SessionState.RECOGNIZING, SessionState.INIT)

        res = asr_sense_service.recognize(bytes_chunk)
        if not is_meaningful_string(res):
            raise SystemExit(f'recognised text ({res}) is not meaningful.')
        user_session.asr_total_text = res

        message_id = str(uuid.uuid1())
        stream_id = stream_id_cache.get(str(user_id), None)
        if stream_id is not None:
            content = {'role': 'user', 'content': user_session.asr_total_text.strip(),
                        'message_id': message_id}
            send_content = json.dumps(content, ensure_ascii=False)
            connection.SendStreamMessage(stream_id, send_content)
            logger.info(f'send asr content: {send_content}')
        else:
            logger.error(f'stream id not found: {user_id}')

        if user_session.state == SessionState.FINISHED:
            raise SystemExit('state error (finish), stop recognise.')

        user_session.set_state(SessionState.RECOGNIZED, SessionState.RECOGNIZING)
        logger.info('recognise finish, finish the last chunk')
        logger.info(f'recognised text: {user_session.asr_total_text}')

    except SystemExit:
        logger.error('system exit, stop recognise.')
    except Exception:
        traceback.print_exc()  # type: ignore
        send_server_message(user_id, 'error asr...')
        time.sleep(0.1)
        user_session.set_state(SessionState.FINISHED)  # 清理状态
    finally:
        logger.info('finish or interrupt, stop recognise.')


def save_frame_to_recog_buffer(user_id: int, frame):
    '''
    将音频帧推入缓冲区
    '''
    user_session = session_cache.get(str(user_id), None)
    if not user_session:
        logger.error(f'user session not found: {user_id}')
        return

    ptr = ctypes.c_void_p(frame.contents.buffer)
    data = ctypes.string_at(
        ptr, frame.contents.samples_per_channel * frame.contents.bytes_per_sample)

    # 取消打断机制：仅当状态为识别阶段时才处理音频片段
    # if user_session.state not in (SessionState.INIT, SessionState.RECOGNIZING):
    #     return
    if user_session.state == SessionState.FINISHED:
        return
    if user_session.state in (SessionState.RECOGNIZING, SessionState.RECOGNIZED):
        return

    vad_cache = vad_event_cache.get(str(user_id), None)
    if not vad_cache:
        logger.error(f'vad cache not found: {user_id}')
        return

    vad_cache['buffer'].extend(data)
    vad_cache['preact'].extend(data)

    if len(vad_cache['buffer']) >= 1024:  # 512 * 2 bytes
        # logger.info(f'vad buffer {len(vad_cache['buffer'])} samples')
        chunk = [vad_cache['buffer'].popleft() for _ in range(1024)]
        speech_prob, vad_event = vad_cache['vad'].detect_speech(bytes(chunk))
        # logger.info(f'vad event: {vad_event}, speech prob: {speech_prob}')

        if vad_event is None:
            if vad_cache['speech'] is True:
                user_session.asr_audio_buffer.extend(chunk)
                # logger.info(f'put to recog buffer {len(chunk)} samples')
            elif vad_cache['ttimer'] > 0:
                vad_cache['ttimer'] += 1
                if vad_cache['ttimer'] < MAX_SILENCE_GAP_CHUNKS:
                    user_session.asr_audio_buffer.extend(chunk)
                    # logger.info(f'put to recog buffer {len(chunk)} samples')
                else:
                    # 初始化用户VAD缓存
                    vad_cache['buffer'].clear()
                    vad_cache['preact'].clear()
                    vad_cache['ttimer'] = 0
                    vad_cache['vad'].reset_states()
                    # 触发/mute
                    # logger.info('auto mute, finish recognise.')
                    # finish_audio_receive(user_id)
                    start_audio_receive_thread(user_id)
                    # 触发/mute
        else:
            if 'start' in vad_event:
                if vad_cache['ttimer'] == 0:
                    # 触发/unmute
                    user_login_cache.set(str(user_id), int(time.time()))
                    logger.info('auto unmute, start recognise.')
                    start = time.perf_counter()
                    user_session.set_state(SessionState.FINISHED)
                    while user_session.thread_cache:
                        thread = user_session.thread_cache.pop()
                        if thread.is_alive():
                            thread.join()
                            logger.info(f"thread {thread.name} killed.")
                    logger.info(f'wait for all threads finished: {time.perf_counter() - start}')
                    user_session.reset()  # = RtcSession(user_id=user_id)
                    # start_audio_receive_thread(user_id)
                    # 触发/unmute
                    preact_size = len(vad_cache['preact'])
                    chunk += [vad_cache['preact'].popleft() for _ in range(preact_size)]
                    # 触发前端抽帧
                    send_server_message(user_id, 'start vad...')
                    user_session.llm_image_time = int(time.time())
                else:
                    vad_cache['ttimer'] = 0
                vad_cache['speech'] = True
                user_session.asr_audio_buffer.extend(chunk)
                logger.info(f'put to recog buffer {len(chunk)} samples')
            elif 'end' in vad_event:
                vad_cache['speech'] = False
                vad_cache['ttimer'] = 1
                user_session.asr_audio_buffer.extend(chunk)
                # logger.info(f'put to recog buffer {len(chunk)} samples')


def process_stream_message(user_id: int, message: str):
    user_session = session_cache.get(str(user_id), None)
    if not user_session:
        logger.error(f'user session not found: {user_id}')
        return

    if message.startswith(('frame::', 'gallery::')):
        logger.info(f"image message from {user_id}: {message}")
        image_url = message.split('::')[-1]
        user_session.llm_image_buffer.append({'image': image_url, 'timestamp': int(time.time())})


def agora_service_cls_to_dict(conn_info):
    '''
    将 Agora Service Class 实例转换为字典
    '''
    result = {}
    for field, _ in conn_info._fields_:
        value = getattr(conn_info, field)
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        result[field] = value
    return result


def on_connected(agora_rtc_conn, conn_info, reason):
    agora_conn_info = agora_service_cls_to_dict(conn_info.contents)
    logger.info(f"Connected: {agora_rtc_conn} {agora_conn_info} {reason}")


def on_disconnected(agora_rtc_conn, conn_info, reason):
    agora_conn_info = agora_service_cls_to_dict(conn_info.contents)
    logger.info(f"Disconnected: {agora_rtc_conn} {agora_conn_info} {reason}")


def on_connecting(agora_rtc_conn, conn_info, reason):
    agora_conn_info = agora_service_cls_to_dict(conn_info.contents)
    logger.info(f"Connecting: {agora_rtc_conn} {agora_conn_info} {reason}")


def on_user_joined(agora_rtc_conn, user_id):
    user_id = int(user_id.decode('utf-8'))
    logger.info(f"**** User joined: {agora_rtc_conn}, {user_id}")

    user_login_cache.set(str(user_id), int(time.time()))
    vad_event_cache.set(str(user_id), {  # 初始化用户的VAD缓存
        'buffer': deque(), 'preact': deque(maxlen=30720),
        'speech': False, 'ttimer': 0, 'vad': VadSileroService(None)})
    session_cache.set(str(user_id), RtcSession(user_id=user_id))


def on_user_left(agora_rtc_conn, user_id, reason):
    user_id = int(user_id.decode('utf-8'))
    logger.info(f"**** User left: {agora_rtc_conn,}, {user_id}, {reason}")

    try:
        session_cache.remove(str(user_id))
        vad_event_cache.remove(str(user_id))
        user_login_cache.remove(str(user_id))
    except Exception as e:
        logger.error(f'user left error: {e}')


def on_playback_audio_frame_before_mixing(agora_local_user, channelId, uid, frame):
    user_id = int(uid.decode('utf-8'))
    # logger.info("on_playbasave_frame_to_recog_bufferck_audio_frame_before_mixing", user_id)
    save_frame_to_recog_buffer(user_id, frame)
    return 0


def on_record_audio_frame(agora_local_user, channelId, frame):
    logger.info("on_record_audio_frame")
    return 0


def on_playback_audio_frame(agora_local_user, channelId, frame):
    logger.info("on_playback_audio_frame")
    return 0


def on_mixed_audio_frame(agora_local_user, channelId, frame):
    logger.info("on_mixed_audio_frame")
    return 0


def on_ear_monitoring_audio_frame(agora_local_user, frame):
    logger.info("on_ear_monitoring_audio_frame")
    return 0


def on_get_audio_frame_position(agora_local_user):
    logger.info("on_get_audio_frame_position")
    return 0


# c api def
# AGORA_HANDLE agora_local_user, user_id_t user_id, int stream_id, const char* data, size_t length
# user_id: string type; steream_id: int, data: byte, lenght: int
def on_stream_message(local_user, user_id, stream_id, data, length):
    user_id = int(user_id.decode('utf-8'))
    message = str(data.decode('utf-8'))
    # logger.info(f"on_stream_message: {user_id}, {stream_id}, {message}, {length}")
    process_stream_message(user_id, message)
    return 0

# void (*on_user_info_updated)(AGORA_HANDLE agora_local_user, user_id_t user_id, int msg, int val);
# user_id: string type; msg: int, val: int


def on_user_info_updated(local_user, user_id, msg, val):
    user_id = int(user_id.decode('utf-8'))
    logger.info(f"on useroff: {user_id}, {msg}, {val}")
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
    token = get_token(rtc_channel_id, SERVER_UID)

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
        connection.Connect(token, rtc_channel_id, str(SERVER_UID))
        stream_id, ret = connection.CreateDataStream(True, True)
        logger.info(f'create data stream: {stream_id}')
        audio_pcm_data_sender = connection.NewPcmSender()
        audio_pcm_data_sender.SetSendBufferSize(320*2000)
        audio_pcm_data_sender.Start()
        logger.info(f'create audio pcm data sender with buffer size: {320*2000}')
        connection_cache.set(str(client_user_id), connection)
        stream_id_cache.set(str(client_user_id), stream_id)
        sender_cache.set(str(client_user_id), audio_pcm_data_sender)
        user_login_cache.set(str(client_user_id), int(time.time()))

        logger.info(
            f'connected to channel: {rtc_channel_id}, user: {client_user_id}')

        while True:
            user_login = user_login_cache.get(str(client_user_id), None)
            if not user_login:
                send_server_message(client_user_id, 'quit connection for user not login')
                logger.info('quit connection for user not login')
                break
            if int(time.time()) - user_login > 300:
                send_server_message(client_user_id, 'quit connection for login timeout')
                logger.info('quit connection for login timeout')
                break
            time.sleep(0.1)
    except Exception:
        traceback.print_exc()  # type: ignore
    finally:
        start = time.perf_counter()
        logger.info(f'clean session cache: {client_user_id}')
        session_cache.remove(str(client_user_id))
        logger.info(f'clean vad cache: {client_user_id}')
        vad_event_cache.remove(str(client_user_id))
        logger.info(f'clean login cache: {client_user_id}')
        user_login_cache.remove(str(client_user_id))

        audio_pcm_data_sender.ClearSendBuffer()
        audio_pcm_data_sender.Stop()
        logger.info(f'disconnect: {rtc_channel_id}, {client_user_id}')
        connection.Disconnect()
        connection.Release()

        logger.info(f'clean connection cache: {client_user_id}')
        connection_cache.remove(str(client_user_id))
        logger.info(f'clean sender cache: {client_user_id}')
        sender_cache.remove(str(client_user_id))
        logger.info(f'clean stream id cache: {client_user_id}')
        stream_id_cache.remove(str(client_user_id))

        logger.info(f'estimated clean time: {time.perf_counter() - start}s')

# agora_service.Destroy()
