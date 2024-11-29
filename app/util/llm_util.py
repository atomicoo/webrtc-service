import time
import requests


def process_single(img_url, prompt, url="https://api.lingyiwanwu.com/v1/chat/completions", img_detail="high", stream=False):
    API_KEY = "Bearer multimodel-peter"
    headers = {
        'Authorization': API_KEY,
        'Content-Type': 'application/json'
    }
    messages = [{'role': 'user', 'content': []}]
    
    messages[0]['content'].append({"type": "text", "text": f"{prompt}"})
    if img_url:
        if type(img_url) == str:
            img_url = [img_url]
        for u in img_url:
            messages[0]['content'].append({"type": "image_url", "image_url": {"url": f"{u}", "detail": img_detail}})
    data = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 2048,
        "stream": stream,
        "temperature": 0.3,
    }
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(url, json=data, headers=headers, stream=stream, timeout=180)
            # print("HTTP 状态码:", response.status_code)
            if response.status_code == 200:
                if stream:
                    return response
                else:
                    response = response.json()
                    text = response['choices'][0]['message']['content']
                    # print(f"response: {text}\n")
                    return text
            else:
                print("Error:", response.status_code, response.text)
                retry_count += 1
                time.sleep(2)
                print(f"Retrying... Attempt {retry_count}")
        except Exception as e:
            print(f"Error: {e}")
            retry_count += 1
            time.sleep(2)
            print(f"Retrying... Attempt {retry_count}")
    return 'error'
