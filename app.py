from flask import Flask, request, jsonify, Response
import requests
import json
import time
import os

app = Flask(__name__)

# Configuration from environment variables
NVIDIA_NIM_BASE_URL = os.getenv('NVIDIA_NIM_BASE_URL', 'https://integrate.api.nvidia.com/v1')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY', '')
PORT = int(os.getenv('PORT', 5000))

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "OpenAI to NVIDIA NIM Proxy",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    })

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        
        # Extract OpenAI format parameters
        messages = data.get('messages', [])
        model = data.get('model', 'meta/llama-3.1-8b-instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        # Convert to NVIDIA NIM format
        nim_payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add optional parameters if present
        if 'top_p' in data:
            nim_payload['top_p'] = data['top_p']
        if 'frequency_penalty' in data:
            nim_payload['frequency_penalty'] = data['frequency_penalty']
        if 'presence_penalty' in data:
            nim_payload['presence_penalty'] = data['presence_penalty']
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Make request to NVIDIA NIM
        if stream:
            return handle_streaming_response(nim_payload, headers)
        else:
            return handle_non_streaming_response(nim_payload, headers)
            
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "proxy_error",
                "code": 500
            }
        }), 500

def handle_non_streaming_response(payload, headers):
    response = requests.post(
        f"{NVIDIA_NIM_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )
    
    if response.status_code != 200:
        return jsonify({
            "error": {
                "message": response.text,
                "type": "nvidia_api_error",
                "code": response.status_code
            }
        }), response.status_code
    
    return jsonify(response.json())

def handle_streaming_response(payload, headers):
    def generate():
        try:
            with requests.post(
                f"{NVIDIA_NIM_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=120
            ) as response:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            yield decoded_line + '\n\n'
        except Exception as e:
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{NVIDIA_NIM_BASE_URL}/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "object": "list",
                "data": [
                    {
                        "id": "meta/llama-3.1-8b-instruct",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "nvidia"
                    }
                ]
            })
    except Exception as e:
        return jsonify({
            "object": "list",
            "data": []
        })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "nvidia_api_configured": bool(NVIDIA_API_KEY)
    })

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

if __name__ == '__main__':
    if not NVIDIA_API_KEY:
        print("WARNING: NVIDIA_API_KEY environment variable not set!")
    print(f"Starting OpenAI to NVIDIA NIM proxy server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT)
