# Voice-To-Voice Assistant with RAG (Retrieval Augmented Generation)

A comprehensive voice-to-voice conversational AI assistant specialized for dental consultations, featuring real-time speech-to-text, retrieval-augmented generation (RAG), large language model processing, and voice cloning for natural responses.

## 🔄 System Architecture & Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   🎤 User       │───▶│  📝 Speech-to-   │───▶│  🔍 RAG System      │
│   Voice Input   │    │     Text (STT)   │    │  (FAISS + Sentence  │
└─────────────────┘    └──────────────────┘    │   Transformers)     │
                                               └─────────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  🔊 Voice       │◀───│  🎯 Voice        │◀───│  🤖 LLM Processing  │
│  Response       │    │     Cloning      │    │  (Gemma 2B/4B or   │
│  (Cloned Voice) │    │     (TTS)        │    │   Gemini Flash)     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

### Complete Pipeline Flow:
1. **Speech-to-Text (STT)** → Google Assistant API transcribes user voice
2. **RAG Processing** → FAISS vector search finds relevant context from dental data  
3. **LLM Generation** → Gemma/Gemini generates contextual response
4. **Voice Cloning** → TTS with speaker reference creates natural voice response
5. **Audio Playback** → Response played back to user

## 📁 Project Structure

```
Voice-To-Voice-Assistant/
├── 🎯 Core Voice System
│   ├── Voice/
│   │   ├── main.py                    # Main entry point
│   │   ├── pushtotalk/
│   │   │   ├── pushtotalk.py         # Core voice assistant logic
│   │   │   ├── audio_helpers.py      # Audio processing utilities
│   │   │   ├── assistant_helpers.py  # Google Assistant API helpers
│   │   │   └── ...                   # Additional helper modules
│   │   ├── sample.wav               # Reference voice for cloning
│   │   ├── requirements.txt         # Voice system dependencies
│   │   ├── .env                     # Environment configuration
│   │   └── run.sh                   # Execution script
│   └── Backend/
│       ├── main.py                  # Backend entry point
│       ├── audiofileinput.py        # File-based audio processing
│       └── assistant_helpers.py     # Assistant API utilities
├── 🧠 LLM Models & Processing
│   ├── gamma.py                     # Gemma 3-4B CPU inference
│   ├── gamma_4b.py                  # Gemma 3-4B 4-bit quantized
│   ├── gamma_gpu.py                 # Gemma 3-4B 8-bit GPU inference
│   └── CLEAR.py                     # GPU memory cleanup utility
├── 🔍 RAG System
│   ├── RAG_with_gemma.py           # Complete RAG + Gemma pipeline
│   ├── rag.py                      # Standalone RAG search system
│   └── rag_formatted_data.txt      # Dental conversation dataset (2041 entries)
└── 📄 Documentation & Assets
    ├── README.md                   # This file
    └── Voice.zip                   # Complete voice system archive
```

## 🚀 Key Features

### 1. **Real-Time Voice Processing**
- **Speech-to-Text**: Google Assistant API for accurate voice transcription
- **Push-to-Talk**: Microphone activation system
- **Audio Processing**: 16kHz LINEAR16 format with real-time streaming

### 2. **Advanced RAG System**
- **Vector Database**: FAISS for efficient similarity search
- **Embeddings**: SentenceTransformer (`all-MiniLM-L6-v2`) for semantic understanding
- **Context Retrieval**: Top-k relevant responses from 2041 dental conversations
- **Emotional Context**: Responses include emotion and tone annotations

### 3. **Multi-Model LLM Support**
- **Gemma 3-4B IT**: Local inference with multiple configurations
  - CPU-only inference (`gamma.py`)
  - 4-bit quantized GPU (`gamma_4b.py`) 
  - 8-bit quantized GPU (`gamma_gpu.py`)
- **Gemini 2.0 Flash**: Cloud-based generation with faster responses
- **Quantization**: BitsAndBytesConfig for memory-efficient inference

### 4. **Voice Cloning & TTS**
- **YourTTS Model**: Multilingual text-to-speech synthesis
- **Speaker Cloning**: Reference voice (`sample.wav`) for consistent identity
- **Audio Enhancement**: Speed control and audio processing
- **Natural Responses**: Emotional tone preservation in generated speech

### 5. **Specialized Domain Knowledge**
- **Dental Expertise**: 2041 real dental consultation conversations
- **Context-Aware**: Maintains conversation history and emotional state
- **Professional Tone**: Responses formatted for medical consultations

## 🛠️ Installation & Setup

### Prerequisites
```bash
# System requirements
- Python 3.8+
- CUDA support (optional, for GPU inference)
- Audio devices (microphone & speakers)
- Google Cloud Project with Assistant API enabled
```

### 1. **Environment Setup**
```bash
# Clone or download the project
cd Voice-To-Voice-Assistant

# Install voice system dependencies  
cd Voice
pip install -r requirements.txt

# Install additional ML dependencies
pip install transformers torch torchaudio
pip install sentence-transformers faiss-cpu
pip install google-generativeai
```

### 2. **Google Assistant API Configuration**
```bash
# Install Google OAuth tool
pip install google-oauthlib-tool

# Initialize credentials
google-oauthlib-tool --client-secrets-file client_secrets.json \
                     --credentials-file credentials.json \
                     --scope https://www.googleapis.com/auth/assistant-sdk-prototype

# Register device (required for first time)
python -m pushtotalk.devicetool --project-id PROJECT_ID register-model \
       --manufacturer "Assistant SDK developer" \
       --product-name "Assistant SDK light" \
       --type LIGHT --model YOUR_MODEL_NAME
```

### 3. **Environment Variables**
Create/update `Voice/.env`:
```bash
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
```

### 4. **Voice Reference Setup**
Place a reference voice file as `Voice/sample.wav` (16kHz, mono recommended) for voice cloning.

## 🎯 Usage Examples

### 1. **Complete Voice Assistant**
```bash
cd Voice
python main.py
# System listens for voice → processes with RAG → responds with cloned voice
```

### 2. **Standalone RAG Testing**
```bash
# Test RAG retrieval only
python rag.py
# Query: "I had pain while eating something cold"
# Returns: Top 3 relevant responses with emotions/tones
```

### 3. **RAG + LLM Integration**
```python
from RAG_with_gemma import main

# Process query with full RAG + Gemma pipeline
response = main("My tooth hurts when I drink cold water")
print(response)
```

### 4. **Different LLM Configurations**
```bash
# CPU inference
python gamma.py

# 4-bit quantized (low memory)
python gamma_4b.py

# 8-bit GPU optimized
python gamma_gpu.py
```

### 5. **File-based Audio Processing**
```bash
cd Backend
python audiofileinput.py -i input.wav -o output.wav \
       --device-model-id YOUR_MODEL_ID \
       --device-id YOUR_DEVICE_ID
```

## ⚙️ Configuration Options

### Voice Assistant Settings
```python
# pushtotalk.py configuration
ASSISTANT_API_ENDPOINT = 'embeddedassistant.googleapis.com'
DEFAULT_GRPC_DEADLINE = 185  # 3 minutes + 5 seconds
audio_sample_rate = 16000    # 16kHz
audio_sample_width = 2       # 16-bit
```

### RAG System Parameters
```python
# RAG_with_gemma.py settings
embed_model = "all-MiniLM-L6-v2"     # Embedding model
top_k = 5                            # Number of retrieved contexts
max_new_tokens = 200                 # LLM response length
temperature = 0.7                    # Response randomness
```

### TTS Configuration
```python
# Voice cloning settings
tts_model = "tts_models/multilingual/multi-dataset/your_tts"
language = "en"
speed_multiplier = 1.0               # Playback speed control
```

## 📊 Dataset Information

### Dental RAG Dataset (`rag_formatted_data.txt`)
- **Size**: 2,041 conversation entries
- **Format**: Structured doctor-patient dialogues
- **Content**: Real dental consultations covering:
  - Tooth pain and sensitivity
  - Dental procedures (cleanings, fillings, root canals)
  - Patient history and symptoms
  - Treatment recommendations
  - Emotional support and reassurance

### Data Structure
```
=== Document ID: N ===
User: [Patient query/concern]
Response: [Doctor's professional response]
[Emotion: Inquiring/Explaining/Reassuring/etc.]
[Tone: Neutral/Gentle/Professional/etc.]
```

## 🔧 Troubleshooting

### Common Issues

1. **Audio Device Errors**
```bash
# Check available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Fix permissions (Linux)
sudo usermod -a -G audio $USER
```

2. **Google Assistant API Issues**
```bash
# Refresh credentials
google-oauthlib-tool --client-secrets-file client_secrets.json \
                     --credentials-file credentials.json \
                     --refresh-token

# Check device registration
python -m pushtotalk.devicetool list --project-id PROJECT_ID
```

3. **CUDA/GPU Memory Issues**
```python
# Use memory cleanup
python CLEAR.py

# Or use CPU-only mode
python gamma.py  # Uses CPU inference
```

4. **TTS Model Loading**
```bash
# Pre-download TTS model
python -c "from TTS.api import TTS; TTS(model_name='tts_models/multilingual/multi-dataset/your_tts')"
```

## 🏗️ Architecture Deep Dive

### 1. **Speech Recognition Pipeline**
```python
# Google Assistant API integration
conversation_stream.start_recording()  # Begin audio capture
assist_requests = gen_assist_requests() # Format for API
responses = assistant.Assist(requests)  # Send to Google
transcript = extract_speech_results()   # Get text results
```

### 2. **RAG Processing Chain**
```python
# Document parsing and indexing
documents = parse_documents("rag_formatted_data.txt")
embeddings = embed_model.encode(user_utterances)
index = faiss.IndexFlatIP(embeddings.shape[1])

# Query processing
query_embedding = embed_model.encode([user_query])
distances, indices = index.search(query_embedding, top_k)
context = format_retrieved_documents(documents[indices])
```

### 3. **LLM Response Generation**
```python
# Gemma inference
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = generator(context_prompt, max_new_tokens=200)

# Gemini inference  
gen_model = genai.GenerativeModel("gemini-2.0-flash-001")
response = gen_model.generate_content(context_prompt)
```

### 4. **Voice Synthesis Pipeline**
```python
# TTS with voice cloning
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
tts.tts_to_file(text=response_text, 
                language="en",
                speaker_wav="sample.wav",
                file_path=output_path)

# Audio processing and playback
audio = AudioSegment.from_wav(output_path)
faster_audio = audio._spawn(audio.raw_data, 
                           overrides={"frame_rate": int(audio.frame_rate * speed_multiplier)})
play(faster_audio)
```

## 🔒 Security & Privacy

### API Keys & Credentials
- Store credentials securely in `.env` files
- Use Google Cloud IAM for access control  
- Rotate API keys regularly

### Data Privacy
- Voice data processed locally when possible
- Conversations not stored permanently
- Compliance with healthcare data regulations

## 🚀 Performance Optimization

### Model Selection Guide
| Model | Memory Usage | Speed | Quality | Use Case |
|-------|-------------|--------|---------|-----------|
| Gemma CPU | ~8GB RAM | Slow | High | Development/Testing |
| Gemma 4-bit | ~4GB VRAM | Medium | High | Production (Low Memory) |  
| Gemma 8-bit | ~6GB VRAM | Fast | High | Production (Balanced) |
| Gemini Flash | ~0GB Local | Fastest | Highest | Cloud-based Production |

### Memory Management
```python
# Clear GPU memory between requests
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
gc.collect()
```

## 🤝 Contributing

### Adding New Models
1. Create new model file (e.g., `new_model.py`)
2. Follow existing quantization patterns
3. Test with RAG integration
4. Update documentation

### Expanding RAG Dataset
1. Format new conversations following existing structure
2. Update `rag_formatted_data.txt`
3. Rebuild FAISS index
4. Validate retrieval quality

### Voice Model Updates
1. Replace `sample.wav` with new reference voice
2. Test voice cloning quality
3. Adjust TTS parameters if needed

## 📈 Future Enhancements

- [ ] **Multi-language Support** - Extend to other languages
- [ ] **Real-time Streaming** - Continuous conversation mode  
- [ ] **Custom Fine-tuning** - Domain-specific model training
- [ ] **Voice Authentication** - Speaker verification
- [ ] **Conversation Memory** - Long-term context retention
- [ ] **Multi-modal Input** - Text + voice hybrid interface

## 📄 License

This project incorporates components with various licenses:
- Google Assistant SDK components: Apache License 2.0
- Transformers/TTS components: Apache License 2.0  
- Custom RAG implementation: MIT License

## 🙏 Acknowledgments

- Google Assistant SDK for speech recognition
- Hugging Face Transformers for LLM support
- Coqui TTS for voice synthesis
- FAISS for efficient vector search
- Sentence Transformers for embeddings

---

**Note**: This system requires proper Google Cloud setup and API credentials. Ensure you have the necessary permissions and quotas configured for production use.

For technical support or feature requests, please refer to the individual component documentation or create an issue in the project repository.