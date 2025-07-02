# simple-terminal-talking
Simple voice-to-text for terminal work. Hold Windows key, speak, get typed commands.

## Requirements
- Python 3.7+
- PyAudio (requires system audio libraries)
- OpenAI Whisper
- pynput

## Setup
```bash
pip install openai-whisper pyaudio pynput
```

## Usage
```bash
python main.py
```
Hold Windows/Super key to record, release to transcribe and type.
