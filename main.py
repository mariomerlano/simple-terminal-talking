#!/usr/bin/env python3
"""
Simple Terminal Talking Script
Press and hold WINDOWS to record, release to transcribe and type
"""

import whisper
import pyaudio
import wave
import tempfile
import os
import threading
import time
import warnings
import signal
import sys
from pynput import keyboard
from pynput.keyboard import Key, Listener

# Suppress warnings
warnings.filterwarnings("ignore")

class WhisperPTT:
    def __init__(self):
        print("Loading Whisper model...")
        self.model = whisper.load_model("base")
        print("Whisper model loaded!")
        
        # Linux command replacements for better effectiveness
        self.command_replacements = {
            "pseudo": "sudo",
            "LS": "ls",
            "change directory": "cd",
            "make directory": "mkdir",
            "remove": "rm",
            "copy": "cp",
            "move": "mv",
            "grep": "grep",
            "pipe": "|",
            "greater than": ">",
            "append": ">>",
            "ampersand": "&",
            "dollar sign": "$",
            "dot": ".",
            "dot dot": "..",
            "slash": "/",
            "home": "~",
            "space dash": " -",
            "dash dash": "--"
        }
        
        self.recording = False
        self.audio_data = []
        self.audio_stream = None
        self.p = None
        self.record_thread = None
        self.processing = False  # Prevent overlapping operations
        self.last_action_time = 0  # Debouncing
        self.min_interval = 0.5  # Minimum time between recordings (seconds)
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
    def _init_audio(self):
        """Initialize PyAudio with error handling"""
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            print(f"Failed to initialize PyAudio: {e}")
            sys.exit(1)
    
    def start_recording(self):
        current_time = time.time()
        
        # Debouncing: ignore rapid key presses
        if current_time - self.last_action_time < self.min_interval:
            return
            
        # Don't start if already recording or processing
        if self.recording or self.processing:
            return
            
        print("Recording... (release WINDOWS key to transcribe)")
        self.recording = True
        self.audio_data = []
        self.last_action_time = current_time
        
        try:
            # Initialize fresh PyAudio for each recording to avoid corruption
            self.p = pyaudio.PyAudio()
            
            # Create new stream for each recording
            self.audio_stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            # Record in a separate thread
            self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.record_thread.start()
        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.recording = False
            self._cleanup_audio()
    
    def _record_audio(self):
        """Record audio with proper error handling"""
        try:
            while self.recording and self.audio_stream:
                try:
                    data = self.audio_stream.read(self.CHUNK, exception_on_overflow=False)
                    if data:
                        self.audio_data.append(data)
                except Exception as e:
                    print(f"Audio read error: {e}")
                    break
        except Exception as e:
            print(f"Recording thread error: {e}")
    
    def stop_recording(self):
        # Don't stop if not recording or already processing
        if not self.recording or self.processing:
            return
            
        print("Stopping recording...")
        self.recording = False
        self.processing = True  # Block new recordings during processing
        
        # Wait for recording thread to finish
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=1.0)
        
        # Clean up audio resources immediately
        self._cleanup_audio()
        
        # Process the recording in a separate thread to avoid blocking
        processing_thread = threading.Thread(target=self._process_recording, daemon=True)
        processing_thread.start()
    
    def _process_recording(self):
        """Process recording in separate thread"""
        try:
            self._transcribe_and_type()
        finally:
            self.processing = False  # Allow new recordings
    
    def _cleanup_audio(self):
        """Clean up audio resources completely"""
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            except Exception as e:
                pass
                
        if self.p:
            try:
                self.p.terminate()
                self.p = None
            except Exception as e:
                pass
        
        # Small delay to let audio system stabilize
        time.sleep(0.1)
    
    def _transcribe_and_type(self):
        if not self.audio_data:
            print("No audio recorded")
            return
        
        # Check for minimum recording length
        if len(self.audio_data) < 10:  # Very short recording
            print("Recording too short, skipping...")
            return
            
        # Save audio to temporary file
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            
            wf = wave.open(temp_file.name, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.audio_data))
            wf.close()
            
            print("Transcribing...")
            
            # Transcribe with Whisper
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.model.transcribe(
                    temp_file.name,
                    initial_prompt="Linux terminal commands: sudo ls cd mkdir rm cp mv grep cat chmod",
                    fp16=False  # Disable FP16 to avoid CPU warning
                )
            
            text = result["text"].strip()
            
            # Filter out repeated patterns (sign of corrupted audio)
            if self._is_repetitive_text(text):
                print("Detected corrupted audio, skipping...")
                return
            
            if text:
                # Apply command replacements
                for spoken, actual in self.command_replacements.items():
                    text = text.replace(spoken, actual)
                
                print(f"Transcribed: {text}")
                self._type_text(text)
            else:
                print("No speech detected")
                
        except Exception as e:
            print(f"Error transcribing: {e}")
        finally:
            # Clean up temp file
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
    
    def _is_repetitive_text(self, text):
        """Check if text contains repetitive patterns indicating corrupted audio"""
        words = text.split()
        if len(words) < 6:
            return False
        
        # Check for same word repeated many times
        for word in set(words):
            if words.count(word) > len(words) * 0.7:  # If one word is >70% of text
                return True
        
        # Check for repeating patterns
        if len(words) > 10:
            pattern_length = 3
            for i in range(len(words) - pattern_length * 3):
                pattern = words[i:i+pattern_length]
                next_pattern = words[i+pattern_length:i+pattern_length*2]
                if pattern == next_pattern:
                    return True
        
        return False
    
    def _type_text(self, text):
        """Type the text using pynput (cross-platform)"""
        try:
            from pynput.keyboard import Controller
            keyboard_controller = Controller()
            
            # Small delay to ensure focus
            time.sleep(0.1)
            keyboard_controller.type(text)
        except Exception as e:
            print(f"Error typing text: {e}")
            print(f"Text to type: {text}")
    
    def on_press(self, key):
        if key == Key.cmd or key == Key.cmd_r:  # Windows/Super key (left or right)
            self.start_recording()
    
    def on_release(self, key):
        if key == Key.cmd or key == Key.cmd_r:  # Windows/Super key (left or right)
            self.stop_recording()
        elif key == Key.esc:
            print("Exiting...")
            return False
    
    def start_listening(self):
        print("Simple Terminal Talking Ready!")
        print("Hold WINDOWS/SUPER key to record, release to transcribe")
        print("Press ESC to exit")
        
        try:
            with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
                listener.join()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        except Exception as e:
            print(f"Listener error: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.recording = False
        self._cleanup_audio()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutting down...")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    ptt = None
    try:
        ptt = WhisperPTT()
        ptt.start_listening()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if ptt:
            ptt.cleanup()
