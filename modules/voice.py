#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import os
import tempfile
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from TTS.api import TTS
from pydub import AudioSegment
import soundfile as sf
import warnings

class VoiceProcessor:
    """Voice processing with ASR (Whisper) and TTS (XTTS) capabilities"""
    
    def __init__(self, asr_model_name="openai/whisper-large-v3", 
                 tts_model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                 device=None, cache_dir=None):
        """
        Initialize the voice processor
        
        Args:
            asr_model_name (str): Name of ASR model to use
            tts_model_name (str): Name of TTS model to use
            device (str, optional): Device to run inference on ('cpu', 'cuda', etc.)
            cache_dir (str, optional): Directory to cache models
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        
        # Initialize ASR components
        self._init_asr(asr_model_name)
        
        # Initialize TTS components
        self._init_tts(tts_model_name)
        
        # Create a temporary directory for audio processing
        self.temp_dir = tempfile.mkdtemp(prefix="vibrio_voice_")
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def _init_asr(self, model_name):
        """Initialize ASR components using Whisper"""
        # Load model and processor with advanced configuration
        print(f"Loading ASR model: {model_name}...")
        
        self.asr_processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=self.cache_dir
        )
        self.asr_model.to(self.device)
        
        # Set up pipeline with optimal parameters for high-quality transcription
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            tokenizer=self.asr_processor.tokenizer,
            feature_extractor=self.asr_processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=30,
            batch_size=16,
            device=self.device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            return_timestamps=True,
            generate_kwargs={"condition_on_previous_text": True, "compression_ratio_threshold": 2.4}
        )
        print("ASR model loaded successfully")
    
    def _init_tts(self, model_name):
        """Initialize TTS components using XTTS v2"""
        print(f"Loading TTS model: {model_name}...")
        
        # Initialize TTS with XTTS v2
        self.tts = TTS(model_name=model_name, progress_bar=True)
        
        # Configure default TTS parameters for high-quality speech
        self.tts_config = {
            "temperature": 0.75,
            "length_penalty": 1.0,
            "repetition_penalty": 2.0,
            "top_k": 50,
            "top_p": 0.85,
        }
        
        print("TTS model loaded successfully")
    
    def transcribe(self, audio, language=None, prompt=None, word_timestamps=True):
        """
        Transcribe audio to text with advanced features
        
        Args:
            audio (np.ndarray/str): Audio data or path to audio file
            language (str, optional): Language code (e.g., 'en', 'fr', 'de')
            prompt (str, optional): Transcription prompt to guide recognition
            word_timestamps (bool): Whether to include word-level timestamps
            
        Returns:
            dict: Dict containing:
                'text': Transcribed text
                'segments': Timestamped segments with word-level detail
                'language': Detected or specified language
                'words': Word-level timestamps if requested
        """
        # Handle both file paths and numpy arrays
        if isinstance(audio, str):
            # Load audio file if path is provided
            if not os.path.exists(audio):
                raise FileNotFoundError(f"Audio file not found: {audio}")
            # We'll let the pipeline handle the file loading
        elif isinstance(audio, np.ndarray):
            # Save numpy array as temporary file
            temp_path = os.path.join(self.temp_dir, "temp_input.wav")
            sf.write(temp_path, audio, 16000)
            audio = temp_path
        
        # Set up parameters for high-quality transcription
        generate_kwargs = {
            "return_token_timestamps": word_timestamps,
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.4,
        }
        
        if language is not None:
            generate_kwargs["language"] = language
        if prompt is not None:
            generate_kwargs["prompt"] = prompt
        
        # Run transcription with detailed parameters
        result = self.asr_pipeline(
            audio,
            generate_kwargs=generate_kwargs,
            return_timestamps=True
        )
        
        # Extract word-level timestamps if available
        words = []
        if word_timestamps and "words" in result:
            words = result["words"]
        
        # Format comprehensive result
        output = {
            'text': result.get('text', ''),
            'segments': result.get('chunks', []),
            'language': language or result.get('language', None),
            'words': words
        }
        
        return output
    
    def speak(self, text, voice_samples=None, speaker_name=None, language=None, output_path=None):
        """
        Convert text to speech with high-quality voice synthesis
        
        Args:
            text (str): Text to convert to speech
            voice_samples (str/np.ndarray): Voice sample path or audio data for voice cloning
            speaker_name (str, optional): Name of predefined speaker (when not using voice_samples)
            language (str, optional): Language code (e.g., 'en', 'fr', 'de')
            output_path (str, optional): Path to save the output audio
            
        Returns:
            dict: Dict containing:
                'audio': Audio data as numpy array
                'sampling_rate': Sampling rate of the audio
                'output_path': Path to saved audio file if output_path was provided
        """
        # Prepare output path if needed
        if output_path is None:
            output_path = os.path.join(self.temp_dir, f"tts_output_{int(torch.rand(1)[0]*10000)}.wav")
        
        # Handle voice samples for cloning
        voice_sample_path = None
        if voice_samples is not None:
            if isinstance(voice_samples, str) and os.path.exists(voice_samples):
                # Use provided audio file directly
                voice_sample_path = voice_samples
            elif isinstance(voice_samples, np.ndarray):
                # Save numpy array as temporary file
                voice_sample_path = os.path.join(self.temp_dir, "voice_sample.wav")
                sf.write(voice_sample_path, voice_samples, 16000)
        
        # Set default language if not provided
        if language is None:
            language = "en"
        
        # Generate speech with either voice cloning or predefined speaker
        if voice_sample_path:
            # Clone voice from the provided sample
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=voice_sample_path,
                language=language,
                temperature=self.tts_config["temperature"],
                length_penalty=self.tts_config["length_penalty"],
                repetition_penalty=self.tts_config["repetition_penalty"],
                top_k=self.tts_config["top_k"],
                top_p=self.tts_config["top_p"]
            )
        elif speaker_name:
            # Use predefined speaker
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker=speaker_name,
                language=language,
                temperature=self.tts_config["temperature"],
                length_penalty=self.tts_config["length_penalty"],
                repetition_penalty=self.tts_config["repetition_penalty"],
                top_k=self.tts_config["top_k"],
                top_p=self.tts_config["top_p"]
            )
        else:
            # Use default speaker
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                language=language,
                temperature=self.tts_config["temperature"],
                length_penalty=self.tts_config["length_penalty"],
                repetition_penalty=self.tts_config["repetition_penalty"],
                top_k=self.tts_config["top_k"],
                top_p=self.tts_config["top_p"]
            )
        
        # Load generated audio to return it
        audio, sr = sf.read(output_path)
        
        return {
            'audio': audio,
            'sampling_rate': sr,
            'output_path': output_path
        }
    
    def clone_voice(self, reference_audio, output_dir=None):
        """
        Extract speaker embedding for high-quality voice cloning
        
        Args:
            reference_audio (str/np.ndarray): Reference audio file path or audio data
            output_dir (str, optional): Directory to save extracted voice data
            
        Returns:
            dict: Dict containing:
                'speaker_embedding': Path to saved speaker embedding
                'voice_sample': Path to processed voice sample
                'duration': Duration of the processed sample
        """
        # Create output directory if needed
        if output_dir is None:
            output_dir = os.path.join(self.temp_dir, f"voice_profile_{int(torch.rand(1)[0]*10000)}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process reference audio
        voice_sample_path = os.path.join(output_dir, "voice_sample.wav")
        
        if isinstance(reference_audio, str):
            if not os.path.exists(reference_audio):
                raise FileNotFoundError(f"Reference audio file not found: {reference_audio}")
            
            # Load and process audio to ensure it's suitable for voice cloning
            audio = AudioSegment.from_file(reference_audio)
            
            # Trim silence and normalize volume for better cloning
            audio = audio.strip_silence(silence_thresh=-40, padding=100)
            audio = audio.normalize(headroom=0.1)
            
            # Ensure sufficient duration (XTTS needs 6+ seconds)
            if len(audio) < 6000:  # less than 6 seconds
                # Loop audio to reach minimum duration
                while len(audio) < 6000:
                    audio += audio
                # Trim to around 10 seconds max
                audio = audio[:10000]
            
            # Export processed audio
            audio.export(voice_sample_path, format="wav")
            
        elif isinstance(reference_audio, np.ndarray):
            # Save numpy array and process it
            temp_path = os.path.join(self.temp_dir, "temp_reference.wav")
            sf.write(temp_path, reference_audio, 16000)
            
            # Process using pydub for consistent quality
            audio = AudioSegment.from_file(temp_path)
            audio = audio.strip_silence(silence_thresh=-40, padding=100)
            audio = audio.normalize(headroom=0.1)
            
            # Ensure sufficient duration
            if len(audio) < 6000:
                while len(audio) < 6000:
                    audio += audio
                audio = audio[:10000]
            
            audio.export(voice_sample_path, format="wav")
        
        # XTTS will extract the embedding on-the-fly during synthesis
        # But we'll also create a metadata file with speaker info
        metadata_path = os.path.join(output_dir, "speaker_metadata.json")
        
        import json
        metadata = {
            "speaker_id": f"custom_voice_{Path(voice_sample_path).stem}",
            "sample_path": voice_sample_path,
            "duration": len(AudioSegment.from_file(voice_sample_path)) / 1000.0,
            "created_at": import_time(),
            "language": "auto"  # Will be detected during usage
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'speaker_embedding': voice_sample_path,  # XTTS uses the audio directly
            'voice_sample': voice_sample_path,
            'duration': metadata["duration"],
            'metadata': metadata_path
        }
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

def import_time():
    """Helper function to get current time"""
    from datetime import datetime
    return datetime.now().isoformat() 