#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
import cv2

class ImageCaptioner:
    """Image captioning using BLIP2-Flan-T5-XL model"""
    
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device=None):
        """
        Initialize the image captioner
        
        Args:
            model_name (str): Name of the captioning model to use
            device (str, optional): Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32
        )
        self.model.to(self.device)
        
        # Default generation parameters
        self.generation_params = {
            'max_length': 75,
            'min_length': 10,
            'num_beams': 5,
            'repetition_penalty': 1.2,
            'length_penalty': 1.0,
            'temperature': 0.7
        }
    
    def generate_caption(self, image, prompt=None, detailed=False):
        """
        Generate a caption for an image
        
        Args:
            image (np.ndarray): BGR image
            prompt (str, optional): Prompt to guide captioning
            detailed (bool): Whether to generate a detailed description
            
        Returns:
            dict: Dict containing:
                'caption': Generated caption
                'confidence': Confidence score (for ranking multiple captions)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare prompt
        if prompt is None:
            if detailed:
                prompt = "Provide a detailed description of this image, focusing on motion and activities."
            else:
                prompt = "Briefly describe this image."
                
        # Process image
        inputs = self.processor(image_rgb, text=prompt, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_params['max_length'],
                min_length=self.generation_params['min_length'],
                num_beams=self.generation_params['num_beams'],
                repetition_penalty=self.generation_params['repetition_penalty'],
                length_penalty=self.generation_params['length_penalty'],
                temperature=self.generation_params['temperature'],
                output_scores=True,
                return_dict_in_generate=True
            )
            
        # Decode generated text
        generated_text = self.processor.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Calculate confidence score (mean of sequence scores)
        scores = outputs.sequences_scores.cpu().numpy()
        confidence = float(scores[0]) if len(scores) > 0 else 0.0
        
        return {
            'caption': generated_text,
            'confidence': confidence
        }
    
    def generate_multiple_captions(self, image, num_captions=3):
        """
        Generate multiple diverse captions for an image
        
        Args:
            image (np.ndarray): BGR image
            num_captions (int): Number of captions to generate
            
        Returns:
            list: List of caption dictionaries
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Different prompts for diverse captions
        prompts = [
            "Describe this image briefly.",
            "What activities are shown in this image?",
            "Analyze the motion and action in this image.",
            "Describe the scene in detail, focusing on motion.",
            "What is happening in this image related to speed and movement?"
        ]
        
        # Generate captions with different prompts
        captions = []
        for i in range(min(num_captions, len(prompts))):
            caption_result = self.generate_caption(image, prompts[i])
            caption_result['prompt'] = prompts[i]
            captions.append(caption_result)
            
        # Sort by confidence
        captions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return captions
    
    def caption_video_frames(self, frames, interval=30, detailed=True):
        """
        Generate captions for video frames at regular intervals
        
        Args:
            frames (list): List of frames (numpy arrays)
            interval (int): Interval between frames to caption
            detailed (bool): Whether to generate detailed captions
            
        Returns:
            list: List of frame caption dictionaries with frame indices
        """
        # Generate captions for frames at specified intervals
        results = []
        for i in range(0, len(frames), interval):
            if i < len(frames):
                caption_result = self.generate_caption(frames[i], detailed=detailed)
                results.append({
                    'frame_idx': i,
                    'timestamp': i / 30.0,  # Assuming 30 fps
                    'caption': caption_result['caption'],
                    'confidence': caption_result['confidence']
                })
                
        return results 