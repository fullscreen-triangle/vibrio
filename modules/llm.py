#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class LLMProcessor:
    """LLM integration using Meta-Llama-3-8B-Instruct"""
    
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
                 device=None, load_in_8bit=False, quantize=True):
        """
        Initialize the LLM processor
        
        Args:
            model_name (str): Name of the LLM to use
            device (str, optional): Device to run inference on ('cpu', 'cuda', etc.)
            load_in_8bit (bool): Whether to load model in 8-bit precision
            quantize (bool): Whether to quantize the model for efficiency
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model with efficiency options
        model_kwargs = {}
        if quantize and 'cuda' in self.device:
            if load_in_8bit:
                model_kwargs = {
                    'load_in_8bit': True,
                    'device_map': 'auto'
                }
            else:
                model_kwargs = {
                    'torch_dtype': torch.float16,
                    'low_cpu_mem_usage': True
                }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if not load_in_8bit and 'device_map' not in model_kwargs:
            self.model.to(self.device)
            
        # Default generation parameters
        self.generation_params = {
            'max_new_tokens': 512,
            'min_length': 1,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1
        }
        
        # System prompt
        self.system_prompt = (
            "You are an assistant specialized in analyzing human movement and velocity. "
            "You provide insightful analysis of activities, techniques, and performance "
            "based on video analysis data. You can explain the biomechanics, physical "
            "constraints, and performance metrics in a clear, educational manner."
        )
    
    def generate_response(self, user_query, context=None, image_captions=None, 
                          pose_data=None, speed_data=None, action_data=None):
        """
        Generate a response from the LLM
        
        Args:
            user_query (str): User's question or request
            context (str, optional): Additional context
            image_captions (list, optional): Image captions from video frames
            pose_data (dict, optional): Pose analysis data
            speed_data (dict, optional): Speed analysis data
            action_data (dict, optional): Action recognition data
            
        Returns:
            dict: Dict containing:
                'response': Generated response
                'prompt_tokens': Number of prompt tokens
                'completion_tokens': Number of completion tokens
        """
        # Construct prompt with context
        full_prompt = self._construct_prompt(
            user_query, context, image_captions, pose_data, speed_data, action_data
        )
        
        # Tokenize prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_params,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode response (excluding prompt)
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = self._extract_assistant_response(full_output)
        
        # Count tokens
        completion_length = outputs.shape[1] - prompt_length
        
        return {
            'response': response,
            'prompt_tokens': prompt_length,
            'completion_tokens': completion_length
        }
    
    def _construct_prompt(self, user_query, context=None, image_captions=None, 
                          pose_data=None, speed_data=None, action_data=None):
        """Construct a prompt with all available data"""
        prompt_parts = []
        
        # System prompt
        prompt_parts.append(f"<system>\n{self.system_prompt}\n</system>\n\n")
        
        # Context section
        if context:
            prompt_parts.append(f"Context information:\n{context}\n\n")
            
        # Image captions
        if image_captions and len(image_captions) > 0:
            prompt_parts.append("Video frame descriptions:\n")
            for cap in image_captions[:5]:  # Limit to 5 captions
                timestamp = cap.get('timestamp', 0)
                caption = cap.get('caption', '')
                prompt_parts.append(f"- At {timestamp:.2f}s: {caption}\n")
            prompt_parts.append("\n")
            
        # Pose data
        if pose_data:
            prompt_parts.append("Pose analysis:\n")
            prompt_parts.append(f"- Number of detected people: {pose_data.get('num_people', 0)}\n")
            if '3d_available' in pose_data and pose_data['3d_available']:
                prompt_parts.append("- 3D pose data available\n")
            prompt_parts.append("\n")
            
        # Speed data
        if speed_data:
            prompt_parts.append("Speed analysis:\n")
            if 'velocity' in speed_data:
                prompt_parts.append(f"- Current velocity: {speed_data['velocity']:.2f} m/s\n")
            if 'max_velocity' in speed_data:
                prompt_parts.append(f"- Maximum velocity: {speed_data['max_velocity']:.2f} m/s\n")
            if 'acceleration' in speed_data:
                prompt_parts.append(f"- Current acceleration: {speed_data['acceleration']:.2f} m/s²\n")
            prompt_parts.append("\n")
            
        # Action data
        if action_data:
            prompt_parts.append("Activity recognition:\n")
            if 'action' in action_data:
                prompt_parts.append(f"- Detected activity: {action_data['action']}\n")
                prompt_parts.append(f"- Confidence: {action_data['confidence']:.2f}\n")
            prompt_parts.append("\n")
            
        # User query
        prompt_parts.append(f"<user>\n{user_query}\n</user>\n\n<assistant>\n")
        
        return ''.join(prompt_parts)
    
    def _extract_assistant_response(self, full_output):
        """Extract just the assistant's response from the output"""
        # This extracts the text between the last <assistant> and </assistant> tags
        assistant_pattern = r"<assistant>(.*?)</assistant>"
        matches = re.findall(assistant_pattern, full_output, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
        else:
            # If no tags found, try to extract the part after the prompt
            assistant_start = full_output.rfind("<assistant>")
            if assistant_start != -1:
                return full_output[assistant_start + len("<assistant>"):].strip()
            
            # Fallback: return everything after the user query
            user_end = full_output.rfind("</user>")
            if user_end != -1:
                return full_output[user_end + len("</user>"):].strip()
                
            # Last resort: return the whole output
            return full_output.strip() 