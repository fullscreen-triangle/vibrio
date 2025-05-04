#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import os
import json
import cv2
from transformers import AutoFeatureExtractor, AutoModelForSeq2SeqLM, AutoConfig
from typing import Dict, List, Tuple, Union, Optional

class Pose3DEstimator:
    """3D pose estimation using MotionBERT-Lite model with full implementation"""
    
    def __init__(self, model_name="walterzhu/MotionBERT-Lite", device=None, cache_dir=None):
        """
        Initialize the 3D pose estimator
        
        Args:
            model_name (str): Name of the 3D pose model to use
            device (str, optional): Device to run inference on ('cpu', 'cuda', etc.)
            cache_dir (str, optional): Directory to cache models
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        self.model_name = model_name
        
        print(f"Initializing Pose3DEstimator with {model_name} on {self.device}")
        
        # Initialize model and processor
        self._load_model()
        
        # Cached sequence data for temporal consistency
        self.pose_sequence = []
        self.max_sequence_length = 27  # MotionBERT typically uses sequences of 27 frames
        
        # Camera configuration for 3D reconstruction
        self.camera_config = {
            'focal_length': 1000.0,  # Default focal length
            'center': [500.0, 500.0],  # Default principal point
            'world_up': np.array([0.0, 1.0, 0.0]),  # Y-up world coordinate system
            'camera_position': np.array([0.0, 0.0, 0.0])  # Camera at origin
        }
        
        # Joint connections for visualization
        self.joint_connections = [
            [0, 1], [1, 2], [2, 3],  # Right leg
            [0, 4], [4, 5], [5, 6],  # Left leg
            [0, 7], [7, 8], [8, 9],  # Spine and head
            [8, 10], [10, 11], [11, 12],  # Right arm
            [8, 13], [13, 14], [14, 15]   # Left arm
        ]
        
        # Joint names for human skeleton
        self.joint_names = [
            "pelvis", "right_hip", "right_knee", "right_ankle",
            "left_hip", "left_knee", "left_ankle", 
            "spine", "neck", "head", "right_shoulder", 
            "right_elbow", "right_wrist", "left_shoulder",
            "left_elbow", "left_wrist"
        ]
    
    def _load_model(self):
        """Load and configure the MotionBERT model"""
        try:
            # Download model configuration and get actual architecture details
            model_config = AutoConfig.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            print(f"Loading MotionBERT model: {self.model_name}")
            
            # Load feature extractor with appropriate configuration
            self.processor = AutoFeatureExtractor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Initialize model with appropriate dtype for hardware
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Store model-specific configurations
            self.embedding_dim = getattr(model_config, 'hidden_size', 512)
            self.num_joints = 16  # MotionBERT standard output is 16 joints
            self.output_format = '3d_pos'  # Default output format is 3D positions
            
            print(f"MotionBERT model loaded successfully: {self.num_joints} joints, {self.embedding_dim}d embeddings")
            
        except Exception as e:
            print(f"Error loading MotionBERT model: {str(e)}")
            raise RuntimeError(f"Failed to initialize 3D pose estimation model: {str(e)}")
        
    def lift_to_3d(self, pose_2d_data, frame=None, camera_params=None):
        """
        Convert 2D pose keypoints to 3D pose with full implementation
        
        Args:
            pose_2d_data (dict): 2D pose data from PoseDetector
            frame (numpy.ndarray, optional): Original frame for reference
            camera_params (dict, optional): Camera parameters for 3D reconstruction
            
        Returns:
            dict: Dict containing:
                'keypoints_3d': List of 3D keypoints, each as [x, y, z, confidence]
                'motion_embedding': Motion embedding vector
                'pose_data': Additional pose-specific data
                'joint_angles': 3D joint angles in degrees
                'trajectory': 3D trajectory data
        """
        # Validate input
        if not pose_2d_data or 'keypoints' not in pose_2d_data or not pose_2d_data['keypoints']:
            print("Warning: No valid 2D pose data provided for 3D lifting")
            return {
                'keypoints_3d': [],
                'motion_embedding': None,
                'pose_data': [],
                'joint_angles': [],
                'trajectory': None
            }
        
        # Update camera configuration if provided
        if camera_params:
            self.camera_config.update(camera_params)
        
        # Update pose sequence with new frame data
        self._update_pose_sequence(pose_2d_data)
        
        # Get a sequence of appropriate length for the model
        sequence = self._get_padded_sequence()
        
        # Normalize the sequence for the model
        normalized_sequence = self._normalize_sequence(sequence, frame.shape if frame is not None else None)
        
        # Process sequence for model input
        inputs = self._prepare_inputs(normalized_sequence)
        
        # Run inference
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Extract 3D pose coordinates
            poses_3d = self._extract_3d_poses(outputs)
            
            # Extract motion embedding for action recognition
            motion_embedding = self._extract_motion_embedding(outputs)
            
            # Calculate joint angles for biomechanical analysis
            joint_angles = self._calculate_joint_angles(poses_3d)
            
            # Estimate trajectory data
            trajectory = self._estimate_trajectory(poses_3d)
            
            # Process the pose data for output
            pose_data = self._prepare_pose_data(poses_3d, pose_2d_data)
            
            return {
                'keypoints_3d': poses_3d,
                'motion_embedding': motion_embedding,
                'pose_data': pose_data,
                'joint_angles': joint_angles,
                'trajectory': trajectory
            }
            
        except Exception as e:
            print(f"Error during 3D pose estimation: {str(e)}")
            # Return empty results on error
            return {
                'keypoints_3d': [],
                'motion_embedding': None,
                'pose_data': [],
                'joint_angles': [],
                'trajectory': None
            }
    
    def _update_pose_sequence(self, pose_2d_data):
        """Update the sequence buffer with new pose data"""
        # For simplicity, we'll just use the first detected person
        if pose_2d_data['keypoints']:
            keypoints = pose_2d_data['keypoints'][0]
            
            # Preprocess keypoints to match MotionBERT input format
            processed_keypoints = self._preprocess_keypoints(keypoints)
            
            # Add to sequence
            self.pose_sequence.append(processed_keypoints)
            
        # Maintain fixed length with sliding window approach
        if len(self.pose_sequence) > self.max_sequence_length:
            self.pose_sequence.pop(0)
    
    def _preprocess_keypoints(self, keypoints):
        """
        Preprocess keypoints to match MotionBERT expected format
        
        Args:
            keypoints: List of [x, y, confidence] values from pose detector
            
        Returns:
            List of preprocessed keypoints in MotionBERT format
        """
        # Typically pose detectors like YOLOv8 provide 17 keypoints 
        # in COCO format, but MotionBERT needs 16 keypoints in a specific format
        
        # Create a mapping from COCO keypoints to MotionBERT joints
        coco_to_motionbert = {
            0: 8,    # nose -> neck (approximation)
            1: 9,    # left_eye -> head (approximation)
            2: 9,    # right_eye -> head (approximation)
            3: 9,    # left_ear -> head (approximation)
            4: 9,    # right_ear -> head (approximation)
            5: 13,   # left_shoulder -> left_shoulder
            6: 10,   # right_shoulder -> right_shoulder
            7: 14,   # left_elbow -> left_elbow
            8: 11,   # right_elbow -> right_elbow
            9: 15,   # left_wrist -> left_wrist
            10: 12,  # right_wrist -> right_wrist 
            11: 4,   # left_hip -> left_hip
            12: 1,   # right_hip -> right_hip
            13: 5,   # left_knee -> left_knee
            14: 2,   # right_knee -> right_knee
            15: 6,   # left_ankle -> left_ankle
            16: 3    # right_ankle -> right_ankle
        }
        
        # Initialize MotionBERT keypoints
        motion_bert_keypoints = [[0, 0, 0] for _ in range(16)]
        
        # Zero-centered coordinates
        center_x = 0.0
        center_y = 0.0
        valid_count = 0
        
        # First pass: calculate center from valid keypoints
        for i, kpt in enumerate(keypoints):
            if kpt[2] > 0.3:  # Only use keypoints with good confidence
                center_x += kpt[0]
                center_y += kpt[1]
                valid_count += 1
        
        if valid_count > 0:
            center_x /= valid_count
            center_y /= valid_count
        
        # Second pass: map keypoints to MotionBERT format
        for i, kpt in enumerate(keypoints):
            if i in coco_to_motionbert:
                mb_idx = coco_to_motionbert[i]
                
                # Transfer coordinates and confidence
                if kpt[2] > 0.3:  # Only use keypoints with good confidence
                    motion_bert_keypoints[mb_idx] = [
                        kpt[0] - center_x,  # X (centered)
                        kpt[1] - center_y,  # Y (centered)
                        kpt[2]              # Confidence
                    ]
        
        # Calculate pelvis position (average of hip joints)
        left_hip = motion_bert_keypoints[4]
        right_hip = motion_bert_keypoints[1]
        
        # If both hips are detected with good confidence
        if left_hip[2] > 0.3 and right_hip[2] > 0.3:
            pelvis_x = (left_hip[0] + right_hip[0]) / 2
            pelvis_y = (left_hip[1] + right_hip[1]) / 2
            pelvis_conf = (left_hip[2] + right_hip[2]) / 2
            motion_bert_keypoints[0] = [pelvis_x, pelvis_y, pelvis_conf]
        
        # Calculate spine position (between pelvis and neck)
        pelvis = motion_bert_keypoints[0]
        neck = motion_bert_keypoints[8]
        
        if pelvis[2] > 0.3 and neck[2] > 0.3:
            spine_x = (pelvis[0] + neck[0]) / 2
            spine_y = (pelvis[1] + neck[1]) / 2
            spine_conf = (pelvis[2] + neck[2]) / 2
            motion_bert_keypoints[7] = [spine_x, spine_y, spine_conf]
        
        return motion_bert_keypoints
    
    def _get_padded_sequence(self):
        """Get a sequence of fixed length with proper padding if necessary"""
        if len(self.pose_sequence) < self.max_sequence_length:
            # Pad with copies of the first frame if sequence is too short
            pad_length = self.max_sequence_length - len(self.pose_sequence)
            pad_frame = self.pose_sequence[0] if self.pose_sequence else [[0, 0, 0]] * 16
            padded_sequence = [pad_frame] * pad_length + self.pose_sequence
        else:
            padded_sequence = self.pose_sequence
            
        return padded_sequence
    
    def _normalize_sequence(self, sequence, image_shape=None):
        """
        Normalize a sequence for MotionBERT input
        
        Args:
            sequence: List of keypoint sequences
            image_shape: Original image shape for scale normalization
            
        Returns:
            Normalized sequence array
        """
        # Convert to numpy array
        sequence_array = np.array(sequence)
        
        # Separate coordinates and confidences
        coords = sequence_array[:, :, :2]  # [frames, joints, xy]
        confs = sequence_array[:, :, 2]    # [frames, joints]
        
        # Calculate scale factor for normalization
        if image_shape is not None:
            # Use image dimensions for scale normalization
            height, width = image_shape[:2]
            scale_factor = max(width, height) / 2.0
        else:
            # Use standard scale
            scale_factor = 256.0
        
        # Normalize coordinates to [-1, 1] range
        normalized_coords = coords / scale_factor
        
        # Apply confidence as weights to coordinates 
        # (low confidence points will be closer to zero)
        weighted_coords = np.zeros_like(normalized_coords)
        for i in range(sequence_array.shape[0]):
            for j in range(sequence_array.shape[1]):
                weighted_coords[i, j, 0] = normalized_coords[i, j, 0] * confs[i, j]
                weighted_coords[i, j, 1] = normalized_coords[i, j, 1] * confs[i, j]
        
        # Reshape to match MotionBERT input format: [frames, joints*2]
        flattened_coords = weighted_coords.reshape(sequence_array.shape[0], -1)
        
        return flattened_coords
    
    def _prepare_inputs(self, normalized_sequence):
        """
        Prepare inputs for the MotionBERT model
        
        Args:
            normalized_sequence: Normalized sequence array
            
        Returns:
            Dictionary of model inputs
        """
        # Convert to tensor and ensure proper shape
        sequence_tensor = torch.tensor(normalized_sequence, dtype=torch.float32).to(self.device)
        
        # Apply the processor to format inputs for the model
        inputs = self.processor(
            sequence_tensor, 
            return_tensors="pt"
        ).to(self.device)
        
        return inputs
    
    def _extract_3d_poses(self, model_outputs):
        """
        Extract 3D poses from model outputs
        
        Args:
            model_outputs: Model output tensor
            
        Returns:
            List of 3D keypoints as [x, y, z, confidence]
        """
        # Get the decoder outputs containing 3D coordinates
        decoder_outputs = model_outputs.last_hidden_state
        
        # Convert to numpy for processing
        poses_3d = decoder_outputs.cpu().numpy()
        
        # Extract coordinates for the current (last) frame
        # MotionBERT outputs are in format [batch, frames, joints*3]
        current_pose_3d = poses_3d[0, -1]
        
        # Reshape to [joints, 3]
        num_joints = self.num_joints
        current_pose_3d = current_pose_3d.reshape(num_joints, 3)
        
        # Format as list of [x, y, z, confidence]
        keypoints_3d = []
        for i in range(num_joints):
            # Add placeholder confidence of 1.0 for 3D points
            # In a real system, this would be derived from 2D confidence and reprojection error
            keypoints_3d.append([
                float(current_pose_3d[i, 0]),
                float(current_pose_3d[i, 1]),
                float(current_pose_3d[i, 2]),
                1.0
            ])
            
        return keypoints_3d
    
    def _extract_motion_embedding(self, model_outputs):
        """
        Extract motion embedding for action recognition
        
        Args:
            model_outputs: Model output tensor
            
        Returns:
            Motion embedding vector
        """
        # MotionBERT produces rich embeddings in the encoder
        # We'll use the mean of the encoder outputs as our motion embedding
        encoder_outputs = model_outputs.encoder_last_hidden_state
        
        # Average across the sequence dimension to get a single embedding vector
        embeddings = encoder_outputs.mean(dim=1).cpu().numpy()
        
        # Post-processing for better action recognition downstream
        # L2 normalization helps with cosine similarity later
        embedding = embeddings[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def _calculate_joint_angles(self, keypoints_3d):
        """
        Calculate 3D joint angles from keypoints
        
        Args:
            keypoints_3d: List of 3D keypoints
            
        Returns:
            Dictionary of joint angles in degrees
        """
        # Joint angle definitions (parent, child, grandchild for each angle)
        joint_angle_defs = [
            # Legs
            {'name': 'right_hip', 'joints': [0, 1, 2]},  # pelvis -> right_hip -> right_knee
            {'name': 'right_knee', 'joints': [1, 2, 3]}, # right_hip -> right_knee -> right_ankle
            {'name': 'left_hip', 'joints': [0, 4, 5]},   # pelvis -> left_hip -> left_knee
            {'name': 'left_knee', 'joints': [4, 5, 6]},  # left_hip -> left_knee -> left_ankle
            
            # Arms
            {'name': 'right_shoulder', 'joints': [8, 10, 11]}, # neck -> right_shoulder -> right_elbow
            {'name': 'right_elbow', 'joints': [10, 11, 12]},   # right_shoulder -> right_elbow -> right_wrist
            {'name': 'left_shoulder', 'joints': [8, 13, 14]},  # neck -> left_shoulder -> left_elbow
            {'name': 'left_elbow', 'joints': [13, 14, 15]},    # left_shoulder -> left_elbow -> left_wrist
            
            # Spine
            {'name': 'spine', 'joints': [0, 7, 8]},  # pelvis -> spine -> neck
            {'name': 'neck', 'joints': [7, 8, 9]},   # spine -> neck -> head
        ]
        
        angles = []
        
        # Calculate angle for each joint
        for angle_def in joint_angle_defs:
            j1, j2, j3 = angle_def['joints']
            
            # Get 3D coordinates for the joints
            v1 = np.array([keypoints_3d[j1][0], keypoints_3d[j1][1], keypoints_3d[j1][2]])
            v2 = np.array([keypoints_3d[j2][0], keypoints_3d[j2][1], keypoints_3d[j2][2]])
            v3 = np.array([keypoints_3d[j3][0], keypoints_3d[j3][1], keypoints_3d[j3][2]])
            
            # Calculate vectors
            vector1 = v1 - v2  # parent to joint
            vector2 = v3 - v2  # child to joint
            
            # Check if vectors are valid
            if np.linalg.norm(vector1) > 1e-6 and np.linalg.norm(vector2) > 1e-6:
                # Calculate angle using dot product formula
                cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                # Ensure cosine is in valid range [-1, 1]
                cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                # Calculate angle in degrees
                angle_rad = np.arccos(cosine_angle)
                angle_deg = np.degrees(angle_rad)
                
                angles.append({
                    'name': angle_def['name'],
                    'angle': float(angle_deg),
                    'joints': angle_def['joints']
                })
            else:
                # Fallback for invalid vectors
                angles.append({
                    'name': angle_def['name'],
                    'angle': 0.0,
                    'joints': angle_def['joints']
                })
        
        return angles
    
    def _estimate_trajectory(self, keypoints_3d):
        """
        Estimate 3D trajectory from pose sequence
        
        Args:
            keypoints_3d: Current frame 3D keypoints
            
        Returns:
            Trajectory data
        """
        # For trajectory estimation, we need more than just the current frame
        # We'll use the full pose sequence to estimate velocity and acceleration
        
        # Use pelvis position as the person's position
        current_position = np.array([keypoints_3d[0][0], keypoints_3d[0][1], keypoints_3d[0][2]])
        
        # Initialize trajectory data
        trajectory = {
            'position': current_position.tolist(),
            'velocity': [0.0, 0.0, 0.0],
            'acceleration': [0.0, 0.0, 0.0],
            'speed': 0.0,
            'direction': [0.0, 0.0, 0.0]
        }
        
        # Check if we have enough frames for velocity calculation
        if len(self.pose_sequence) > 2:
            # Extract pelvis positions from recent frames
            positions = []
            for i in range(min(5, len(self.pose_sequence))):
                frame_idx = len(self.pose_sequence) - i - 1
                if frame_idx >= 0:
                    # Get pelvis position from historical data
                    pelvis_pos = self.pose_sequence[frame_idx][0]  # Pelvis is index 0
                    if isinstance(pelvis_pos, list) and len(pelvis_pos) >= 2:
                        positions.append([pelvis_pos[0], pelvis_pos[1], 0.0])  # Use 2D positions
            
            # If we have enough valid positions
            if len(positions) >= 3:
                # Calculate velocity (from t-2 to t)
                v1 = np.array(positions[0]) - np.array(positions[2])
                # Normalize for speed calculation (assuming constant frame rate)
                speed = np.linalg.norm(v1) / 2.0  # Units per frame
                
                # Calculate acceleration (difference of velocities)
                v2 = np.array(positions[1]) - np.array(positions[2])  # Velocity at t-1
                accel = v1 - v2  # Change in velocity
                
                # Calculate direction vector
                direction = v1.copy()
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                
                # Update trajectory data
                trajectory['velocity'] = v1.tolist()
                trajectory['acceleration'] = accel.tolist()
                trajectory['speed'] = float(speed)
                trajectory['direction'] = direction.tolist()
        
        return trajectory
    
    def _prepare_pose_data(self, keypoints_3d, pose_2d_data):
        """
        Prepare comprehensive pose data by combining 2D and 3D information
        
        Args:
            keypoints_3d: 3D keypoints data
            pose_2d_data: Original 2D pose data
            
        Returns:
            List of enhanced pose data
        """
        pose_data = []
        
        # If we have 2D pose data, enhance it with 3D information
        if pose_2d_data and 'pose_data' in pose_2d_data and pose_2d_data['pose_data']:
            for i, pose in enumerate(pose_2d_data['pose_data']):
                # Only process the first person for simplicity
                if i == 0:
                    enhanced_pose = pose.copy()  # Start with original 2D pose data
                    
                    # Add 3D-specific data
                    enhanced_pose['3d_available'] = True
                    enhanced_pose['3d_joint_positions'] = keypoints_3d
                    enhanced_pose['3d_joint_names'] = self.joint_names
                    enhanced_pose['3d_connections'] = self.joint_connections
                    
                    # Add a unique ID for this 3D pose
                    enhanced_pose['3d_id'] = f"3d_pose_{len(self.pose_sequence)}"
                    
                    pose_data.append(enhanced_pose)
                else:
                    # For other people, just copy the 2D data
                    pose_data.append(pose)
        
        return pose_data
    
    def get_motion_embedding(self):
        """
        Get the latest motion embedding for action recognition
        
        Returns:
            Motion embedding vector normalized for similarity comparison
        """
        if not self.pose_sequence:
            return None
            
        # Get properly formatted sequence
        sequence = self._get_padded_sequence()
        normalized_sequence = self._normalize_sequence(sequence)
        inputs = self._prepare_inputs(normalized_sequence)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        embedding = self._extract_motion_embedding(outputs)
        return embedding
    
    def visualize_3d_pose(self, keypoints_3d, output_path=None, 
                          azimuth=30, elevation=30, dist=7.0):
        """
        Generate a 3D visualization of the pose
        
        Args:
            keypoints_3d: List of 3D keypoints
            output_path: Path to save visualization image
            azimuth: Camera azimuth angle
            elevation: Camera elevation angle
            dist: Camera distance
            
        Returns:
            Visualization image as numpy array (if matplotlib is available)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create figure
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get points
            xs = [kp[0] for kp in keypoints_3d]
            ys = [kp[2] for kp in keypoints_3d]  # MotionBERT uses y as up-axis, but we want z as up
            zs = [-kp[1] for kp in keypoints_3d]  # Negate y to match standard 3D coordinate system
            
            # Plot joints
            ax.scatter(xs, ys, zs, c='r', marker='o')
            
            # Plot connections
            for connection in self.joint_connections:
                i, j = connection
                if i < len(keypoints_3d) and j < len(keypoints_3d):
                    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], 'b-')
            
            # Set labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            
            # Set equal aspect ratio
            # Calculate max range for equal aspect ratio
            max_range = np.array([max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]).max() / 2.0
            mid_x = (max(xs) + min(xs)) * 0.5
            mid_y = (max(ys) + min(ys)) * 0.5
            mid_z = (max(zs) + min(zs)) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Set camera position
            ax.view_init(elev=elevation, azim=azimuth)
            
            # Add title
            ax.set_title('3D Human Pose')
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
                plt.close(fig)
                return None
            else:
                # Convert to image
                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                return img
                
        except ImportError:
            print("Matplotlib not available for 3D visualization")
            return None 