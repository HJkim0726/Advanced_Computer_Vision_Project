"""
Visualize Bench2Drive detection results compared with ground truth annotations
- Predicted bboxes from result_json (right-hand coordinate system)
- Ground truth bboxes from anno.json.gz (left-hand coordinate system)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import gzip
import argparse
import pickle
from PIL import Image
from scipy.spatial.transform import Rotation


class BboxComparisonVisualizer:
    """
    Visualize predicted vs ground truth bounding boxes in 3D space
    """
    
    def __init__(self, 
                 data_root: str,
                 result_json_path: str,
                 pkl_path: str = None,
                 output_root: str = None):
        """
        Args:
            data_root: Path to Bench2Drive dataset root
            result_json_path: Path to results JSON file
            pkl_path: Path to ground truth pkl file (e.g., b2d_infos_val.pkl)
            output_root: Output directory for visualizations
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root) if output_root else Path(data_root) / 'visualizations'
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(result_json_path, 'r') as f:
            result_data = json.load(f)
        
        # Handle both NuScenes format (with 'meta' key) and direct dict format
        if isinstance(result_data, dict) and 'results' in result_data:
            self.results = result_data['results']
            self.meta = result_data.get('meta', {})
        else:
            self.results = result_data
            self.meta = {}
        
        # Load pkl file for GT boxes
        self.gt_data = None
        if pkl_path:
            try:
                with open(pkl_path, 'rb') as f:
                    pkl_list = pickle.load(f)
                # Create dict for fast lookup: (folder, frame_idx) -> sample_info
                self.gt_data = {}
                for sample in pkl_list:
                    key = (sample['folder'], sample['frame_idx'])
                    self.gt_data[key] = sample
                print(f"[INFO] Loaded {len(self.gt_data)} GT samples from pkl")
            except Exception as e:
                print(f"[WARNING] Failed to load pkl file: {e}")
                self.gt_data = None
        
        # Color map for different classes
        self.class_colors = {
            'car': '#FF6B6B',
            'truck': '#4ECDC4',
            'bus': '#45B7D1',
            'pedestrian': '#FFA07A',
            'motorcycle': '#98D8C8',
            'bicycle': '#F7DC6F',
            'traffic_cone': '#BB8FCE',
            'traffic_light': '#85C1E2',
            'traffic_sign': '#F8B88B',
            'unknown': '#95A5A6',
        }
    
    def get_color(self, class_name: str) -> str:
        """Get color for a class name"""
        for key, color in self.class_colors.items():
            if key in class_name.lower():
                return color
        return self.class_colors['unknown']
    
    def quaternion_to_yaw(self, rotation_quat: List[float]) -> float:
        """
        Convert quaternion [qx, qy, qz, qw] to yaw angle (Z-axis rotation)
        
        BEVFormer outputs rotation as [sin(yaw/2), 0, 0, cos(yaw/2)]
        (essentially a pure X-axis rotation in quaternion form, which is unconventional)
        
        Args:
            rotation_quat: [qx, qy, qz, qw] quaternion components
            
        Returns:
            yaw: rotation angle in radians around Z-axis
        """
        if len(rotation_quat) != 4:
            return 0.0
        
        qx, qy, qz, qw = rotation_quat
        
        # BEVFormer encodes yaw as: qx = sin(yaw/2), qw = cos(yaw/2)
        # Therefore: yaw = 2 * atan2(qx, qw)
        # Note: Add 90 degrees (pi/2) for visualization coordinate system adjustment
        try:
            yaw = 2.0 * np.arctan2(qx, qw) + np.pi / 2.0
            # Normalize to [-pi, pi]
            while yaw > np.pi:
                yaw -= 2 * np.pi
            while yaw < -np.pi:
                yaw += 2 * np.pi
            return yaw
        except Exception as e:
            print(f"[WARNING] Failed to convert quaternion {rotation_quat}: {e}")
            return 0.0
    
    def draw_rotated_box(self, ax, center, size, yaw, color, linestyle='-', label=None, alpha=0.7):
        """
        Draw a rotated bounding box
        
        Args:
            ax: matplotlib axis
            center: [x, y] center position
            size: [width, length] (width=x direction, length=y direction in 2D)
            yaw: rotation angle in radians (counter-clockwise from x-axis)
            color: edge color
            linestyle: line style
            label: label for legend
            alpha: transparency
        """
        # Create box corners in local frame (centered at origin)
        w, l = size[0], size[1]
        corners = np.array([
            [-w/2, -l/2],
            [w/2, -l/2],
            [w/2, l/2],
            [-w/2, l/2],
            [-w/2, -l/2]  # Close the box
        ])
        
        # Rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        
        # Apply rotation
        corners_rotated = corners @ rotation.T
        
        # Translate to center
        corners_rotated += center
        
        # Draw polygon
        polygon = Polygon(corners_rotated[:-1], closed=True, 
                         edgecolor=color, facecolor='none', 
                         linestyle=linestyle, linewidth=2, alpha=alpha, label=label)
        ax.add_patch(polygon)
    
    def visualize_xy_comparison(self,
                                sample_token: str,
                                pred_bboxes: List,
                                gt_bboxes: np.ndarray,
                                scenario_path: Path,
                                frame_idx: int,
                                score_threshold: float = 0.3) -> None:
        """
        Visualize XY plane (top-down view) - Overlay only with ROI limits
        Side-by-side with TOP_DOWN camera image
        
        Args:
            sample_token: Sample token
            pred_bboxes: Predicted bboxes from result_json (right-hand coordinate system)
            gt_bboxes: Ground truth bboxes from pkl (numpy array, world coordinate system)
            scenario_path: Path to scenario folder
            frame_idx: Frame index
            score_threshold: Confidence threshold for predictions
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # LEFT PLOT: Bbox overlay comparison
        # Filter predictions by score
        pred_filtered = [b for b in pred_bboxes if b.get('detection_score', 0) >= score_threshold]
        
        # ROI bounds
        roi_x_min, roi_x_max = -51.2, 51.2
        roi_y_min, roi_y_max = -51.2, 51.2
        
        # Plot GT from pkl (world coordinates, [x, y, z, w, l, h, yaw, vx, vy])
        gt_plotted = False
        if gt_bboxes is not None and len(gt_bboxes) > 0:
            for gt_box in gt_bboxes:
                x, y, z, width, length, height, yaw, vx, vy = gt_box[:9]
                
                label = 'GT' if not gt_plotted else None
                # Draw with world coordinates (already right-hand system)
                # Reverse yaw direction for GT boxes (counter-clockwise rotation like predicted)
                gt_yaw = -yaw
                self.draw_rotated_box(ax1, [x, y], [width, length], gt_yaw,
                                     'red', linestyle='--', label=label, alpha=0.7)
                gt_plotted = True
        
        # Plot predictions (right-hand coordinate system)
        pred_plotted = False
        for pred in pred_filtered:
            trans = pred['translation']
            size = pred['size']
            
            # Get yaw from rotation (quaternion format [qx, qy, qz, qw])
            rotation_quat = pred.get('rotation', [0, 0, 0, 1])
            yaw = self.quaternion_to_yaw(rotation_quat)
            
            # Reverse yaw direction for predicted boxes (counter-clockwise rotation like GT)
            pred_yaw = -yaw
            
            label = 'Predicted' if not pred_plotted else None
            self.draw_rotated_box(ax1, trans[:2], size[:2], pred_yaw,
                                 'blue', linestyle='-', label=label, alpha=0.7)
            pred_plotted = True
        
        # Draw ROI boundary
        roi_rect = Rectangle((roi_x_min, roi_y_min), roi_x_max - roi_x_min, roi_y_max - roi_y_min,
                            linewidth=3, edgecolor='black', facecolor='none', alpha=0.3,
                            linestyle=':', label='ROI')
        ax1.add_patch(roi_rect)
        
        ax1.set_xlabel('X (m)', fontsize=12)
        ax1.set_ylabel('Y (m)', fontsize=12)
        ax1.set_title('Bbox Overlay (XY Plane - Top-Down)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.legend(loc='upper right', fontsize=10)
        
        # Set ROI limits
        ax1.set_xlim([roi_x_min - 5, roi_x_max + 5])
        ax1.set_ylim([roi_y_min - 5, roi_y_max + 5])
        
        # RIGHT PLOT: TOP_DOWN camera image
        img_path = scenario_path / 'camera' / 'rgb_top_down' / f'{frame_idx:05d}.jpg'
        
        if img_path.exists():
            try:
                img = Image.open(img_path)
                ax2.imshow(img)
                ax2.set_title(f'TOP_DOWN Camera Image\n{sample_token}', fontsize=12, fontweight='bold')
                ax2.axis('off')
            except Exception as e:
                ax2.text(0.5, 0.5, f'Error loading image:\n{str(e)}',
                        ha='center', va='center', fontsize=12)
                ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, f'Image not found:\n{img_path}',
                    ha='center', va='center', fontsize=12)
            ax2.axis('off')
        
        # Save
        safe_token = sample_token.replace('/', '_')
        output_path = self.output_root / f'{safe_token}_xy_comparison.jpg'
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
        plt.close()
    
    def visualize_sample(self,
                        sample_token: str,
                        score_threshold: float = 0.3) -> None:
        """
        Visualize predicted vs GT bboxes for a sample
        
        Args:
            sample_token: Sample token (scenario_name_frameindex)
            score_threshold: Confidence threshold
        """
        # Parse sample token
        parts = sample_token.rsplit('_', 1)
        if len(parts) == 2:
            scenario_name = parts[0]
            try:
                frame_idx = int(parts[1])
            except ValueError:
                print(f"[ERROR] Cannot parse frame index from: {sample_token}")
                return
        else:
            print(f"[ERROR] Invalid sample token: {sample_token}")
            return
        
        # Get scenario path
        scenario_path = self.data_root / scenario_name
        if not scenario_path.exists():
            print(f"[ERROR] Scenario path not found: {scenario_path}")
            return
        
        # Get predictions
        pred_bboxes = self.results.get(sample_token, [])
        if not pred_bboxes:
            print(f"[SKIP] No predictions found for {sample_token}")
            return
        
        # Get ground truth bboxes from pkl
        gt_bboxes = None
        if self.gt_data is not None:
            key = (scenario_name, frame_idx)
            if key in self.gt_data:
                gt_sample = self.gt_data[key]
                gt_bboxes = gt_sample.get('gt_boxes', None)
            else:
                print(f"[SKIP] No GT found in pkl for {sample_token}")
                return
        else:
            print(f"[SKIP] No pkl data loaded")
            return
        
        if gt_bboxes is None or len(gt_bboxes) == 0:
            print(f"[SKIP] No ground truth bboxes found for {sample_token}")
            return
        
        # Visualize
        print(f"[PROCESS] {sample_token}: {len(pred_bboxes)} predictions, {len(gt_bboxes)} GT bboxes")
        
        self.visualize_xy_comparison(sample_token, pred_bboxes, gt_bboxes, scenario_path, frame_idx, score_threshold)
    
    def visualize_all(self,
                      num_samples: Optional[int] = None,
                      score_threshold: float = 0.3) -> None:
        """
        Visualize all samples
        
        Args:
            num_samples: Number of samples to visualize (None = all)
            score_threshold: Confidence threshold
        """
        sample_tokens = list(self.results.keys())
        
        if num_samples is not None:
            sample_tokens = sample_tokens[:num_samples]
        
        print(f"\nTotal samples to visualize: {len(sample_tokens)}")
        print(f"Output directory: {self.output_root}\n")
        
        for sample_token in tqdm(sample_tokens):
            self.visualize_sample(sample_token, score_threshold)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Bench2Drive Bbox Comparison Visualizer')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to Bench2Drive dataset root')
    parser.add_argument('--result_json', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--pkl_path', type=str, default=None,
                       help='Path to ground truth pkl file (e.g., b2d_infos_val.pkl)')
    parser.add_argument('--output_root', type=str, default=None,
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to visualize')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = BboxComparisonVisualizer(
        data_root=args.data_root,
        result_json_path=args.result_json,
        pkl_path=args.pkl_path,
        output_root=args.output_root
    )
    
    # Visualize
    visualizer.visualize_all(
        num_samples=args.num_samples,
        score_threshold=args.score_threshold
    )
    
    print(f"\n✓ Visualizations saved to: {visualizer.output_root}")


if __name__ == '__main__':
    main()
