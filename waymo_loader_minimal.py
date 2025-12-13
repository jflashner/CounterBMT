"""
waymo_loader_minimal.py

Load and visualize Waymo TFRecords WITHOUT TensorFlow or waymo-open-dataset.
Works with Python 3.8+.

Requirements:
    pip install protobuf matplotlib numpy Pillow

Usage:
    python waymo_loader_minimal.py your_file.tfrecord --list
    python waymo_loader_minimal.py your_file.tfrecord --scenario 0 --output ./screenshots
"""

import struct
from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Optional, Any, Union
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TFRECORD READER
# =============================================================================

def read_tfrecord(filepath: str) -> Iterator[bytes]:
    """
    Read TFRecord file without TensorFlow.
    
    TFRecord format per record:
    - uint64 length
    - uint32 masked_crc32_of_length  
    - byte   data[length]
    - uint32 masked_crc32_of_data
    """
    with open(filepath, 'rb') as f:
        while True:
            # Read length (8 bytes, little-endian uint64)
            length_bytes = f.read(8)
            if len(length_bytes) < 8:
                break
            
            length = struct.unpack('<Q', length_bytes)[0]
            
            # Skip length CRC (4 bytes)
            f.read(4)
            
            # Read data
            data = f.read(length)
            if len(data) < length:
                break
            
            # Skip data CRC (4 bytes)
            f.read(4)
            
            yield data


# =============================================================================
# PROTOBUF PARSER
# =============================================================================

class ProtoParser:
    """Simple protobuf wire format parser."""
    
    VARINT = 0
    FIXED64 = 1
    LENGTH_DELIMITED = 2
    START_GROUP = 3  # Deprecated
    END_GROUP = 4    # Deprecated
    FIXED32 = 5
    
    @classmethod
    def parse(cls, data: bytes) -> Dict[int, List[Tuple[int, Any]]]:
        """Parse protobuf message into {field_number: [(wire_type, value), ...]}"""
        if not isinstance(data, (bytes, bytearray)):
            return {}
        
        result = {}
        pos = 0
        data_len = len(data)
        
        while pos < data_len:
            try:
                # Read tag (varint)
                tag, pos = cls._read_varint(data, pos)
                if pos > data_len:
                    break
                    
                field_num = tag >> 3
                wire_type = tag & 0x7
                
                # Read value based on wire type
                if wire_type == cls.VARINT:
                    value, pos = cls._read_varint(data, pos)
                elif wire_type == cls.FIXED64:
                    if pos + 8 > data_len:
                        break
                    value = struct.unpack('<Q', data[pos:pos+8])[0]
                    pos += 8
                elif wire_type == cls.LENGTH_DELIMITED:
                    length, pos = cls._read_varint(data, pos)
                    if pos + length > data_len:
                        break
                    value = data[pos:pos+length]
                    pos += length
                elif wire_type == cls.FIXED32:
                    if pos + 4 > data_len:
                        break
                    value = struct.unpack('<I', data[pos:pos+4])[0]
                    pos += 4
                elif wire_type in (cls.START_GROUP, cls.END_GROUP):
                    # Deprecated, skip
                    continue
                else:
                    # Unknown wire type, stop parsing
                    break
                
                result.setdefault(field_num, []).append((wire_type, value))
                
            except (struct.error, IndexError, ValueError):
                break
        
        return result
    
    @staticmethod
    def _read_varint(data: bytes, pos: int) -> Tuple[int, int]:
        """Decode varint at position."""
        result = 0
        shift = 0
        while pos < len(data):
            byte = data[pos]
            pos += 1
            result |= (byte & 0x7f) << shift
            if not (byte & 0x80):
                break
            shift += 7
            if shift > 63:  # Overflow protection
                break
        return result, pos
    
    @classmethod
    def get_bytes(cls, msg: Dict, field: int) -> Optional[bytes]:
        """Get bytes field value."""
        if field in msg and msg[field]:
            wire_type, value = msg[field][0]
            if wire_type == cls.LENGTH_DELIMITED and isinstance(value, bytes):
                return value
        return None
    
    @classmethod
    def get_string(cls, msg: Dict, field: int, default: str = "") -> str:
        """Get string field value."""
        data = cls.get_bytes(msg, field)
        if data:
            try:
                return data.decode('utf-8', errors='ignore')
            except:
                return default
        return default
    
    @classmethod
    def get_int(cls, msg: Dict, field: int, default: int = 0) -> int:
        """Get integer field value."""
        if field in msg and msg[field]:
            wire_type, value = msg[field][0]
            if wire_type == cls.VARINT:
                return int(value)
            elif wire_type == cls.FIXED32:
                return int(value)
            elif wire_type == cls.FIXED64:
                return int(value)
        return default
    
    @classmethod
    def get_float32(cls, msg: Dict, field: int, default: float = 0.0) -> float:
        """Get float32 field value."""
        if field in msg and msg[field]:
            wire_type, value = msg[field][0]
            if wire_type == cls.FIXED32:
                # Reinterpret uint32 as float32
                return struct.unpack('<f', struct.pack('<I', value))[0]
            elif wire_type == cls.FIXED64:
                # Reinterpret uint64 as float64, then convert
                return float(struct.unpack('<d', struct.pack('<Q', value))[0])
        return default
    
    @classmethod
    def get_float64(cls, msg: Dict, field: int, default: float = 0.0) -> float:
        """Get float64/double field value."""
        if field in msg and msg[field]:
            wire_type, value = msg[field][0]
            if wire_type == cls.FIXED64:
                # Reinterpret uint64 as float64
                return struct.unpack('<d', struct.pack('<Q', value))[0]
            elif wire_type == cls.FIXED32:
                return struct.unpack('<f', struct.pack('<I', value))[0]
        return default
    
    @classmethod
    def get_repeated_float64(cls, msg: Dict, field: int) -> List[float]:
        """Get repeated double values (packed or unpacked)."""
        result = []
        if field not in msg:
            return result
        
        for wire_type, value in msg[field]:
            if wire_type == cls.FIXED64:
                # Single unpacked double (stored as uint64)
                try:
                    f = struct.unpack('<d', struct.pack('<Q', value))[0]
                    result.append(f)
                except:
                    pass
            elif wire_type == cls.LENGTH_DELIMITED and isinstance(value, bytes):
                # Packed doubles
                for i in range(0, len(value) - 7, 8):
                    try:
                        f = struct.unpack('<d', value[i:i+8])[0]
                        result.append(f)
                    except:
                        pass
        return result
    
    @classmethod
    def get_repeated_messages(cls, msg: Dict, field: int) -> List[bytes]:
        """Get repeated embedded messages."""
        result = []
        if field not in msg:
            return result
        
        for wire_type, value in msg[field]:
            if wire_type == cls.LENGTH_DELIMITED and isinstance(value, bytes):
                result.append(value)
        return result


# =============================================================================
# WAYMO SCENARIO PARSER  
# =============================================================================

def parse_scenario(data: bytes) -> Dict:
    """
    Parse Waymo scenario protobuf.
    
    Field numbers based on waymo scenario.proto:
    - 1: tracks (repeated Track)
    - 5: scenario_id (string)
    - 7: timestamps_seconds (repeated double)
    - 8: map_features (repeated MapFeature)
    - 9: sdc_track_index (int32)
    """
    msg = ProtoParser.parse(data)
    
    scenario_id = ProtoParser.get_string(msg, 5, "unknown")
    timestamps = ProtoParser.get_repeated_float64(msg, 7)
    sdc_track_index = ProtoParser.get_int(msg, 9, 0)
    
    # Parse tracks
    tracks = {}
    sdc_id = None
    
    track_messages = ProtoParser.get_repeated_messages(msg, 1)
    for i, track_data in enumerate(track_messages):
        try:
            track = parse_track(track_data)
            if track:
                track['is_sdc'] = (i == sdc_track_index)
                tracks[track['id']] = track
                if i == sdc_track_index:
                    sdc_id = track['id']
        except Exception as e:
            logger.debug(f"Failed to parse track {i}: {e}")
            continue
    
    # Parse map features
    map_features = {}
    feature_messages = ProtoParser.get_repeated_messages(msg, 8)
    for feat_data in feature_messages:
        try:
            feat = parse_map_feature(feat_data)
            if feat and 'polyline' in feat and len(feat['polyline']) > 0:
                map_features[feat['id']] = feat
        except Exception as e:
            logger.debug(f"Failed to parse map feature: {e}")
            continue
    
    # Default timestamps if not found
    if not timestamps:
        num_steps = 91
        if tracks:
            first_track = next(iter(tracks.values()))
            num_steps = len(first_track['state']['position'])
        timestamps = [i * 0.1 for i in range(num_steps)]
    
    return {
        'id': scenario_id,
        'timestamps': np.array(timestamps),
        'tracks': tracks,
        'map_features': map_features,
        'metadata': {
            'sdc_id': sdc_id,
            'sdc_track_index': sdc_track_index,
            'num_timesteps': len(timestamps)
        }
    }


def parse_track(data: bytes) -> Optional[Dict]:
    """
    Parse Track message.
    
    Fields:
    - 1: id (int32)
    - 2: object_type (enum: 1=vehicle, 2=pedestrian, 3=cyclist)
    - 3: states (repeated ObjectState)
    """
    msg = ProtoParser.parse(data)
    
    track_id = ProtoParser.get_int(msg, 1, 0)
    obj_type = ProtoParser.get_int(msg, 2, 1)
    type_names = {0: 'UNSET', 1: 'VEHICLE', 2: 'PEDESTRIAN', 3: 'CYCLIST'}
    
    # Parse states
    positions, headings, velocities, valid_mask = [], [], [], []
    lengths, widths = [], []
    
    state_messages = ProtoParser.get_repeated_messages(msg, 3)
    for state_data in state_messages:
        try:
            state = parse_object_state(state_data)
            if state:
                positions.append([state['x'], state['y'], state['z']])
                headings.append(state['heading'])
                velocities.append([state['vx'], state['vy']])
                valid_mask.append(state['valid'])
                lengths.append(state['length'])
                widths.append(state['width'])
        except Exception as e:
            logger.debug(f"Failed to parse state: {e}")
            # Add default invalid state
            positions.append([0, 0, 0])
            headings.append(0)
            velocities.append([0, 0])
            valid_mask.append(False)
            lengths.append(4.5)
            widths.append(2.0)
    
    if not positions:
        return None
    
    return {
        'id': track_id,
        'type': type_names.get(obj_type, 'VEHICLE'),
        'state': {
            'position': np.array(positions),
            'heading': np.array(headings),
            'velocity': np.array(velocities),
            'valid': np.array(valid_mask, dtype=bool),
            'length': np.array(lengths),
            'width': np.array(widths),
        }
    }


def parse_object_state(data: bytes) -> Optional[Dict]:
    """
    Parse ObjectState message.
    
    Fields: 1=x, 2=y, 3=z, 4=length, 5=width, 6=height, 7=heading, 8=vx, 9=vy, 10=valid
    All position/velocity fields are float (FIXED32).
    """
    msg = ProtoParser.parse(data)
    
    return {
        'x': ProtoParser.get_float32(msg, 1, 0.0),
        'y': ProtoParser.get_float32(msg, 2, 0.0),
        'z': ProtoParser.get_float32(msg, 3, 0.0),
        'length': ProtoParser.get_float32(msg, 4, 4.5),
        'width': ProtoParser.get_float32(msg, 5, 2.0),
        'height': ProtoParser.get_float32(msg, 6, 1.5),
        'heading': ProtoParser.get_float32(msg, 7, 0.0),
        'vx': ProtoParser.get_float32(msg, 8, 0.0),
        'vy': ProtoParser.get_float32(msg, 9, 0.0),
        'valid': ProtoParser.get_int(msg, 10, 1) != 0,
    }


def parse_map_feature(data: bytes) -> Optional[Dict]:
    """
    Parse MapFeature message.
    
    Fields: 1=id, 2=lane, 3=road_line, 4=road_edge, 5=stop_sign, 6=crosswalk, 7=speed_bump
    """
    msg = ProtoParser.parse(data)
    
    feat_id = ProtoParser.get_int(msg, 1, 0)
    
    # Determine type and extract polyline
    type_fields = [
        (2, 'LANE'), 
        (3, 'ROAD_LINE'), 
        (4, 'ROAD_EDGE'), 
        (5, 'STOP_SIGN'), 
        (6, 'CROSSWALK'), 
        (7, 'SPEED_BUMP'),
        (8, 'DRIVEWAY'),
    ]
    
    for field_num, type_name in type_fields:
        feat_bytes = ProtoParser.get_bytes(msg, field_num)
        if feat_bytes:
            polyline = extract_polyline(feat_bytes)
            if polyline:
                return {
                    'id': feat_id, 
                    'type': type_name, 
                    'polyline': np.array(polyline)
                }
    
    return None


def extract_polyline(data: bytes) -> List[List[float]]:
    """Extract polyline points from lane/road_line/etc message."""
    msg = ProtoParser.parse(data)
    points = []
    
    # Field 1 is usually the polyline/polygon (repeated MapPoint)
    point_messages = ProtoParser.get_repeated_messages(msg, 1)
    for point_data in point_messages:
        try:
            point_msg = ProtoParser.parse(point_data)
            x = ProtoParser.get_float64(point_msg, 1, 0.0)
            y = ProtoParser.get_float64(point_msg, 2, 0.0)
            z = ProtoParser.get_float64(point_msg, 3, 0.0)
            points.append([x, y, z])
        except Exception:
            continue
    
    return points


# =============================================================================
# LOADER CLASS
# =============================================================================

class WaymoLoader:
    """Load Waymo scenarios from TFRecord files."""
    
    def load_all(self, filepath: str) -> Iterator[Dict]:
        """Iterate over all scenarios in file."""
        for i, record in enumerate(read_tfrecord(filepath)):
            try:
                scenario = parse_scenario(record)
                # Validate we got something useful
                if scenario['tracks']:
                    yield scenario
                else:
                    logger.debug(f"Scenario {i} has no tracks, skipping")
            except Exception as e:
                logger.warning(f"Failed to parse scenario {i}: {e}")
    
    def load_one(self, filepath: str, index: int = 0) -> Dict:
        """Load specific scenario by index."""
        for i, scenario in enumerate(self.load_all(filepath)):
            if i == index:
                return scenario
        raise IndexError(f"Scenario {index} not found (or failed to parse)")
    
    def count(self, filepath: str) -> int:
        """Count raw records in file (including unparseable ones)."""
        return sum(1 for _ in read_tfrecord(filepath))
    
    def count_valid(self, filepath: str) -> int:
        """Count successfully parsed scenarios."""
        return sum(1 for _ in self.load_all(filepath))
    
    def list_scenarios(self, filepath: str, max_items: int = 30) -> List[Dict]:
        """Get summary info for scenarios."""
        summaries = []
        for i, scenario in enumerate(self.load_all(filepath)):
            if i >= max_items:
                break
            ts = scenario['timestamps']
            summaries.append({
                'index': i,
                'id': scenario['id'][:50] if scenario['id'] else f"scenario_{i}",
                'agents': len(scenario['tracks']),
                'steps': len(ts),
                'duration': float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
            })
        return summaries


# =============================================================================
# VISUALIZER
# =============================================================================

def visualize_scenario(
    scenario: Dict,
    output_dir: str = "./screenshots",
    num_frames: int = 8,
    view_range: float = 50.0
) -> Tuple[List[Tuple[str, float]], np.ndarray, str]:
    """
    Render scenario frames and save as images.
    
    Returns: (saved_images, ego_trajectory, scenario_id)
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.transforms import Affine2D
    
    # Extract data
    scenario_id = scenario['id']
    tracks = scenario['tracks']
    map_features = scenario['map_features']
    sdc_id = scenario['metadata']['sdc_id']
    timestamps = scenario['timestamps']
    
    # Get SDC trajectory
    if sdc_id is None or sdc_id not in tracks:
        # Fallback to first track
        sdc_id = next(iter(tracks.keys())) if tracks else None
    
    if sdc_id is None:
        logger.error("No tracks found in scenario")
        return [], np.zeros((1, 4)), scenario_id
    
    sdc = tracks[sdc_id]
    state = sdc['state']
    pos = state['position']
    head = state['heading']
    vel = state['velocity']
    valid = state['valid']
    
    T = len(pos)
    speeds = np.linalg.norm(vel, axis=1)
    ego_traj = np.column_stack([pos[:, 0], pos[:, 1], head, speeds])
    
    # Select frames
    frame_steps = np.linspace(0, T - 1, num_frames, dtype=int).tolist()
    
    # Setup output
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    saved = []
    
    for step in frame_steps:
        ts = timestamps[step] if step < len(timestamps) else step * 0.1
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')
        
        # Center on ego
        cx = pos[step, 0] if valid[step] else np.nanmean(pos[:, 0])
        cy = pos[step, 1] if valid[step] else np.nanmean(pos[:, 1])
        
        # Draw map
        for fid, feat in map_features.items():
            poly = feat['polyline']
            if len(poly) < 2:
                continue
            # Skip if not in view
            if not np.any((np.abs(poly[:, 0] - cx) < view_range * 1.5) &
                         (np.abs(poly[:, 1] - cy) < view_range * 1.5)):
                continue
            
            colors = {'LANE': '#404060', 'ROAD_LINE': '#ffff00', 
                     'ROAD_EDGE': '#808080', 'CROSSWALK': '#aaaaaa'}
            ax.plot(poly[:, 0], poly[:, 1], 
                   color=colors.get(feat['type'], '#444444'),
                   linewidth=0.8, alpha=0.6)
        
        # Draw agents
        for tid, track in tracks.items():
            st = track['state']
            p = st['position']
            h = st['heading']
            v = st['valid']
            L = st['length']
            W = st['width']
            
            if step >= len(v) or not v[step]:
                continue
            
            x, y = p[step, 0], p[step, 1]
            if abs(x - cx) > view_range or abs(y - cy) > view_range:
                continue
            
            is_ego = (tid == sdc_id)
            color = '#ff3333' if is_ego else {'PEDESTRIAN': '#33ff33', 'CYCLIST': '#ff9933'}.get(track['type'], '#3399ff')
            
            length = L[step] if step < len(L) else 4.5
            width = W[step] if step < len(W) else 2.0
            heading = h[step]
            
            # Draw rectangle
            rect = patches.FancyBboxPatch(
                (-length/2, -width/2), length, width,
                boxstyle="round,rounding_size=0.3",
                facecolor=color, edgecolor='white',
                linewidth=1.5 if is_ego else 0.5, alpha=0.9,
                zorder=100 if is_ego else 50
            )
            rect.set_transform(Affine2D().rotate(heading).translate(x, y) + ax.transData)
            ax.add_patch(rect)
            
            # Heading arrow
            ax.arrow(x, y, (length/2)*np.cos(heading), (length/2)*np.sin(heading),
                    head_width=0.4, head_length=0.2, fc='white', ec='white',
                    zorder=101 if is_ego else 51)
        
        # Draw ego trajectory
        past = valid[:step+1]
        if np.any(past):
            ax.plot(pos[:step+1, 0][past], pos[:step+1, 1][past],
                   color='#00ff88', linewidth=2, alpha=0.8, zorder=90)
        future = valid[step:]
        if np.any(future):
            ax.plot(pos[step:, 0][future], pos[step:, 1][future],
                   color='#ff00ff', linewidth=2, alpha=0.5, linestyle='--', zorder=89)
        
        ax.set_xlim(cx - view_range, cx + view_range)
        ax.set_ylim(cy - view_range, cy + view_range)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"t = {ts:.2f}s", color='white', fontsize=12)
        
        fname = f"frame_{ts:.2f}.png"
        fpath = out_path / fname
        fig.savefig(fpath, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        
        saved.append((str(fpath), float(ts)))
        logger.info(f"Saved {fname}")
    
    return saved, ego_traj, scenario_id


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def prepare_for_vlm(
    tfrecord_path: str,
    scenario_index: int = 0,
    output_dir: str = "./screenshots",
    num_frames: int = 8,
    view_range: float = 50.0
) -> Tuple[List[Tuple[str, float]], np.ndarray, str]:
    """
    Load Waymo scenario and generate screenshots for VLM analysis.
    """
    loader = WaymoLoader()
    scenario = loader.load_one(tfrecord_path, scenario_index)
    logger.info(f"Loaded scenario: {scenario['id']}")
    logger.info(f"  Agents: {len(scenario['tracks'])}, Steps: {scenario['metadata']['num_timesteps']}")
    
    return visualize_scenario(scenario, output_dir, num_frames, view_range)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Waymo TFRecord loader (no TensorFlow)")
    parser.add_argument("tfrecord", help="Path to .tfrecord file")
    parser.add_argument("--list", action="store_true", help="List scenarios")
    parser.add_argument("--count", action="store_true", help="Count scenarios")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario index")
    parser.add_argument("--output", default="./screenshots", help="Output directory")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames")
    parser.add_argument("--view-range", type=float, default=50.0, help="View range (m)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    loader = WaymoLoader()
    
    if args.count:
        n_raw = loader.count(args.tfrecord)
        n_valid = loader.count_valid(args.tfrecord)
        print(f"Raw records: {n_raw}")
        print(f"Valid scenarios: {n_valid}")
    
    elif args.list:
        print(f"\nScenarios in {args.tfrecord}:\n" + "-"*70)
        summaries = loader.list_scenarios(args.tfrecord)
        if not summaries:
            print("No scenarios could be parsed. Try --debug for more info.")
        else:
            for s in summaries:
                print(f"[{s['index']:3d}] {s['id']}  |  {s['duration']:.1f}s  |  {s['agents']} agents")
            print(f"\n(Showing {len(summaries)} scenarios)")
    
    else:
        try:
            saved, traj, sid = prepare_for_vlm(
                args.tfrecord, args.scenario, args.output, args.frames, args.view_range
            )
            print(f"\n✓ Saved {len(saved)} frames to {args.output}/")
            print(f"  Scenario: {sid}")
            print(f"  Trajectory: {traj.shape}")
            
            print(f"\nTo use with VLM extractor:")
            print(f"  from vlm_extractor import VLMSafetyCriticalExtractor, TimestampedImage")
            print(f"  images = [TimestampedImage(p, t) for p, t in saved_images]")
        except IndexError as e:
            print(f"Error: {e}")
            print("Use --list to see available scenarios")
        except Exception as e:
            logger.exception(f"Error: {e}")


if __name__ == "__main__":
    main()