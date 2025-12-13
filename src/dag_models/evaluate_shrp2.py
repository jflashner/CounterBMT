#!/usr/bin/env python3
"""
SynSHRP2 Evaluation Script with Comprehensive Logging

Usage:
    export OPENAI_API_KEY="your-api-key"  # Optional
    python evaluate_shrp2.py --data_dir /path/to/synSHRP2 --log_level DEBUG
"""

import os
import sys
import json
import base64
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import Counter
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.veacon_schema import (
    VeaconEvent, Environment, VehicleState, Accident,
    CrashDynamics, CrashState, InjuryOutcome,
    WeatherCondition, LightCondition, SurfaceCondition,
    RoadType, ConflictType, EventType, BrakingEffectiveness,
    PointOfImpact, Visibility, TrafficDensity
)
from models.causal_dag import CausalDAG, create_default_dag
from models.llm_world_model import LLMWorldModel
from models.safety_analyzer import SafetyCriticalAnalyzer


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with console and file output."""
    logger = logging.getLogger("synSHRP2_eval")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console)
    
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    
    return logger

logger = logging.getLogger("synSHRP2_eval")


# =============================================================================
# DIAGNOSTIC DATACLASSES
# =============================================================================

@dataclass
class DatasetDiagnostics:
    total_events: int = 0
    crashes: int = 0
    near_crashes: int = 0
    other_types: int = 0
    with_keyframes: int = 0
    with_kinematics: int = 0
    columns_found: List[str] = field(default_factory=list)
    event_type_distribution: Dict[str, int] = field(default_factory=dict)
    severity_distribution: Dict[str, int] = field(default_factory=dict)
    incident_type_distribution: Dict[str, int] = field(default_factory=dict)
    events_with_speed: int = 0
    events_with_brake_data: int = 0
    kinematic_keys_found: Dict[str, int] = field(default_factory=dict)
    parsing_warnings: List[str] = field(default_factory=list)


@dataclass 
class EvaluationDiagnostics:
    classification_details: List[Dict] = field(default_factory=list)
    severity_details: List[Dict] = field(default_factory=list)
    consistency_details: Dict = field(default_factory=dict)
    safety_details: List[Dict] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)


# =============================================================================
# IMPORTS
# =============================================================================

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# =============================================================================
# GPT-4o CLIENT
# =============================================================================

class GPT4oClient:
    def __init__(self, api_key: Optional[str] = None):
        if not HAS_OPENAI:
            raise ImportError("openai required")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"
        self.call_count = 0
        self.call_log = []  # Detailed log of all calls
    
    def complete(self, prompt: str, images: Optional[List[str]] = None,
                 temperature: float = 0.3, max_tokens: int = 1000) -> str:
        content = []
        if images:
            for img in images:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img[:50]}...", "detail": "high"}})
        content.append({"type": "text", "text": prompt})
        
        self.call_count += 1
        call_id = self.call_count
        
        logger.debug(f"=" * 50)
        logger.debug(f"GPT-4o CALL #{call_id}")
        logger.debug(f"Images: {len(images) if images else 0}")
        logger.debug(f"Temperature: {temperature}")
        logger.debug(f"Prompt:\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}")
        logger.debug(f"=" * 50)
        
        # Build actual content for API
        actual_content = []
        if images:
            for img in images:
                actual_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"}})
        actual_content.append({"type": "text", "text": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model, 
                messages=[{"role": "user", "content": actual_content}],
                temperature=temperature, 
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content
            
            logger.debug(f"GPT-4o RESPONSE #{call_id}:\n{result}")
            
            # Log for diagnostics
            self.call_log.append({
                "call_id": call_id,
                "n_images": len(images) if images else 0,
                "prompt_preview": prompt[:200],
                "response_preview": result[:300],
                "temperature": temperature
            })
            
            return result
            
        except Exception as e:
            logger.error(f"GPT-4o CALL #{call_id} FAILED: {e}")
            self.call_log.append({
                "call_id": call_id,
                "error": str(e)
            })
            raise
    
    def get_call_log(self) -> List[Dict]:
        """Return log of all API calls for diagnostics."""
        return self.call_log


# =============================================================================
# VLM EXTRACTOR
# =============================================================================

class GPT4oVLMExtractor:
    PROMPT = """Analyze these dashcam images showing a driving event. Images are in chronological order.

Look for these crash indicators:
- Vehicle damage or deformation visible
- Collision with another vehicle/object
- Airbag deployment
- Debris on road
- Stopped vehicles at unusual angles
- Impact marks or scrapes

Look for near-crash indicators (no actual collision):
- Hard braking (nose dive) 
- Evasive swerving
- Close call with obstacle
- No visible damage or contact

Respond ONLY with JSON:
{
    "weather": "clear" | "rain" | "snow" | "fog",
    "light": "daylight" | "dark" | "dark_lighted" | "dawn" | "dusk", 
    "surface_condition": "dry" | "wet" | "snow_ice",
    "road_type": "highway" | "urban" | "rural" | "intersection",
    "event_type": "crash" | "near_crash",
    "crash_evidence": "describe what indicates crash or near-crash",
    "confidence": 0.0-1.0
}

IMPORTANT: Only classify as "crash" if you see clear evidence of collision/impact.
If unsure, classify as "near_crash"."""

    def __init__(self, client: GPT4oClient):
        self.client = client
        self.extraction_log = []  # Log all extractions for diagnostics
    
    def extract_from_keyframes(self, keyframe_dir: str) -> Dict[str, Any]:
        kf_path = Path(keyframe_dir)
        images = []
        frames_found = []
        
        for name in ["Prior.jpg", "Start.jpg", "Reaction.jpg", "Impact.jpg", "End.jpg"]:
            p = kf_path / name
            if p.exists():
                with open(p, "rb") as f:
                    images.append(base64.b64encode(f.read()).decode())
                    frames_found.append(name)
        
        if not images:
            return {"error": "No keyframes"}
        
        logger.debug(f"VLM analyzing {kf_path.name}: frames={frames_found}")
        
        response = self.client.complete(self.PROMPT, images, 0.1)
        
        # Log full raw response
        logger.debug(f"VLM raw response for {kf_path.name}:\n{response}")
        
        try:
            resp = response.strip()
            if resp.startswith("```"):
                resp = "\n".join(resp.split("\n")[1:-1])
            result = json.loads(resp)
            
            # Log parsed result
            logger.info(f"VLM {kf_path.name}: "
                       f"event_type={result.get('event_type')}, "
                       f"confidence={result.get('confidence')}, "
                       f"weather={result.get('weather')}, "
                       f"light={result.get('light')}")
            logger.info(f"VLM evidence: {result.get('crash_evidence', 'N/A')[:100]}")
            
            # Store in log
            self.extraction_log.append({
                "keyframe_dir": str(kf_path.name),
                "frames_found": frames_found,
                "result": result,
                "raw_response": response[:500]
            })
            
            return result
        except Exception as e:
            logger.error(f"VLM parse error for {kf_path.name}: {e}")
            logger.error(f"VLM unparseable response: {response[:300]}")
            
            self.extraction_log.append({
                "keyframe_dir": str(kf_path.name),
                "frames_found": frames_found,
                "error": str(e),
                "raw_response": response[:500]
            })
            
            return {"error": "Parse failed", "raw": response[:200]}
    
    def get_extraction_log(self) -> List[Dict]:
        """Return log of all VLM extractions."""
        return self.extraction_log


# =============================================================================
# GPT-4o WORLD MODEL WITH PHYSICS CONSTRAINTS
# =============================================================================

class GPT4oWorldModel(LLMWorldModel):
    """LLM World Model using GPT-4o with physics-constrained prompting."""
    
    PHYSICS_RULES = """
CRITICAL PHYSICS CONSTRAINTS (absolute laws - never violate):
1. LOWER speed → EQUAL or LOWER severity (never higher)
2. LOWER delta-V → EQUAL or LOWER severity
3. BETTER braking → LOWER impact speed
4. LOWER impact speed → EQUAL or LOWER delta-V
These are monotonic from crash physics. Violations are physically impossible.
"""
    
    # Nodes that should NOT be modified during counterfactual propagation
    # These are scenario descriptors, not causal outcomes
    FIXED_NODES = {
        'accident.event_type',      # If it's a crash, it stays a crash
        'accident.conflict_type',   # The type of conflict doesn't change
        'environment.weather',      # Weather is exogenous
        'environment.light',        # Light condition is exogenous  
        'environment.road_type',    # Road type is exogenous
        'crash_state.point_of_impact',  # Impact location is scenario-specific
    }
    
    # Nodes that must return specific categorical values
    CATEGORICAL_NODES = {
        'accident.event_type': ['crash', 'near_crash', 'normal'],
        'crash_dynamics.braking_effectiveness': ['none', 'low', 'medium', 'high', 'unknown'],
        'environment.surface_condition': ['dry', 'wet', 'snow_ice', 'unknown'],
        'environment.visibility': ['good', 'moderate', 'poor', 'unknown'],
    }
    
    def __init__(self, client: GPT4oClient, dag: Optional[CausalDAG] = None):
        super().__init__(dag=dag, llm_client=None)
        self.gpt4o_client = client
        self.reasoning_log = []  # Store reasoning for diagnostics
    
    def _query_llm_for_node(self, node: str, parent_values: Dict, current_state: Dict, verbose: bool = False) -> Any:
        """Query LLM for node value, with validation for categorical nodes."""
        
        # Skip fixed nodes - return original value
        if node in self.FIXED_NODES:
            logger.debug(f"WorldModel {node}: FIXED (keeping original value)")
            return current_state.get(node)
        
        dag_node = self.dag.get_node(node)
        if not dag_node:
            return current_state.get(node)
        
        # For categorical nodes, use rule-based logic instead of LLM
        if node in self.CATEGORICAL_NODES:
            logger.debug(f"WorldModel {node}: Categorical, using rule-based")
            return self._rule_based_propagation(node, parent_values, current_state)
        
        # Build physics-aware prompt for numeric nodes
        prompt = self._build_physics_prompt(node, parent_values, dag_node, current_state)
        
        logger.debug(f"WorldModel querying node: {node}")
        logger.debug(f"WorldModel parent values: {json.dumps(parent_values, default=str)}")
        
        try:
            response = self.gpt4o_client.complete(prompt, temperature=0.1, max_tokens=400)
            
            # Log raw response
            logger.debug(f"WorldModel raw response for {node}:\n{response}")
            
            import re
            match = re.search(r'\{[^{}]*"value"[^{}]*\}', response, re.DOTALL)
            if match:
                result = json.loads(match.group())
                value = result.get("value")
                reasoning = result.get("reasoning", "no reasoning provided")
                
                # Validate numeric value
                if not isinstance(value, (int, float)):
                    logger.warning(f"WorldModel {node}: Non-numeric value '{value}', using rule-based")
                    value = self._rule_based_propagation(node, parent_values, current_state)
                    reasoning = "Fell back to rule-based (non-numeric LLM response)"
                
                # Log the reasoning
                logger.info(f"WorldModel {node}: value={value}, reasoning='{reasoning[:100]}...' " 
                           if len(str(reasoning)) > 100 else f"WorldModel {node}: value={value}, reasoning='{reasoning}'")
                
                # Store for diagnostics
                self.reasoning_log.append({
                    "node": node,
                    "parents": parent_values,
                    "value": value,
                    "reasoning": reasoning,
                    "raw_response": response[:300]
                })
                
                # Validate physics for severity
                validated = self._validate_physics(node, value, parent_values, current_state)
                if validated is not None:
                    self.reasoning_log[-1]["physics_corrected"] = True
                    self.reasoning_log[-1]["original_value"] = value
                    self.reasoning_log[-1]["corrected_value"] = validated
                    return validated
                return value
            else:
                logger.warning(f"WorldModel {node}: Could not parse JSON from response")
                logger.debug(f"Full response: {response}")
                
        except Exception as e:
            logger.error(f"WorldModel error for {node}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Fallback to rules
        logger.debug(f"WorldModel {node}: Falling back to rule-based propagation")
        fallback_value = self._rule_based_propagation(node, parent_values, current_state)
        logger.debug(f"WorldModel {node}: Rule-based result = {fallback_value}")
        return fallback_value
    
    def _build_physics_prompt(self, node: str, parent_values: Dict, dag_node: Any, current_state: Dict) -> str:
        """Build prompt with physics constraints."""
        
        constraint = ""
        if "severity" in node.lower():
            orig_speed = current_state.get("vehicle_state.pre_crash_speed_kph") or 0
            orig_sev = current_state.get("injury_outcome.severity_score") or 0
            new_speed = parent_values.get("crash_state.speed_at_impact_kph") or parent_values.get("vehicle_state.pre_crash_speed_kph") or orig_speed
            
            if new_speed < orig_speed and orig_sev > 0:
                constraint = f"""
MANDATORY CONSTRAINT:
- Original: speed={orig_speed:.0f}kph, severity={orig_sev:.2f}
- New speed={new_speed:.0f}kph is LOWER
- Therefore severity MUST be ≤ {orig_sev:.2f}
"""
        
        prompt = f"""You are a crash physics expert.
{self.PHYSICS_RULES}{constraint}
Given: {json.dumps(parent_values, indent=2)}

Calculate: {node} ({dag_node.description})
Type: {dag_node.value_type}

IMPORTANT: Return a NUMERIC value only. 
Respond ONLY: {{"reasoning": "brief", "value": <number>}}

For severity_score: use 0.0 (none) to 1.0 (fatal).
For speed values: use kph.
For acceleration: use m/s²."""

        logger.debug(f"WorldModel prompt for {node}:\n{prompt}")
        return prompt
    
    def _validate_physics(self, node: str, value: Any, parent_values: Dict, current_state: Dict) -> Optional[Any]:
        """Hard-enforce physics constraints."""
        
        if "severity" not in node.lower() or not isinstance(value, (int, float)):
            return None
        
        orig_speed = current_state.get("vehicle_state.pre_crash_speed_kph") or 0
        new_speed = parent_values.get("crash_state.speed_at_impact_kph") or \
                   parent_values.get("vehicle_state.pre_crash_speed_kph") or orig_speed
        orig_sev = current_state.get("injury_outcome.severity_score") or 0
        
        # If speed decreased but severity increased, FORCE correction
        if new_speed < orig_speed and value > orig_sev and orig_speed > 0:
            ratio = new_speed / orig_speed
            corrected = round(orig_sev * ratio, 2)
            logger.warning(f"PHYSICS FIX: speed {orig_speed:.0f}→{new_speed:.0f}, "
                          f"severity {orig_sev:.2f}→{value:.2f} CORRECTED to {corrected:.2f}")
            return corrected
        
        return None
    
    def get_reasoning_log(self) -> List[Dict]:
        """Return all reasoning logs for diagnostics."""
        return self.reasoning_log
    
    def clear_reasoning_log(self):
        """Clear reasoning log."""
        self.reasoning_log = []


# =============================================================================
# SynSHRP2 EVENT
# =============================================================================

@dataclass
class SynSHRP2Event:
    event_id: str
    event_start_ms: int
    reaction_start_ms: Optional[int]
    impact_ms: Optional[int]
    event_end_ms: int
    event_type: str
    crash_severity: str
    incident_type: str
    conflict_type: str
    narrative: Optional[str]
    raw_row: Optional[Dict] = None
    kinematics: Optional[List[Dict]] = None
    keyframe_dir: Optional[str] = None
    
    def has_impact(self) -> bool:
        if self.impact_ms is None:
            return False
        if isinstance(self.impact_ms, float) and np.isnan(self.impact_ms):
            return False
        return True


# =============================================================================
# DATA LOADER
# =============================================================================

class SynSHRP2DataLoader:
    SEVERITY_TO_KABCO = {
        # Exact matches from SynSHRP2 dataset
        "i - most severe": 4,
        "ii - police-reportable crash": 3,
        "ii - police reportable": 3,
        "iii - minor crash": 2,   # handles both single and double space
        "iii  - minor crash": 2,  # explicit double space
        "iv - low-risk tire strike": 1,
        "iv - low risk": 1,
        "not a crash": None,
    }
    
    INCIDENT_TO_CONFLICT = {
        "rear-end, striking": ConflictType.LEAD_VEHICLE,
        "rear-end, struck": ConflictType.FOLLOWING_VEHICLE,
        "sideswipe": ConflictType.ADJACENT_VEHICLE,
        "road departure": ConflictType.SINGLE_VEHICLE,
        "pedestrian": ConflictType.PEDESTRIAN,
        "animal": ConflictType.ANIMAL,
        "object": ConflictType.FIXED_OBJECT,
    }
    
    @classmethod
    def load_dataset(cls, data_dir: str) -> Tuple[List[SynSHRP2Event], DatasetDiagnostics]:
        if not HAS_PANDAS:
            raise ImportError("pandas required")
        
        diag = DatasetDiagnostics()
        data_path = Path(data_dir)
        
        # Find CSV
        csv_path = None
        for name in ["Tabular records.csv", "Tabular_records.csv", "tabular_records.csv"]:
            if (data_path / name).exists():
                csv_path = data_path / name
                break
        if not csv_path:
            raise FileNotFoundError(f"No CSV in {data_dir}")
        
        logger.info(f"Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        diag.columns_found = list(df.columns)
        logger.info(f"Columns: {diag.columns_found}")
        
        # Rename columns
        col_map = {'Event ID': 'event_id', 'Event_ID': 'event_id',
                   'Event start': 'event_start', 'Event_start': 'event_start',
                   'Reaction start': 'reaction_start', 'Impact': 'impact',
                   'Event end': 'event_end', 'Event_end': 'event_end',
                   'Event type': 'event_type', 'Event_type': 'event_type',
                   'Crash severity': 'crash_severity', 'Crash_severity': 'crash_severity',
                   'Incident type': 'incident_type', 'Incident_type': 'incident_type',
                   'Conflict type': 'conflict_type', 'Narrative': 'narrative'}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        # Find directories
        kin_dir = kf_dir = None
        for n in ["Kinematic signals", "Kinematic_signals"]:
            if (data_path / n).exists(): kin_dir = data_path / n; break
        for n in ["Keyframes", "keyframes"]:
            if (data_path / n).exists(): kf_dir = data_path / n; break
        
        events = []
        all_kin_keys = Counter()
        
        for _, row in df.iterrows():
            eid = str(row.get('event_id', row.name))
            
            def safe_int(v):
                try: return int(float(v)) if pd.notna(v) else None
                except: return None
            def safe_str(v):
                return str(v).strip() if pd.notna(v) else ""
            
            et = safe_str(row.get('event_type', ''))
            sev = safe_str(row.get('crash_severity', ''))
            inc = safe_str(row.get('incident_type', ''))
            
            diag.event_type_distribution[et] = diag.event_type_distribution.get(et, 0) + 1
            diag.severity_distribution[sev] = diag.severity_distribution.get(sev, 0) + 1
            diag.incident_type_distribution[inc] = diag.incident_type_distribution.get(inc, 0) + 1
            
            event = SynSHRP2Event(
                event_id=eid,
                event_start_ms=safe_int(row.get('event_start', 0)) or 0,
                reaction_start_ms=safe_int(row.get('reaction_start')),
                impact_ms=safe_int(row.get('impact')),
                event_end_ms=safe_int(row.get('event_end', 0)) or 0,
                event_type=et, crash_severity=sev, incident_type=inc,
                conflict_type=safe_str(row.get('conflict_type', '')),
                narrative=safe_str(row.get('narrative')) or None,
                raw_row=row.to_dict()
            )
            
            # Load kinematics
            if kin_dir and (kin_dir / f"{eid}.json").exists():
                try:
                    with open(kin_dir / f"{eid}.json") as f:
                        event.kinematics = json.load(f)
                        diag.with_kinematics += 1
                        if event.kinematics:
                            for k in event.kinematics[0].keys():
                                all_kin_keys[k] += 1
                            if any(s.get('Speed') for s in event.kinematics):
                                diag.events_with_speed += 1
                            for bk in ['Ped_BS', 'Ped BS', 'PedBS', 'Brake', 'brake']:
                                if any(s.get(bk) is not None for s in event.kinematics):
                                    diag.events_with_brake_data += 1
                                    break
                except Exception as e:
                    diag.parsing_warnings.append(f"{eid}: {e}")
            
            # Keyframes
            if kf_dir and (kf_dir / eid).is_dir():
                event.keyframe_dir = str(kf_dir / eid)
                diag.with_keyframes += 1
            
            events.append(event)
            
            # Count types
            if "crash" in et.lower() and "near" not in et.lower():
                diag.crashes += 1
            elif "near" in et.lower():
                diag.near_crashes += 1
            else:
                diag.other_types += 1
        
        diag.total_events = len(events)
        diag.kinematic_keys_found = dict(all_kin_keys)
        return events, diag
    
    @classmethod
    def extract_kinematics_features(cls, event: SynSHRP2Event) -> Dict[str, Any]:
        if not event.kinematics:
            return {}
        
        features = {}
        kin = event.kinematics
        
        # Speed
        speeds = [k.get('Speed') for k in kin if k.get('Speed') is not None]
        if speeds:
            features['max_speed_mph'] = max(speeds)
            features['pre_event_speed_mph'] = speeds[0]
            features['avg_speed_mph'] = np.mean(speeds)
            
            if event.has_impact():
                for i, k in enumerate(kin):
                    ts = k.get('TimeStamp', i)
                    if ts >= event.impact_ms and k.get('Speed') is not None:
                        features['impact_speed_mph'] = k['Speed']
                        break
        
        # Acceleration - try underscore variant first
        lon_accs = [k.get('Lon_Acc') or k.get('Lon Acc') for k in kin]
        lon_accs = [x for x in lon_accs if x is not None]
        if lon_accs:
            features['max_braking_g'] = abs(min(lon_accs))
        
        # Brake - use underscore variant first
        for bk in ['Ped_BS', 'Ped BS', 'PedBS', 'Brake']:
            bd = [k.get(bk) for k in kin if k.get(bk) is not None]
            if bd:
                features['brake_applied'] = any(b == 1 or b == True for b in bd)
                logger.debug(f"Found brake data under key '{bk}'")
                break
        
        # Delta-V
        if 'impact_speed_mph' in features and speeds:
            post = [k.get('Speed') for k in kin 
                   if k.get('TimeStamp', 0) > (event.impact_ms or 0) and k.get('Speed') is not None]
            if post:
                features['delta_v_mph'] = features['impact_speed_mph'] - min(post)
        
        logger.debug(f"Event {event.event_id} kinematics: {list(features.keys())}")
        return features
    
    @classmethod
    def get_kabco(cls, severity_str: str) -> Optional[int]:
        # Normalize: lowercase and collapse multiple spaces
        s = ' '.join(severity_str.lower().strip().split())
        
        for pattern, kabco in cls.SEVERITY_TO_KABCO.items():
            # Also normalize pattern
            pattern_norm = ' '.join(pattern.split())
            if pattern_norm in s or s in pattern_norm:
                return kabco
        
        # Fallback patterns using Roman numerals
        if s.startswith("i -") or "most severe" in s: return 4
        if s.startswith("ii -") or "police" in s: return 3
        if s.startswith("iii") or "minor crash" in s: return 2
        if s.startswith("iv") or "low-risk" in s or "tire strike" in s: return 1
        if "not a crash" in s: return None
        
        logger.debug(f"Unknown severity: '{severity_str}' -> normalized: '{s}'")
        return None
    
    @classmethod
    def to_veacon(cls, event: SynSHRP2Event, vlm: Optional[Dict] = None) -> VeaconEvent:
        kin = cls.extract_kinematics_features(event)
        
        # Event type
        et_lower = event.event_type.lower()
        if "crash" in et_lower and "near" not in et_lower:
            event_type = EventType.CRASH
        elif "near" in et_lower:
            event_type = EventType.NEAR_CRASH
        else:
            event_type = EventType.NORMAL
        
        # Conflict
        conflict = ConflictType.UNKNOWN
        inc_lower = event.incident_type.lower()
        for pattern, ct in cls.INCIDENT_TO_CONFLICT.items():
            if pattern in inc_lower:
                conflict = ct
                break
        
        # KABCO
        kabco = cls.get_kabco(event.crash_severity)
        severity_score = kabco / 4.0 if kabco else 0.0
        
        # Braking
        braking = BrakingEffectiveness.UNKNOWN
        if kin.get('brake_applied'):
            mb = kin.get('max_braking_g', 0)
            if mb > 0.6: braking = BrakingEffectiveness.HIGH
            elif mb > 0.3: braking = BrakingEffectiveness.MEDIUM
            elif mb > 0.1: braking = BrakingEffectiveness.LOW
            else: braking = BrakingEffectiveness.NONE
        elif kin.get('brake_applied') is False:
            braking = BrakingEffectiveness.NONE
        
        # Point of impact
        poi = PointOfImpact.UNKNOWN
        if "rear-end, striking" in inc_lower: poi = PointOfImpact.FRONT
        elif "rear-end, struck" in inc_lower: poi = PointOfImpact.REAR
        
        # Speeds
        pre_kph = kin.get('pre_event_speed_mph', 0) * 1.60934 if kin.get('pre_event_speed_mph') else None
        impact_kph = kin.get('impact_speed_mph', 0) * 1.60934 if kin.get('impact_speed_mph') else None
        dv_kph = kin.get('delta_v_mph', 0) * 1.60934 if kin.get('delta_v_mph') else None
        
        # VLM features
        weather, light, surface, road = WeatherCondition.UNKNOWN, LightCondition.UNKNOWN, SurfaceCondition.UNKNOWN, RoadType.UNKNOWN
        if vlm and 'error' not in vlm:
            wmap = {"clear": WeatherCondition.CLEAR, "rain": WeatherCondition.RAIN, "snow": WeatherCondition.SNOW}
            lmap = {"daylight": LightCondition.DAYLIGHT, "dark": LightCondition.DARK}
            smap = {"dry": SurfaceCondition.DRY, "wet": SurfaceCondition.WET}
            weather = wmap.get(vlm.get('weather', ''), WeatherCondition.UNKNOWN)
            light = lmap.get(vlm.get('light', ''), LightCondition.UNKNOWN)
            surface = smap.get(vlm.get('surface_condition', ''), SurfaceCondition.UNKNOWN)
        
        return VeaconEvent(
            event_id=event.event_id,
            environment=Environment(weather=weather, light=light, surface_condition=surface, road_type=road),
            vehicle_state=VehicleState(pre_crash_speed_kph=pre_kph),
            accident=Accident(conflict_type=conflict, event_type=event_type),
            crash_dynamics=CrashDynamics(braking_effectiveness=braking),
            crash_state=CrashState(speed_at_impact_kph=impact_kph, point_of_impact=poi, delta_v_kph=dv_kph),
            injury_outcome=InjuryOutcome(severity_score=severity_score, kabco_level=kabco)
        )


# =============================================================================
# EVALUATOR
# =============================================================================

class SynSHRP2Evaluator:
    def __init__(self, gpt4o_client: Optional[GPT4oClient] = None, use_vlm: bool = True):
        self.client = gpt4o_client
        self.use_vlm = use_vlm and gpt4o_client is not None
        
        if self.client:
            self.vlm = GPT4oVLMExtractor(gpt4o_client)
            self.world_model = GPT4oWorldModel(gpt4o_client)
        else:
            self.vlm = None
            self.world_model = LLMWorldModel()
        
        self.safety_analyzer = SafetyCriticalAnalyzer(world_model=self.world_model)
        self.diag = EvaluationDiagnostics()
    
    def evaluate_classification(self, events: List[SynSHRP2Event]) -> Dict:
        from sklearn.metrics import classification_report, confusion_matrix
        
        y_true, y_pred = [], []
        
        for e in tqdm(events, desc="Classification"):
            # Ground truth
            et = e.event_type.lower()
            if "crash" in et and "near" not in et:
                gt = "crash"
            elif "near" in et:
                gt = "near_crash"
            else:
                gt = "other"
            y_true.append(gt)
            
            # Prediction
            if self.use_vlm and e.keyframe_dir:
                try:
                    feat = self.vlm.extract_from_keyframes(e.keyframe_dir)
                    pred_raw = feat.get('event_type', 'unknown')
                    pred = pred_raw.lower().replace("-", "_")
                    if "crash" in pred and "near" not in pred: pred = "crash"
                    elif "near" in pred: pred = "near_crash"
                    else: pred = "near_crash"
                    
                    # Store VLM evidence for diagnostics
                    vlm_evidence = feat.get('crash_evidence', 'N/A')
                except Exception as ex:
                    logger.error(f"VLM error for {e.event_id}: {ex}")
                    pred = "near_crash"
                    vlm_evidence = f"ERROR: {ex}"
            else:
                # Without VLM, use heuristics from data
                kin = SynSHRP2DataLoader.extract_kinematics_features(e)
                delta_v = kin.get('delta_v_mph', 0)
                has_significant_impact = e.has_impact() and delta_v > 15
                
                if has_significant_impact:
                    pred = "crash"
                else:
                    pred = "near_crash"
                
                vlm_evidence = f"heuristic: has_impact={e.has_impact()}, delta_v={delta_v:.1f}"
            
            y_pred.append(pred)
            
            self.diag.classification_details.append({
                "event_id": e.event_id, "gt": gt, "pred": pred,
                "correct": gt == pred, "raw_type": e.event_type,
                "evidence": vlm_evidence[:100] if vlm_evidence else "N/A"
            })
        
        # Filter valid
        valid = [(t, p) for t, p in zip(y_true, y_pred) if t in ["crash", "near_crash"]]
        if not valid:
            return {"error": "No valid events"}
        
        yt, yp = zip(*valid)
        labels = ["crash", "near_crash"]
        report = classification_report(yt, yp, labels=labels, output_dict=True, zero_division=0)
        cm = confusion_matrix(yt, yp, labels=labels)
        
        logger.info(f"Classification CM: crash[{cm[0,0]},{cm[0,1]}] near[{cm[1,0]},{cm[1,1]}]")
        
        return {
            "accuracy": report.get("accuracy", 0),
            "macro_f1": report.get("macro avg", {}).get("f1-score", 0),
            "per_class": {l: report.get(l, {}) for l in labels},
            "confusion_matrix": cm.tolist(),
            "n_evaluated": len(valid)
        }
    
    def evaluate_severity(self, events: List[SynSHRP2Event]) -> Dict:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        crashes = [e for e in events if "crash" in e.event_type.lower() and "near" not in e.event_type.lower()]
        logger.info(f"Severity: {len(crashes)} crash events")
        
        y_true, y_pred = [], []
        
        for e in tqdm(crashes, desc="Severity"):
            kabco = SynSHRP2DataLoader.get_kabco(e.crash_severity)
            if kabco is None:
                self.diag.severity_details.append({"event_id": e.event_id, "skipped": True, "severity_raw": e.crash_severity})
                continue
            
            y_true.append(kabco)
            
            kin = SynSHRP2DataLoader.extract_kinematics_features(e)
            dv = kin.get('delta_v_mph', 0) * 1.60934
            
            # Prediction
            if dv < 10: pred = 1
            elif dv < 25: pred = 2  
            elif dv < 45: pred = 3
            else: pred = 4
            
            y_pred.append(pred)
            self.diag.severity_details.append({
                "event_id": e.event_id, "true": kabco, "pred": pred,
                "delta_v": dv, "error": abs(kabco - pred)
            })
        
        if not y_true:
            return {"error": "No valid severity", "n_crashes": len(crashes)}
        
        logger.info(f"Severity: {len(y_true)} samples, true dist: {Counter(y_true)}")
        
        return {
            "n_samples": len(y_true),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "within_one": np.mean([abs(t-p) <= 1 for t, p in zip(y_true, y_pred)]),
            "exact": np.mean([t == p for t, p in zip(y_true, y_pred)]),
            "true_dist": dict(Counter(y_true)),
            "pred_dist": dict(Counter(y_pred))
        }
    
    def evaluate_consistency(self, events: List[SynSHRP2Event], n: int = 50) -> Dict:
        speed_tests, brake_tests = [], []
        
        # IMPORTANT: Only test on CRASH events, not near-crashes
        # Near-crashes have severity=0 because no crash occurred
        # Testing counterfactuals on them is meaningless
        crash_events = [e for e in events if "crash" in e.event_type.lower() and "near" not in e.event_type.lower()]
        
        logger.info(f"Consistency test: {len(crash_events)} crash events (of {len(events)} total)")
        
        for e in tqdm(crash_events[:n], desc="Consistency"):
            veacon = SynSHRP2DataLoader.to_veacon(e)
            
            # Skip if no severity (shouldn't happen for crashes, but safety check)
            if veacon.injury_outcome.severity_score <= 0:
                logger.debug(f"Event {e.event_id}: Skipping, severity={veacon.injury_outcome.severity_score}")
                continue
            
            # Clear reasoning log before each test
            if hasattr(self.world_model, 'clear_reasoning_log'):
                self.world_model.clear_reasoning_log()
            
            # Speed test
            if veacon.vehicle_state.pre_crash_speed_kph and veacon.vehicle_state.pre_crash_speed_kph > 40:
                orig = veacon.injury_outcome.severity_score
                new_speed = veacon.vehicle_state.pre_crash_speed_kph * 0.5
                
                logger.info(f"Speed test {e.event_id}: {veacon.vehicle_state.pre_crash_speed_kph:.0f} → {new_speed:.0f} kph, orig_sev={orig:.2f}")
                
                cf = self.world_model.propagate_intervention(veacon, {"vehicle_state.pre_crash_speed_kph": new_speed})
                new_sev = cf.injury_outcome.severity_score
                
                consistent = new_sev <= orig
                
                # Get reasoning log
                reasoning_log = []
                if hasattr(self.world_model, 'get_reasoning_log'):
                    reasoning_log = self.world_model.get_reasoning_log()
                
                # Extract severity reasoning specifically
                severity_reasoning = None
                for r in reasoning_log:
                    if 'severity' in r.get('node', '').lower():
                        severity_reasoning = r
                        break
                
                speed_tests.append({
                    "event_id": e.event_id,
                    "orig_speed": veacon.vehicle_state.pre_crash_speed_kph,
                    "new_speed": new_speed,
                    "orig_sev": orig, "new_sev": new_sev,
                    "consistent": consistent,
                    "llm_reasoning": severity_reasoning.get('reasoning') if severity_reasoning else None,
                    "physics_corrected": severity_reasoning.get('physics_corrected', False) if severity_reasoning else False
                })
                
                if not consistent:
                    logger.warning(f"{e.event_id}: Speed↓ but severity↑ ({orig:.2f}->{new_sev:.2f})")
                    if severity_reasoning:
                        logger.warning(f"  LLM reasoning: {severity_reasoning.get('reasoning', 'N/A')[:200]}")
                else:
                    logger.info(f"{e.event_id}: Consistent ✓ sev {orig:.2f}->{new_sev:.2f}")
            
            # Clear log for brake test
            if hasattr(self.world_model, 'clear_reasoning_log'):
                self.world_model.clear_reasoning_log()
            
            # Brake test
            if veacon.crash_dynamics.braking_effectiveness in [BrakingEffectiveness.LOW, BrakingEffectiveness.NONE, BrakingEffectiveness.UNKNOWN]:
                orig = veacon.injury_outcome.severity_score
                
                logger.info(f"Brake test {e.event_id}: {veacon.crash_dynamics.braking_effectiveness.value} → high, orig_sev={orig:.2f}")
                
                cf = self.world_model.propagate_intervention(veacon, {"crash_dynamics.braking_effectiveness": "high"})
                new_sev = cf.injury_outcome.severity_score
                
                # Get reasoning log
                reasoning_log = []
                if hasattr(self.world_model, 'get_reasoning_log'):
                    reasoning_log = self.world_model.get_reasoning_log()
                
                severity_reasoning = None
                for r in reasoning_log:
                    if 'severity' in r.get('node', '').lower():
                        severity_reasoning = r
                        break
                
                brake_tests.append({
                    "event_id": e.event_id,
                    "orig_brake": veacon.crash_dynamics.braking_effectiveness.value,
                    "orig_sev": orig, "new_sev": new_sev,
                    "consistent": new_sev <= orig,
                    "llm_reasoning": severity_reasoning.get('reasoning') if severity_reasoning else None,
                    "physics_corrected": severity_reasoning.get('physics_corrected', False) if severity_reasoning else False
                })
        
        self.diag.consistency_details = {"speed": speed_tests, "brake": brake_tests}
        
        return {
            "speed_consistency": np.mean([t["consistent"] for t in speed_tests]) if speed_tests else None,
            "brake_consistency": np.mean([t["consistent"] for t in brake_tests]) if brake_tests else None,
            "n_speed": len(speed_tests),
            "n_brake": len(brake_tests),
            "speed_samples": speed_tests[:5],
            "brake_samples": brake_tests[:5]
        }
    
    def evaluate_safety(self, events: List[SynSHRP2Event], n: int = 30) -> Dict:
        crashes = [e for e in events if "crash" in e.event_type.lower() and "near" not in e.event_type.lower()][:n]
        
        results = {"analyses": [], "top_factors": {}}
        
        for e in tqdm(crashes, desc="Safety"):
            veacon = SynSHRP2DataLoader.to_veacon(e)
            ranking = self.safety_analyzer.rank_safety_critical_features(veacon)
            
            analysis = {"event_id": e.event_id, "orig_sev": ranking.original_severity, "n_interventions": len(ranking.rankings)}
            
            if ranking.rankings:
                top = ranking.rankings[0]
                analysis["top"] = {"node": top.node, "reduction": top.severity_reduction}
                results["top_factors"][top.node] = results["top_factors"].get(top.node, 0) + 1
            
            results["analyses"].append(analysis)
        
        self.diag.safety_details = results["analyses"]
        reductions = [a["top"]["reduction"] for a in results["analyses"] if "top" in a]
        
        return {
            "top_factors": results["top_factors"],
            "avg_reduction": np.mean(reductions) if reductions else 0,
            "n_events": len(crashes),
            "n_with_interventions": len([a for a in results["analyses"] if "top" in a])
        }
    
    def run_full(self, events: List[SynSHRP2Event], diag: DatasetDiagnostics) -> Dict:
        logger.info("=" * 60)
        logger.info("SynSHRP2 EVALUATION")
        logger.info("=" * 60)
        
        # Model mode indicator
        if self.use_vlm:
            logger.info("MODE: Using GPT-4o for VLM and World Model")
        else:
            logger.info("MODE: Rule-based only (no LLM)")
        
        logger.info(f"Events: {diag.total_events} (crashes: {diag.crashes}, near: {diag.near_crashes})")
        logger.info(f"With keyframes: {diag.with_keyframes}, kinematics: {diag.with_kinematics}")
        logger.info(f"With speed: {diag.events_with_speed}, brake: {diag.events_with_brake_data}")
        logger.info(f"Event types: {diag.event_type_distribution}")
        logger.info(f"Severities: {diag.severity_distribution}")
        logger.info(f"Kinematic keys: {diag.kinematic_keys_found}")
        
        results = {"dataset": asdict(diag), "using_vlm": self.use_vlm}
        
        logger.info("\n1. Classification...")
        results["classification"] = self.evaluate_classification(events)
        logger.info(f"   Acc: {results['classification'].get('accuracy', 0):.2%}, F1: {results['classification'].get('macro_f1', 0):.3f}")
        
        logger.info("\n2. Severity...")
        results["severity"] = self.evaluate_severity(events)
        if "mae" in results["severity"]:
            logger.info(f"   MAE: {results['severity']['mae']:.2f}, Within-1: {results['severity']['within_one']:.2%}")
        
        logger.info("\n3. Consistency...")
        results["consistency"] = self.evaluate_consistency(events)
        if results["consistency"]["speed_consistency"]:
            logger.info(f"   Speed: {results['consistency']['speed_consistency']:.2%}")
        
        logger.info("\n4. Safety...")
        results["safety"] = self.evaluate_safety(events)
        logger.info(f"   Top factors: {results['safety']['top_factors']}")
        
        results["diagnostics"] = {
            "classification": self.diag.classification_details[:20],
            "severity": self.diag.severity_details[:20],
            "consistency": {
                "speed": self.diag.consistency_details.get("speed", [])[:10],
                "brake": self.diag.consistency_details.get("brake", [])[:10]
            }
        }
        
        # Add world model reasoning logs if available
        if hasattr(self.world_model, 'get_reasoning_log'):
            reasoning_log = self.world_model.get_reasoning_log()
            results["world_model_reasoning"] = reasoning_log[-20:] if len(reasoning_log) > 20 else reasoning_log
            
            # Summarize physics corrections
            corrections = [r for r in reasoning_log if r.get('physics_corrected')]
            results["physics_corrections"] = {
                "count": len(corrections),
                "samples": corrections[:5]
            }
        
        # Add VLM extraction log if available
        if self.vlm and hasattr(self.vlm, 'get_extraction_log'):
            vlm_log = self.vlm.get_extraction_log()
            results["vlm_extraction_log"] = {
                "total_extractions": len(vlm_log),
                "samples": vlm_log[-10:]  # Last 10 extractions
            }
        
        # Add API call log if available
        if self.client and hasattr(self.client, 'get_call_log'):
            call_log = self.client.get_call_log()
            results["api_call_log"] = {
                "total_calls": len(call_log),
                "samples": call_log[-10:]  # Last 10 calls
            }
        
        logger.info("\n" + "=" * 60)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("=" * 60)
        
        # Classification summary
        if self.diag.classification_details:
            correct = sum(1 for d in self.diag.classification_details if d.get('correct'))
            total = len(self.diag.classification_details)
            logger.info(f"Classification: {correct}/{total} correct ({100*correct/total:.1f}%)")
            
            # Show misclassifications
            misses = [d for d in self.diag.classification_details if not d.get('correct')][:3]
            for m in misses:
                logger.info(f"  MISS: {m['event_id']} - GT={m['gt']}, Pred={m['pred']}, Evidence={m.get('evidence', 'N/A')[:50]}")
        
        # Physics corrections summary
        if hasattr(self.world_model, 'get_reasoning_log'):
            reasoning_log = self.world_model.get_reasoning_log()
            corrections = [r for r in reasoning_log if r.get('physics_corrected')]
            if corrections:
                logger.info(f"Physics corrections: {len(corrections)} violations corrected")
                for c in corrections[:3]:
                    logger.info(f"  FIX: {c['node']} - LLM said {c.get('original_value')}, corrected to {c.get('corrected_value')}")
        
        # Consistency summary
        if self.diag.consistency_details:
            speed_tests = self.diag.consistency_details.get('speed', [])
            if speed_tests:
                inconsistent = [t for t in speed_tests if not t.get('consistent')]
                logger.info(f"Speed consistency: {len(speed_tests) - len(inconsistent)}/{len(speed_tests)} consistent")
                for t in inconsistent[:2]:
                    logger.info(f"  FAIL: {t['event_id']} - speed {t['orig_speed']:.0f}→{t['new_speed']:.0f}, sev {t['orig_sev']:.2f}→{t['new_sev']:.2f}")
                    if t.get('llm_reasoning'):
                        logger.info(f"    Reasoning: {t['llm_reasoning'][:100]}")
        
        logger.info("=" * 60)
        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./synSHRP2")
    parser.add_argument("--output", default="./synSHRP2_results.json")
    parser.add_argument("--log_file", default="./synSHRP2.log")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--no_vlm", action="store_true")
    parser.add_argument("--max_events", type=int)
    parser.add_argument("--crash_ratio", type=float, default=0.5,
                       help="Target ratio of crashes in sample (default 0.5 = 50%% crashes)")
    parser.add_argument("--verbose_diagnostics", type=str, default=None,
                       help="Output file for detailed VLM/LLM reasoning logs (JSON)")
    args = parser.parse_args()
    
    global logger
    logger = setup_logging(args.log_level, args.log_file)
    
    logger.info(f"Started: {datetime.now()}")
    
    if not Path(args.data_dir).exists():
        logger.error(f"Not found: {args.data_dir}")
        return
    
    events, diag = SynSHRP2DataLoader.load_dataset(args.data_dir)
    
    # Sample events with target crash ratio
    if args.max_events and args.max_events < len(events):
        crashes = [e for e in events if "crash" in e.event_type.lower() and "near" not in e.event_type.lower()]
        near_crashes = [e for e in events if "near" in e.event_type.lower()]
        
        n_crashes = int(args.max_events * args.crash_ratio)
        n_near = args.max_events - n_crashes
        
        # Sample from each category
        np.random.seed(42)
        sampled_crashes = list(np.random.choice(crashes, min(n_crashes, len(crashes)), replace=False))
        sampled_near = list(np.random.choice(near_crashes, min(n_near, len(near_crashes)), replace=False))
        
        events = sampled_crashes + sampled_near
        np.random.shuffle(events)
        
        logger.info(f"Sampled {len(events)} events: {len(sampled_crashes)} crashes, {len(sampled_near)} near-crashes")
        
        # Update diagnostics for sampled data
        diag.total_events = len(events)
        diag.crashes = len(sampled_crashes)
        diag.near_crashes = len(sampled_near)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    client = None
    
    if api_key:
        logger.info(f"OPENAI_API_KEY found (length: {len(api_key)})")
        try:
            client = GPT4oClient(api_key)
            logger.info("GPT-4o client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GPT-4o client: {e}")
    else:
        logger.warning("=" * 50)
        logger.warning("NO OPENAI_API_KEY SET!")
        logger.warning("Using RULE-BASED model only (no LLM calls)")
        logger.warning("Set with: export OPENAI_API_KEY='your-key'")
        logger.warning("=" * 50)
    
    evaluator = SynSHRP2Evaluator(client, not args.no_vlm)
    results = evaluator.run_full(events, diag)
    
    # Add API usage stats
    if client:
        results["api_stats"] = {"calls": client.call_count}
        logger.info(f"Total GPT-4o API calls: {client.call_count}")
    else:
        results["api_stats"] = {"calls": 0, "note": "Rule-based only, no LLM"}
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved: {args.output}")
    
    # Save verbose diagnostics if requested
    if args.verbose_diagnostics:
        verbose_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "data_dir": args.data_dir,
                "max_events": args.max_events,
                "crash_ratio": args.crash_ratio,
                "use_vlm": not args.no_vlm
            }
        }
        
        # Full VLM extraction log
        if evaluator.vlm and hasattr(evaluator.vlm, 'get_extraction_log'):
            verbose_data["vlm_extractions"] = evaluator.vlm.get_extraction_log()
        
        # Full world model reasoning log
        if hasattr(evaluator.world_model, 'get_reasoning_log'):
            verbose_data["world_model_reasoning"] = evaluator.world_model.get_reasoning_log()
        
        # Full API call log
        if client and hasattr(client, 'get_call_log'):
            verbose_data["api_calls"] = client.get_call_log()
        
        # Full classification details
        verbose_data["classification_details"] = evaluator.diag.classification_details
        verbose_data["severity_details"] = evaluator.diag.severity_details
        verbose_data["consistency_details"] = evaluator.diag.consistency_details
        
        with open(args.verbose_diagnostics, "w") as f:
            json.dump(verbose_data, f, indent=2, default=str)
        logger.info(f"Verbose diagnostics saved: {args.verbose_diagnostics}")


if __name__ == "__main__":
    main()