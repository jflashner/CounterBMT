# scripts/test_extraction.py

from pathlib import Path

# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

from counter_bmt.scenarionet_visualizer import prepare_for_vlm
from counter_bmt.vlm_extractor import (
    VLMSafetyCriticalExtractor, 
    MockGPT4oClient,
    GPT4oClient,
    TimestampedImage
)

# Use absolute paths
DATA_DIR = SRC_DIR / "exp_converted"
OUTPUT_DIR = SRC_DIR / "outputs" / "screenshots"

# Step 1: Generate screenshots
print("Loading scenario and generating screenshots...")
saved_images, trajectory, scenario_id = prepare_for_vlm(
    data_dir=str(DATA_DIR),
    scenario_index=0,
    output_dir=str(OUTPUT_DIR),
    num_frames=8
)
print(f"Generated {len(saved_images)} images for scenario {scenario_id}")
print(f"Trajectory shape: {trajectory.shape}")

# Step 2: Create image objects
images = [TimestampedImage(path=path, timestamp=ts) for path, ts in saved_images]

# Step 3: Choose client
USE_REAL_API = True  # Set to True to use GPT-4o

if USE_REAL_API:
    print("\nUsing real GPT-4o API...")
    client = GPT4oClient()
else:
    print("\nUsing mock client (no API calls)...")
    client = MockGPT4oClient()

# Step 4: Extract
extractor = VLMSafetyCriticalExtractor(client, debug=True)
features = extractor.extract(images=images, scenario_id=scenario_id, trajectory=trajectory)

# Step 5: Print results
print("\n" + "="*60)
print("EXTRACTION RESULTS")
print("="*60)
print(features.summary())

# Step 6: Save results
import json
output_file = OUTPUT_DIR / "extraction_result.json"
with open(output_file, "w") as f:
    json.dump(features.to_dict(), f, indent=2)
print(f"\nSaved to {output_file}")