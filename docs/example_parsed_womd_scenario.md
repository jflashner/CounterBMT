# Example Parsed WOMD Scenario Object

This is a compact reference for what a parsed WOMD scenario looks like in this repo after it has been loaded through the ScenarioNet/MetaDrive path.

Source path used for this example:
- `data/scenarionet_waymo_training_500/sd_waymo_v1.2_4245da4b159fa62c.pkl`

Loader path in the codebase:
- `src/Adv-BMT/bmt/dataset/dataset.py`
- `metadrive/metadrive/scenario/scenario_description.py`

## What gets loaded

The dataset loader reads a ScenarioNet `.pkl` and returns a `ScenarioDescription`, which is a dict-like object with these top-level keys:

```python
[
    "id",
    "version",
    "length",
    "tracks",
    "dynamic_map_states",
    "map_features",
    "metadata",
]
```

For the concrete example below:

```python
{
    "type": "ScenarioDescription",
    "id": "4245da4b159fa62c",
    "length": 91,
    "num_tracks": 35,
    "num_map_features": 170,
    "num_dynamic_map_states": 6,
}
```

## Real parsed object shape

This is a compact pretty-printed version of the actual parsed object structure:

```python
scenario = {
    "id": "4245da4b159fa62c",
    "version": "...",
    "length": 91,
    "tracks": {
        "0": {
            "type": "VEHICLE",
            "state": {
                "position": np.ndarray(shape=(91, 3)),
                "length": np.ndarray(shape=(91,)),
                "width": np.ndarray(shape=(91,)),
                "height": np.ndarray(shape=(91,)),
                "heading": np.ndarray(shape=(91,)),
                "velocity": np.ndarray(shape=(91, 2)),
                "valid": np.ndarray(shape=(91,)),
            },
            "metadata": {
                "track_length": 91,
                "type": "VEHICLE",
                "object_id": "0",
                "dataset": "waymo",
            },
        },
        "...": "...",
    },
    "dynamic_map_states": {
        "128": {
            "type": "TRAFFIC_LIGHT",
            "state": {
                "object_state": np.ndarray(shape=(91,)),
            },
            "lane": "...",
            "stop_point": "...",
            "metadata": {
                "track_length": 91,
                "type": "TRAFFIC_LIGHT",
                "object_id": "128",
                "dataset": "waymo",
            },
        },
        "...": "...",
    },
    "map_features": {
        "2": {
            "type": "ROAD_EDGE_BOUNDARY",
            "polyline": np.ndarray(shape=(99, 3)),
        },
        "...": "...",
    },
    "metadata": {
        "id": "...",
        "coordinate": "...",
        "ts": np.ndarray(shape=(91,)),
        "metadrive_processed": True,
        "sdc_id": "...",
        "dataset": "waymo",
        "scenario_id": "4245da4b159fa62c",
        "source_file": "...",
        "track_length": 91,
        "current_time_index": "...",
        "sdc_track_index": "...",
        "objects_of_interest": [...],
        "tracks_to_predict": {...},
        "object_summary": {...},
        "number_summary": {...},
    },
}
```

## Concrete example pieces

### Metadata keys

```python
[
    "id",
    "coordinate",
    "ts",
    "metadrive_processed",
    "sdc_id",
    "dataset",
    "scenario_id",
    "source_file",
    "track_length",
    "current_time_index",
    "sdc_track_index",
    "objects_of_interest",
    "tracks_to_predict",
    "object_summary",
    "number_summary",
]
```

### First track

The first track in this scenario is track `"0"`:

```python
scenario["tracks"]["0"] = {
    "type": "VEHICLE",
    "state": {
        "position": np.ndarray(shape=(91, 3)),
        "length": np.ndarray(shape=(91,)),
        "width": np.ndarray(shape=(91,)),
        "height": np.ndarray(shape=(91,)),
        "heading": np.ndarray(shape=(91,)),
        "velocity": np.ndarray(shape=(91, 2)),
        "valid": np.ndarray(shape=(91,)),
    },
    "metadata": {
        "track_length": 91,
        "type": "VEHICLE",
        "object_id": "0",
        "dataset": "waymo",
    },
}
```

A real preview from that track:

```python
{
    "type": "VEHICLE",
    "metadata": {
        "type": "VEHICLE",
        "object_id": "0",
        "track_length": 91,
        "dataset": "waymo",
    },
    "state_preview": {
        "position_first2": [
            [7269.59423828125, 12748.716796875, 154.98045349121094],
            [7269.59423828125, 12748.716796875, 154.9809112548828],
        ],
        "heading_first5": [
            1.571661114692688,
            1.571661114692688,
            1.571661114692688,
            1.571661114692688,
            1.571661114692688,
        ],
        "velocity_first2": [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "valid_first10": [
            True, True, True, True, True,
            True, True, True, True, True,
        ],
    },
}
```

### First map feature

The first map feature in this scenario is feature `"2"`:

```python
scenario["map_features"]["2"] = {
    "type": "ROAD_EDGE_BOUNDARY",
    "polyline": np.ndarray(shape=(99, 3)),
}
```

And its first few polyline points were:

```python
{
    "type": "ROAD_EDGE_BOUNDARY",
    "polyline": {
        "shape": [99, 3],
        "head": [
            [7147.3515625, 12707.09375, 153.05690002441406],
            [7147.85009765625, 12707.095703125, 153.05841064453125],
            [7148.3486328125, 12707.0986328125, 153.0599365234375],
        ],
    },
}
```

### First dynamic map state

This scenario also has traffic light state:

```python
scenario["dynamic_map_states"]["128"] = {
    "type": "TRAFFIC_LIGHT",
    "state": {
        "object_state": np.ndarray(shape=(91,)),
    },
    "lane": "...",
    "stop_point": "...",
    "metadata": {
        "track_length": 91,
        "type": "TRAFFIC_LIGHT",
        "object_id": "128",
        "dataset": "waymo",
    },
}
```

## Mental model

The easiest way to think about the parsed WOMD scenario object in this codebase is:

- `metadata`: scene-level information and prediction metadata
- `tracks`: per-agent trajectories and box states over time
- `dynamic_map_states`: time-varying infrastructure state, mainly traffic lights
- `map_features`: static roadway geometry

This is the object that later gets preprocessed into the tensorized scene representation used by Adv-BMT.
