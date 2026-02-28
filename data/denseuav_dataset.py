"""
data/denseuav_dataset.py — DenseUAV training pairs dataset.

Dataset layout (data_root = path to the DenseUAV folder):
  data_root/
    Dense_GPS_train.txt          # satellite GPS for train split
    Dense_GPS_test.txt           # satellite GPS for test split
    train/
      drone/{loc_id}/H80.JPG    # UAV image at altitude H80
      drone/{loc_id}/H90.JPG
      drone/{loc_id}/H100.JPG
      satellite/{loc_id}/H80.tif
      satellite/{loc_id}/H80_old.tif
      satellite/{loc_id}/H90.tif
      ...
    test/
      query_drone/{loc_id}/H80.JPG
      gallery_satellite/{loc_id}/H80.tif
      ...

GPS file format (one entry per line):
  <rel_path> E<lon> N<lat> <alt>
  e.g.  train/satellite/000000/H80.tif E120.38756 N30.32413 94.606

IMPORTANT notes discovered from the real dataset:
  - GPS entries only exist for altitude H80 (H80.tif + H80_old.tif).
    H90 and H100 have NO GPS lines in the GPS files.
  - GPS entries are for satellite images only; drone GPS is inferred from the
    matching satellite at the same location (same lon/lat).
  - Training: 2256 location classes.
  - Test query: 777 drone folders; gallery: 3033 satellite folders.

__getitem__ returns a dict with:
  uav_img  : FloatTensor (3, 512, 512)  — transformed UAV image
  sat_img  : FloatTensor (3, 512, 512)  — transformed satellite image
  label    : LongTensor  ()             — integer class label (0-indexed)
  uav_gps  : FloatTensor (2,)           — [lon, lat] for UAV location
  sat_gps  : FloatTensor (2,)           — [lon, lat] for satellite location
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# GPS parsing
# ---------------------------------------------------------------------------

def _parse_gps_file(gps_path: str) -> Dict[str, Tuple[float, float, float]]:
    """Parse a DenseUAV GPS text file into a lookup dict.

    GPS file format (one entry per line):
        <rel_path> E<lon> N<lat> <alt>

    Args:
        gps_path: Absolute path to the GPS file.

    Returns:
        Dict mapping normalised relative path (forward-slash) to
        (lon, lat, alt) floats.

    TODO: If a future dataset version includes drone GPS entries, this
          parser will pick them up automatically (path contains "drone").
    """
    gps: Dict[str, Tuple[float, float, float]] = {}
    with open(gps_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"{gps_path}:{lineno}: expected 4 tokens, got {len(parts)}: {line!r}"
                )
            rel_path = parts[0].replace("\\", "/")   # normalise to forward-slash
            try:
                lon = float(parts[1].lstrip("E"))
                lat = float(parts[2].lstrip("N"))
                alt = float(parts[3])
            except ValueError as exc:
                raise ValueError(
                    f"{gps_path}:{lineno}: cannot parse GPS values: {line!r}"
                ) from exc
            gps[rel_path] = (lon, lat, alt)
    return gps


# ---------------------------------------------------------------------------
# Training pairs dataset
# ---------------------------------------------------------------------------

class DenseUAVPairs(Dataset):
    """Paired (UAV image, satellite image) dataset for training.

    Each item is one (UAV, satellite) pair sharing the same location label.
    The label is a contiguous integer in [0, num_classes).

    Args:
        data_root:      Path to the DenseUAV root directory.
        gps_file:       Filename of the GPS annotation file relative to
                        data_root (e.g. "Dense_GPS_train.txt").
        drone_altitude: Altitude tag to select images, one of "H80", "H90",
                        "H100", or None (use all available altitudes).
                        NOTE: GPS is only available for H80; H90/H100 images
                        will inherit GPS from the H80 entry of the same location.
        transform_uav:  Transform applied to the UAV (PIL) image.
        transform_sat:  Transform applied to the satellite (PIL) image.
    """

    # Satellite image is always .tif; UAV image is always .JPG
    _DRONE_EXT = ".JPG"
    _SAT_EXT   = ".tif"

    # Only H80 has GPS in the annotation files (confirmed on actual dataset)
    _GPS_ALTITUDE = "H80"

    def __init__(
        self,
        data_root:      str,
        gps_file:       str = "Dense_GPS_train.txt",
        drone_altitude: Optional[str] = "H80",
        transform_uav:  Optional[Callable] = None,
        transform_sat:  Optional[Callable] = None,
    ) -> None:
        self.data_root      = os.path.abspath(data_root)
        self.drone_altitude = drone_altitude
        self.transform_uav  = transform_uav
        self.transform_sat  = transform_sat

        # --- parse GPS file ---
        gps_path = os.path.join(self.data_root, gps_file)
        if not os.path.isfile(gps_path):
            raise FileNotFoundError(f"GPS file not found: {gps_path}")
        self._gps = _parse_gps_file(gps_path)

        # --- build sample list ---
        self.samples: List[Tuple[str, str, int, Tuple[float,float], Tuple[float,float]]]
        self.samples = self._build_pairs()

        assert len(self.samples) > 0, (
            f"No valid (drone, satellite) pairs found under {self.data_root}. "
            "Check data_root and drone_altitude settings."
        )

        # Expose num_classes for downstream use (e.g. classifier head)
        self.num_classes: int = len({s[2] for s in self.samples})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_gps_for_location(
        self, loc_id: str, split_prefix: str = "train"
    ) -> Optional[Tuple[float, float]]:
        """Look up (lon, lat) for a location from the GPS dict.

        Only H80 entries exist in the GPS file.  Returns None if not found.

        Args:
            loc_id:       Zero-padded location folder name, e.g. "000042".
            split_prefix: "train" or "test".
        """
        # GPS key always uses the H80 satellite entry
        key = f"{split_prefix}/satellite/{loc_id}/{self._GPS_ALTITUDE}{self._SAT_EXT}"
        entry = self._gps.get(key)
        if entry is None:
            return None
        lon, lat, _ = entry
        return (lon, lat)

    def _altitudes_for_location(self, drone_dir: str) -> List[str]:
        """Return altitude tags present in a drone directory.

        Scans for files matching *<tag>.JPG* (no underscore suffix).

        Args:
            drone_dir: Absolute path to the drone location folder.

        Returns:
            Sorted list of altitude tags, e.g. ["H80", "H90", "H100"].
        """
        tags = []
        for fname in os.listdir(drone_dir):
            if fname.endswith(self._DRONE_EXT) and "_" not in fname:
                tags.append(os.path.splitext(fname)[0])   # e.g. "H80"
        return sorted(tags)

    def _build_pairs(
        self,
    ) -> List[Tuple[str, str, int, Tuple[float,float], Tuple[float,float]]]:
        """Build the list of all valid (drone_path, sat_path, label, gps_d, gps_s).

        Label assignment:
            Locations are sorted lexicographically by their folder name.
            Label = index in that sorted list (0-indexed, contiguous).

        Returns:
            List of tuples:
              (drone_abs_path, sat_abs_path, label_int,
               (drone_lon, drone_lat), (sat_lon, sat_lat))
        """
        drone_base = os.path.join(self.data_root, "train", "drone")
        sat_base   = os.path.join(self.data_root, "train", "satellite")

        if not os.path.isdir(drone_base):
            raise FileNotFoundError(f"Drone train directory not found: {drone_base}")
        if not os.path.isdir(sat_base):
            raise FileNotFoundError(f"Satellite train directory not found: {sat_base}")

        loc_ids = sorted(os.listdir(drone_base))   # ["000000", "000001", ...]
        samples = []
        skipped = 0

        for label_idx, loc_id in enumerate(loc_ids):
            drone_dir = os.path.join(drone_base, loc_id)
            sat_dir   = os.path.join(sat_base,   loc_id)

            if not os.path.isdir(drone_dir) or not os.path.isdir(sat_dir):
                skipped += 1
                continue

            # Decide which altitudes to iterate
            if self.drone_altitude is not None:
                altitudes = [self.drone_altitude]
            else:
                # TODO: if all-altitude mode is used for augmentation,
                #       consider weighting by altitude to balance the dataset.
                altitudes = self._altitudes_for_location(drone_dir)

            # GPS for this location (satellite GPS, reused for drone)
            gps_xy = self._get_gps_for_location(loc_id, split_prefix="train")
            if gps_xy is None:
                # TODO: if future GPS files cover H90/H100, update lookup.
                # Fallback: GPS unavailable → skip this location entirely.
                skipped += 1
                continue

            for alt in altitudes:
                drone_path = os.path.join(drone_dir, f"{alt}{self._DRONE_EXT}")
                sat_path   = os.path.join(sat_dir,   f"{alt}{self._SAT_EXT}")

                if not os.path.isfile(drone_path):
                    # TODO: warn if expected altitude is explicitly requested
                    continue
                if not os.path.isfile(sat_path):
                    continue

                # Both drone and satellite share the same location GPS.
                # Drone GPS: inherited from satellite (same lat/lon by definition).
                # TODO: if drone-specific GPS becomes available in the annotation
                #       file, parse it here and use it instead.
                samples.append((
                    drone_path,
                    sat_path,
                    label_idx,    # contiguous 0-indexed label
                    gps_xy,       # (lon, lat) for drone
                    gps_xy,       # (lon, lat) for satellite
                ))

        if skipped:
            # Non-fatal: some locations may lack one image or GPS
            print(f"[DenseUAVPairs] Skipped {skipped} locations "
                  f"(missing image or GPS). {len(samples)} pairs loaded.")

        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one (UAV, satellite) training pair.

        Returns:
            dict with keys:
              "uav_img" : FloatTensor (3, img_size, img_size)
              "sat_img" : FloatTensor (3, img_size, img_size)
              "label"   : LongTensor  ()       — scalar class index
              "uav_gps" : FloatTensor (2,)     — [lon, lat]
              "sat_gps" : FloatTensor (2,)     — [lon, lat]
        """
        drone_path, sat_path, label, gps_d, gps_s = self.samples[idx]

        # Load images as RGB PIL; .tif files are handled by Pillow natively
        uav_pil = Image.open(drone_path).convert("RGB")
        sat_pil = Image.open(sat_path).convert("RGB")

        if self.transform_uav is not None:
            uav_img = self.transform_uav(uav_pil)   # (3, H, W)
        else:
            from torchvision.transforms.functional import to_tensor
            uav_img = to_tensor(uav_pil)

        if self.transform_sat is not None:
            sat_img = self.transform_sat(sat_pil)   # (3, H, W)
        else:
            from torchvision.transforms.functional import to_tensor
            sat_img = to_tensor(sat_pil)

        # Sanity: shapes must be (3, img_size, img_size)
        assert uav_img.ndim == 3 and uav_img.shape[0] == 3, (
            f"Unexpected UAV image shape {uav_img.shape} at index {idx}"
        )
        assert sat_img.ndim == 3 and sat_img.shape[0] == 3, (
            f"Unexpected satellite image shape {sat_img.shape} at index {idx}"
        )

        return {
            "uav_img": uav_img,                                         # (3,512,512)
            "sat_img": sat_img,                                         # (3,512,512)
            "label":   torch.tensor(label, dtype=torch.long),          # ()
            "uav_gps": torch.tensor(gps_d, dtype=torch.float32),       # (2,)
            "sat_gps": torch.tensor(gps_s, dtype=torch.float32),       # (2,)
        }

    def __repr__(self) -> str:
        return (
            f"DenseUAVPairs("
            f"n_pairs={len(self)}, "
            f"num_classes={self.num_classes}, "
            f"altitude={self.drone_altitude!r})"
        )


# ---------------------------------------------------------------------------
# TODO: Test/Evaluation datasets (implement when engine/evaluator.py is added)
# ---------------------------------------------------------------------------

# class DenseUAVQuery(Dataset):
#     """Query set: drone images from test/query_drone/{loc_id}/H80.JPG.
#     Returns: {"uav_img": (3,512,512), "label": (), "uav_gps": (2,)}
#     777 query items (test split).
#     """
#     ...
#
# class DenseUAVGallery(Dataset):
#     """Gallery set: satellite images from test/gallery_satellite/{loc_id}/H80.tif.
#     Returns: {"sat_img": (3,512,512), "label": (), "sat_gps": (2,)}
#     3033 gallery items (test split).
#     """
#     ...
