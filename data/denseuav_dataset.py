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

import glob
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
        data_root:        Path to the DenseUAV root directory.
        gps_file:         Filename of the GPS annotation file relative to
                          data_root (e.g. "Dense_GPS_train.txt").
        drone_altitude:   Altitude tag to select images, one of "H80", "H90",
                          "H100", or None (use all available altitudes).
                          NOTE: GPS is only available for H80; H90/H100 images
                          will inherit GPS from the H80 entry of the same location.
        transform_uav:    Transform applied to the UAV (PIL) image independently.
        transform_sat:    Transform applied to the satellite (PIL) image independently.
        paired_transform: If provided, called as ``paired_transform(uav_pil, sat_pil)``
                          returning ``(uav_tensor, sat_tensor)``.  Takes priority over
                          transform_uav/transform_sat when set.  Use this to apply
                          geometrically consistent augmentation (shared flips/rotation)
                          so HomographyNet receives a valid gradient signal.
    """

    # Satellite image is always .tif; UAV image is always .JPG
    _DRONE_EXT = ".JPG"
    _SAT_EXT   = ".tif"

    # Only H80 has GPS in the annotation files (confirmed on actual dataset)
    _GPS_ALTITUDE = "H80"

    def __init__(
        self,
        data_root:        str,
        gps_file:         str = "Dense_GPS_train.txt",
        drone_altitude:   Optional[str] = "H80",
        transform_uav:    Optional[Callable] = None,
        transform_sat:    Optional[Callable] = None,
        paired_transform: Optional[Callable] = None,
    ) -> None:
        self.data_root        = os.path.abspath(data_root)
        self.drone_altitude   = drone_altitude
        self.transform_uav    = transform_uav
        self.transform_sat    = transform_sat
        self.paired_transform = paired_transform

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

        if self.paired_transform is not None:
            # Geometrically consistent: shared random flip/rotation
            uav_img, sat_img = self.paired_transform(uav_pil, sat_pil)
        else:
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
# Test/Evaluation datasets
# ---------------------------------------------------------------------------

class DenseUAVQuery(Dataset):
    """Query set: drone images from test/query_drone/{loc_id}/H80.JPG.

    Implements the DenseUAV test query set (777 items).

    Returns per item:
        "uav_img" : FloatTensor (3, img_size, img_size)
        "label"   : LongTensor  ()   — int(loc_id); consistent with DenseUAVGallery
        "uav_gps" : FloatTensor (2,) — [lon, lat]; zeros tensor if GPS unavailable

    Args:
        data_root:      Path to DenseUAV root directory.
        gps_file:       GPS annotation file relative to data_root.  GPS entries
                        are satellite-side; query GPS is inferred from the same
                        location's satellite entry.  If absent or unmatched,
                        has_gps=False and uav_gps tensors are zeros.
        drone_altitude: Altitude tag (default "H80").
        transform_uav:  Transform applied to the UAV PIL image.

    Attributes:
        has_gps (bool): True if at least one sample has a valid GPS coordinate.
    """

    _DRONE_EXT    = ".JPG"
    _GPS_ALTITUDE = "H80"

    def __init__(
        self,
        data_root:      str,
        gps_file:       str                = "Dense_GPS_test.txt",
        drone_altitude: str                = "H80",
        transform_uav:  Optional[Callable] = None,
    ) -> None:
        self.data_root      = os.path.abspath(data_root)
        self.drone_altitude = drone_altitude
        self.transform_uav  = transform_uav

        gps_path = os.path.join(self.data_root, gps_file)
        self._gps: Dict[str, Tuple[float, float, float]] = (
            _parse_gps_file(gps_path) if os.path.isfile(gps_path) else {}
        )

        self.samples: List[Tuple[str, int, Optional[Tuple[float, float]]]]
        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No query drone images found under "
                f"{os.path.join(self.data_root, 'test', 'query_drone')}. "
                "Verify that data_root points to the DenseUAV root directory "
                "and that the test/query_drone/ sub-tree is present."
            )

        self.has_gps: bool = any(s[2] is not None for s in self.samples)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_gps(self, loc_id: str) -> Optional[Tuple[float, float]]:
        """Look up GPS for a test query location via its satellite entry.

        Tries multiple GPS-file key formats to be robust against dataset
        variants that use "test/satellite/" vs "test/gallery_satellite/".
        """
        sat_ext = DenseUAVPairs._SAT_EXT
        for prefix in ("test/satellite", "test/gallery_satellite"):
            key   = f"{prefix}/{loc_id}/{self._GPS_ALTITUDE}{sat_ext}"
            entry = self._gps.get(key)
            if entry is not None:
                lon, lat, _ = entry
                return (lon, lat)
        return None

    def _build_samples(
        self,
    ) -> List[Tuple[str, int, Optional[Tuple[float, float]]]]:
        """Discover query images under test/query_drone/ and build sample list."""
        query_base = os.path.join(self.data_root, "test", "query_drone")
        if not os.path.isdir(query_base):
            raise FileNotFoundError(
                f"Test query_drone directory not found: {query_base}"
            )

        loc_ids = sorted(
            d for d in os.listdir(query_base)
            if os.path.isdir(os.path.join(query_base, d))
        )
        samples = []

        for loc_id in loc_ids:
            drone_dir = os.path.join(query_base, loc_id)
            img_path  = os.path.join(
                drone_dir, f"{self.drone_altitude}{self._DRONE_EXT}"
            )
            if not os.path.isfile(img_path):
                # Fallback: first sorted JPG in the folder
                candidates = sorted(
                    glob.glob(os.path.join(drone_dir, f"*{self._DRONE_EXT}"))
                )
                if not candidates:
                    continue
                img_path = candidates[0]

            try:
                label = int(loc_id)
            except ValueError:
                label = len(samples)   # non-numeric fallback: sequential index

            gps_xy = self._resolve_gps(loc_id)
            samples.append((img_path, label, gps_xy))

        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label, gps_xy = self.samples[idx]

        uav_pil = Image.open(img_path).convert("RGB")
        if self.transform_uav is not None:
            uav_img = self.transform_uav(uav_pil)
        else:
            from torchvision.transforms.functional import to_tensor
            uav_img = to_tensor(uav_pil)

        gps = (
            torch.tensor(gps_xy, dtype=torch.float32)
            if gps_xy is not None
            else torch.zeros(2, dtype=torch.float32)
        )

        return {
            "uav_img": uav_img,
            "label":   torch.tensor(label, dtype=torch.long),
            "uav_gps": gps,
        }

    def __repr__(self) -> str:
        return (
            f"DenseUAVQuery("
            f"n_queries={len(self)}, "
            f"altitude={self.drone_altitude!r}, "
            f"has_gps={self.has_gps})"
        )


class DenseUAVGallery(Dataset):
    """Gallery set: satellite images from test/gallery_satellite/{loc_id}/H80.tif.

    Implements the DenseUAV test gallery set (3033 items).

    Returns per item:
        "sat_img" : FloatTensor (3, img_size, img_size)
        "label"   : LongTensor  ()   — int(loc_id); consistent with DenseUAVQuery
        "sat_gps" : FloatTensor (2,) — [lon, lat]; zeros tensor if GPS unavailable

    Args:
        data_root:      Path to DenseUAV root directory.
        gps_file:       GPS annotation file relative to data_root.
        drone_altitude: Altitude tag used to select the .tif file (default "H80").
        transform_sat:  Transform applied to the satellite PIL image.

    Attributes:
        has_gps (bool): True if at least one sample has a valid GPS coordinate.
    """

    _SAT_EXT      = ".tif"
    _GPS_ALTITUDE = "H80"

    def __init__(
        self,
        data_root:      str,
        gps_file:       str                = "Dense_GPS_test.txt",
        drone_altitude: str                = "H80",
        transform_sat:  Optional[Callable] = None,
    ) -> None:
        self.data_root      = os.path.abspath(data_root)
        self.drone_altitude = drone_altitude
        self.transform_sat  = transform_sat

        gps_path = os.path.join(self.data_root, gps_file)
        self._gps: Dict[str, Tuple[float, float, float]] = (
            _parse_gps_file(gps_path) if os.path.isfile(gps_path) else {}
        )

        self.samples: List[Tuple[str, int, Optional[Tuple[float, float]]]]
        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No gallery satellite images found under "
                f"{os.path.join(self.data_root, 'test', 'gallery_satellite')}. "
                "Verify that data_root points to the DenseUAV root directory "
                "and that the test/gallery_satellite/ sub-tree is present."
            )

        self.has_gps: bool = any(s[2] is not None for s in self.samples)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_gps(self, loc_id: str) -> Optional[Tuple[float, float]]:
        """Look up GPS for a gallery satellite location.

        Tries multiple path prefixes and filename stems (H80 / H80_old) to be
        robust against GPS-file variants.
        """
        for prefix in ("test/satellite", "test/gallery_satellite"):
            for stem in (self._GPS_ALTITUDE, f"{self._GPS_ALTITUDE}_old"):
                key   = f"{prefix}/{loc_id}/{stem}{self._SAT_EXT}"
                entry = self._gps.get(key)
                if entry is not None:
                    lon, lat, _ = entry
                    return (lon, lat)
        return None

    def _build_samples(
        self,
    ) -> List[Tuple[str, int, Optional[Tuple[float, float]]]]:
        """Discover gallery images under test/gallery_satellite/ and build sample list."""
        gallery_base = os.path.join(self.data_root, "test", "gallery_satellite")
        if not os.path.isdir(gallery_base):
            raise FileNotFoundError(
                f"Test gallery_satellite directory not found: {gallery_base}"
            )

        loc_ids = sorted(
            d for d in os.listdir(gallery_base)
            if os.path.isdir(os.path.join(gallery_base, d))
        )
        samples = []

        for loc_id in loc_ids:
            sat_dir  = os.path.join(gallery_base, loc_id)
            img_path = os.path.join(
                sat_dir, f"{self.drone_altitude}{self._SAT_EXT}"
            )
            if not os.path.isfile(img_path):
                # Fallback: first sorted .tif in the folder
                candidates = sorted(
                    glob.glob(os.path.join(sat_dir, f"*{self._SAT_EXT}"))
                )
                if not candidates:
                    continue
                img_path = candidates[0]

            try:
                label = int(loc_id)
            except ValueError:
                label = len(samples)

            gps_xy = self._resolve_gps(loc_id)
            samples.append((img_path, label, gps_xy))

        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label, gps_xy = self.samples[idx]

        sat_pil = Image.open(img_path).convert("RGB")
        if self.transform_sat is not None:
            sat_img = self.transform_sat(sat_pil)
        else:
            from torchvision.transforms.functional import to_tensor
            sat_img = to_tensor(sat_pil)

        gps = (
            torch.tensor(gps_xy, dtype=torch.float32)
            if gps_xy is not None
            else torch.zeros(2, dtype=torch.float32)
        )

        return {
            "sat_img": sat_img,
            "label":   torch.tensor(label, dtype=torch.long),
            "sat_gps": gps,
        }

    def __repr__(self) -> str:
        return (
            f"DenseUAVGallery("
            f"n_gallery={len(self)}, "
            f"altitude={self.drone_altitude!r}, "
            f"has_gps={self.has_gps})"
        )
