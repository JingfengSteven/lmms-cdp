"""Metadata manager for storing and retrieving per-sample metadata during evaluation.

This module provides a global registry for storing metadata associated with
specific doc_ids during model inference, which can then be retrieved in
process_results for logging.

Example usage:
    # In model's generate_until method:
    from lmms_eval.metadata_manager import metadata_manager

    metadata_manager.set_metadata(doc_id, {
        "frame_0": {"path": "video1_frame0.jpg", "resolution": "224x224", "patches": 256},
        "frame_1": {"path": "video1_frame1.jpg", "resolution": "448x224", "patches": 512},
    })

    # In task's process_results function:
    from lmms_eval.metadata_manager import metadata_manager

    def longvideobench_process_results(doc, results):
        # ... existing logic ...
        metadata = metadata_manager.get_metadata(doc["id"])
        data_dict["metadata"] = metadata
        return {"lvb_acc": data_dict}
"""

from threading import Lock
from typing import Any, Dict, Optional


class MetadataManager:
    """Thread-safe manager for storing and retrieving per-sample metadata.

    This manager uses a dictionary to store metadata keyed by doc_id, with
    thread-safe operations to support distributed evaluation.
    """

    def __init__(self):
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def set_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Store metadata for a specific doc_id.

        Args:
            doc_id: The document identifier
            metadata: Dictionary containing the metadata to store
        """
        with self._lock:
            self._storage[doc_id] = metadata

    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific doc_id.

        Args:
            doc_id: The document identifier

        Returns:
            The metadata dictionary if found, None otherwise
        """
        with self._lock:
            return self._storage.get(doc_id)

    def update_metadata(self, doc_id: str, key: str, value: Any) -> None:
        """Update a specific key in the metadata for a doc_id.

        Args:
            doc_id: The document identifier
            key: The metadata key to update
            value: The value to set
        """
        with self._lock:
            if doc_id not in self._storage:
                self._storage[doc_id] = {}
            self._storage[doc_id][key] = value

    def add_frame_metadata(
        self, doc_id: str, frame_idx: int, path: str, resolution: str, patches: int
    ) -> None:
        """Convenience method to add frame-level metadata.

        Args:
            doc_id: The document identifier
            frame_idx: The frame index
            path: The frame path or basename
            resolution: The resolution as "WxH" string
            patches: The number of patches
        """
        with self._lock:
            if doc_id not in self._storage:
                self._storage[doc_id] = {}
            if "frames" not in self._storage[doc_id]:
                self._storage[doc_id]["frames"] = []

            self._storage[doc_id]["frames"].append({
                "frame_idx": frame_idx,
                "path": path,
                "resolution": resolution,
                "patches": patches,
            })

    def clear(self) -> None:
        """Clear all stored metadata."""
        with self._lock:
            self._storage.clear()

    def has_metadata(self, doc_id: str) -> bool:
        """Check if metadata exists for a doc_id.

        Args:
            doc_id: The document identifier

        Returns:
            True if metadata exists, False otherwise
        """
        with self._lock:
            return doc_id in self._storage


# Global singleton instance
metadata_manager = MetadataManager()
