"""Configuration helpers for the tornado geospatial pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved project paths used by the pipeline."""

    project_root: Path
    source_roots: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def inventory_dir(self) -> Path:
        return self.outputs_dir / "inventory"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    def ensure_phase1_dirs(self) -> None:
        for path in [
            self.data_dir / "incoming",
            self.raw_dir,
            self.raw_dir / "source_mirror",
            self.raw_dir / "rasters",
            self.raw_dir / "shapefiles",
            self.data_dir / "processed",
            self.project_root / "notebooks" / "exploratory",
            self.inventory_dir,
            self.outputs_dir / "preprocessed",
            self.outputs_dir / "plots",
            self.outputs_dir / "overlays",
            self.outputs_dir / "masks",
            self.outputs_dir / "models",
            self.outputs_dir / "predictions",
            self.reports_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def build_config(project_root: Path, source_roots: list[str] | None = None) -> ProjectConfig:
    """Create a project config from CLI values."""

    roots = tuple(Path(p).expanduser().resolve() for p in (source_roots or []))
    return ProjectConfig(project_root=project_root.resolve(), source_roots=roots)
