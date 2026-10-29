"""
Anatomical Placement Geometry Engine
====================================
Translates spatial coordinates of detected catheter distal tips and the
tracheobronchial carina into interpretable, instance-specific clinical
placement decisions (Normal, Borderline, Abnormal/Malpositioned).

Unlike ungrounded global CNN classifiers that fail on multi-CVC cases,
this engine grounds each catheter in physical anatomy.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PlacementCategory(str, Enum):
    NORMAL = "CVC - Normal"
    BORDERLINE = "CVC - Borderline"
    ABNORMAL = "CVC - Abnormal"
    UNCERTAIN = "Uncertain / Indeterminate"


class TriageRecommendation(str, Enum):
    ACCEPTABLE = "Predicted Acceptable Placement"
    STANDARD_REVIEW = "Standard Clinical Verification"
    URGENT_REVIEW = "Placement Needs Urgent Review"
    NO_ASSESSMENT = "No CVC Assessed"


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def distance_to(self, other: Point2D) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass
class CatheterInstancePlacement:
    catheter_id: str
    tip_coord: Point2D
    carina_coord: Optional[Point2D]
    delta_y_px: Optional[float]
    delta_x_px: Optional[float]
    delta_y_mm: Optional[float]
    category: PlacementCategory
    anatomical_zone: str
    confidence: float
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class StudyPlacementVerdict:
    study_uid: str
    catheters: List[CatheterInstancePlacement]
    has_abnormal: bool
    has_borderline: bool
    all_normal: bool
    triage_recommendation: TriageRecommendation
    summary_text: str


class AnatomicalPlacementEngine:
    """
    Evaluates catheter placement relative to the tracheobronchial carina.

    Standard Clinical Anatomy Rules:
    - Normal: Lower third of SVC or at the Cavoatrial Junction (CAJ),
      defined as 0 mm to +25 mm below the carina (or roughly 0 to +0.05 normalized height).
    - Borderline: Mid-to-high SVC, defined as -25 mm to 0 mm above the carina.
    - Abnormal:
        * Deep Right Atrium / Ventricle: > +40 mm below carina (arrhythmia risk).
        * High / Subclavian / Internal Jugular: < -35 mm above carina.
        * Extravascular / Contralateral: lateral deviation exceeding anatomically expected SVC track.
    """

    def __init__(
        self,
        default_pixel_spacing_mm: float = 0.168,  # Typical adult PA radiograph ~0.14-0.17 mm/px
        normal_range_mm: Tuple[float, float] = (0.0, 25.0),
        borderline_range_mm: Tuple[float, float] = (-25.0, 0.0),
        deep_atrial_threshold_mm: float = 40.0,
        subclavian_threshold_mm: float = -35.0,
        max_lateral_dev_mm: float = 45.0,
    ):
        self.default_pixel_spacing_mm = default_pixel_spacing_mm
        self.normal_range_mm = normal_range_mm
        self.borderline_range_mm = borderline_range_mm
        self.deep_atrial_threshold_mm = deep_atrial_threshold_mm
        self.subclavian_threshold_mm = subclavian_threshold_mm
        self.max_lateral_dev_mm = max_lateral_dev_mm

    def evaluate_instance(
        self,
        catheter_id: str,
        tip_coord: Point2D,
        carina_coord: Optional[Point2D],
        pixel_spacing_mm: Optional[float] = None,
        tip_confidence: float = 1.0,
    ) -> CatheterInstancePlacement:
        spacing = pixel_spacing_mm or self.default_pixel_spacing_mm

        if carina_coord is None:
            return CatheterInstancePlacement(
                catheter_id=catheter_id,
                tip_coord=tip_coord,
                carina_coord=None,
                delta_y_px=None,
                delta_x_px=None,
                delta_y_mm=None,
                category=PlacementCategory.UNCERTAIN,
                anatomical_zone="Carina landmark missing or unidentifiable",
                confidence=tip_confidence * 0.5,
            )

        # In image coordinates: Y increases downwards.
        # delta_y > 0 means tip is below carina (towards heart / atrium).
        # delta_y < 0 means tip is above carina (towards neck / clavicle).
        delta_y_px = tip_coord.y - carina_coord.y
        delta_x_px = tip_coord.x - carina_coord.x
        delta_y_mm = delta_y_px * spacing
        delta_x_mm = delta_x_px * spacing

        # Placement logic
        if delta_y_mm > self.deep_atrial_threshold_mm:
            category = PlacementCategory.ABNORMAL
            zone = f"Deep cardiac placement: {delta_y_mm:.1f}mm below carina (Right Atrium / Ventricle risk)"
        elif delta_y_mm < self.subclavian_threshold_mm:
            category = PlacementCategory.ABNORMAL
            zone = f"High malposition: {abs(delta_y_mm):.1f}mm above carina (Subclavian / Jugular)"
        elif abs(delta_x_mm) > self.max_lateral_dev_mm and delta_y_mm < 10.0:
            category = PlacementCategory.ABNORMAL
            zone = f"Lateral malposition: {abs(delta_x_mm):.1f}mm lateral offset from carina (Azygos / Contralateral)"
        elif self.normal_range_mm[0] <= delta_y_mm <= self.normal_range_mm[1]:
            category = PlacementCategory.NORMAL
            zone = f"Acceptable lower-SVC / Cavoatrial Junction zone: {delta_y_mm:.1f}mm below carina"
        elif self.borderline_range_mm[0] <= delta_y_mm < self.normal_range_mm[0]:
            category = PlacementCategory.BORDERLINE
            zone = f"Mid-to-high SVC: {abs(delta_y_mm):.1f}mm above carina"
        else:
            # Transition zones (e.g. 25-40mm below carina)
            category = PlacementCategory.BORDERLINE
            zone = f"Low cavoatrial border: {delta_y_mm:.1f}mm below carina"

        return CatheterInstancePlacement(
            catheter_id=catheter_id,
            tip_coord=tip_coord,
            carina_coord=carina_coord,
            delta_y_px=delta_y_px,
            delta_x_px=delta_x_px,
            delta_y_mm=delta_y_mm,
            category=category,
            anatomical_zone=zone,
            confidence=tip_confidence,
            metadata={"delta_x_mm": delta_x_mm, "pixel_spacing": spacing},
        )

    def evaluate_study(
        self,
        study_uid: str,
        catheter_tips: List[Tuple[str, Point2D, float]],  # List of (catheter_id, tip_coord, confidence)
        carina_coord: Optional[Point2D],
        pixel_spacing_mm: Optional[float] = None,
    ) -> StudyPlacementVerdict:
        if not catheter_tips:
            return StudyPlacementVerdict(
                study_uid=study_uid,
                catheters=[],
                has_abnormal=False,
                has_borderline=False,
                all_normal=False,
                triage_recommendation=TriageRecommendation.NO_ASSESSMENT,
                summary_text="No catheter instances detected in radiograph.",
            )

        evaluated_catheters: List[CatheterInstancePlacement] = []
        for cat_id, tip, conf in catheter_tips:
            res = self.evaluate_instance(
                catheter_id=cat_id,
                tip_coord=tip,
                carina_coord=carina_coord,
                pixel_spacing_mm=pixel_spacing_mm,
                tip_confidence=conf,
            )
            evaluated_catheters.append(res)

        has_abnormal = any(c.category == PlacementCategory.ABNORMAL for c in evaluated_catheters)
        has_borderline = any(c.category == PlacementCategory.BORDERLINE for c in evaluated_catheters)
        all_normal = all(c.category == PlacementCategory.NORMAL for c in evaluated_catheters)

        if has_abnormal:
            triage = TriageRecommendation.URGENT_REVIEW
            summary = f"URGENT: {sum(1 for c in evaluated_catheters if c.category == PlacementCategory.ABNORMAL)} catheter(s) malpositioned."
        elif has_borderline:
            triage = TriageRecommendation.STANDARD_REVIEW
            summary = f"REVIEW: {sum(1 for c in evaluated_catheters if c.category == PlacementCategory.BORDERLINE)} catheter(s) in borderline position."
        elif all_normal:
            triage = TriageRecommendation.ACCEPTABLE
            summary = f"ACCEPTABLE: All {len(evaluated_catheters)} catheter(s) placed within acceptable cavoatrial target zone."
        else:
            triage = TriageRecommendation.STANDARD_REVIEW
            summary = "INDETERMINATE: Anatomical landmarks could not be unambiguously confirmed."

        return StudyPlacementVerdict(
            study_uid=study_uid,
            catheters=evaluated_catheters,
            has_abnormal=has_abnormal,
            has_borderline=has_borderline,
            all_normal=all_normal,
            triage_recommendation=triage,
            summary_text=summary,
        )

