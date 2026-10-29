"""
Unit tests for Anatomical Placement Geometry Engine
"""

import pytest
from src.models.placement_geometry import (
    AnatomicalPlacementEngine,
    Point2D,
    PlacementCategory,
    TriageRecommendation,
)


def test_normal_placement():
    engine = AnatomicalPlacementEngine(default_pixel_spacing_mm=0.168)
    carina = Point2D(x=1000.0, y=1000.0)
    # Tip 90 pixels (15.12 mm) below carina
    tip = Point2D(x=1010.0, y=1090.0)

    verdict = engine.evaluate_instance(
        catheter_id="CVC_1",
        tip_coord=tip,
        carina_coord=carina,
    )
    assert verdict.category == PlacementCategory.NORMAL
    assert verdict.delta_y_mm is not None
    assert 0.0 <= verdict.delta_y_mm <= 25.0
    assert "Acceptable" in verdict.anatomical_zone


def test_borderline_placement():
    engine = AnatomicalPlacementEngine(default_pixel_spacing_mm=0.168)
    carina = Point2D(x=1000.0, y=1000.0)
    # Tip 60 pixels (10.08 mm) above carina
    tip = Point2D(x=1005.0, y=940.0)

    verdict = engine.evaluate_instance(
        catheter_id="CVC_1",
        tip_coord=tip,
        carina_coord=carina,
    )
    assert verdict.category == PlacementCategory.BORDERLINE
    assert verdict.delta_y_mm is not None
    assert verdict.delta_y_mm < 0.0


def test_deep_atrial_abnormal_placement():
    engine = AnatomicalPlacementEngine(default_pixel_spacing_mm=0.168)
    carina = Point2D(x=1000.0, y=1000.0)
    # Tip 300 pixels (50.4 mm) below carina (deep right atrium)
    tip = Point2D(x=1010.0, y=1300.0)

    verdict = engine.evaluate_instance(
        catheter_id="CVC_1",
        tip_coord=tip,
        carina_coord=carina,
    )
    assert verdict.category == PlacementCategory.ABNORMAL
    assert "Deep cardiac" in verdict.anatomical_zone


def test_high_subclavian_abnormal_placement():
    engine = AnatomicalPlacementEngine(default_pixel_spacing_mm=0.168)
    carina = Point2D(x=1000.0, y=1000.0)
    # Tip 250 pixels (42.0 mm) above carina
    tip = Point2D(x=900.0, y=750.0)

    verdict = engine.evaluate_instance(
        catheter_id="CVC_1",
        tip_coord=tip,
        carina_coord=carina,
    )
    assert verdict.category == PlacementCategory.ABNORMAL
    assert "High malposition" in verdict.anatomical_zone


def test_multi_cvc_discordant_resolution():
    engine = AnatomicalPlacementEngine(default_pixel_spacing_mm=0.168)
    carina = Point2D(x=1000.0, y=1000.0)

    tips = [
        ("CVC_Normal", Point2D(x=1015.0, y=1080.0), 0.95),      # ~13.4mm below carina (Normal)
        ("CVC_Malpositioned", Point2D(x=1005.0, y=1350.0), 0.92), # ~58.8mm below carina (Deep RA Abnormal)
    ]

    study_verdict = engine.evaluate_study(
        study_uid="test_study_discordant",
        catheter_tips=tips,
        carina_coord=carina,
    )

    assert len(study_verdict.catheters) == 2
    assert study_verdict.catheters[0].category == PlacementCategory.NORMAL
    assert study_verdict.catheters[1].category == PlacementCategory.ABNORMAL
    assert study_verdict.has_abnormal is True
    assert study_verdict.all_normal is False
    assert study_verdict.triage_recommendation == TriageRecommendation.URGENT_REVIEW

