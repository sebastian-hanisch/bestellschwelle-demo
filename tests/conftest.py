"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

Mini-Instanz (Handbeispiel): Nachfrage 0, 1, 2 mit ¼, ½, ¼; h = 1, p = 4, K = 2, Lieferzeit 0, Rückstand; Positionen -3 bis 4.
Handrechnung (Bruchrechnung, siehe README und Tab „Von Hand“): L(y) = 4, 5/4, 1, 2, 3 für y = 0, 1, 2, 3, 4 und 8 bei y = -1;
C₁ = 3, 3, 3, 3, 5/4, 1, 2, 3 für x = -3 ... 4; W₂ = 7, 61/16, 21/8, 53/16, 5 für y = 0 ... 4; C₂ = 37/8 (x ≤ 0), 61/16, 21/8, 53/16, 5;
g(0, 2) = 2, g(1, 2) = 5/2.
"""

import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bsw_dp as D  # noqa: E402

HAND_K = 2.0
HAND_SPEC = D.ModelSpec(L=0, h=1.0, p=4.0, Q=7, zmin=-3, ymax=4, pmf=(0.25, 0.5, 0.25))


@pytest.fixture(scope="session")
def hand_model():
    return D.build_model(HAND_SPEC)


@pytest.fixture(scope="session")
def hand_W1():
    """W für noch eine Periode (= L(y)) bei y = -3 ... 4, von Hand: 4·(1 - y) für y <= 0, danach 5/4, 1, 2, 3."""
    return [F(16), F(12), F(8), F(4), F(5, 4), F(1), F(2), F(3)]
