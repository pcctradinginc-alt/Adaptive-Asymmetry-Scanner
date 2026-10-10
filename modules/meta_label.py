"""modules/meta_label.py – Learning Engine: Meta-Labeling bestehender Champion-Signale (SHADOW bis Promotion).

Frage (Meta-Label nach López de Prado): Lohnt sich ein vom Champion bereits erzeugtes Richtungssignal
(Kandidat nach Prescreening/Deep Analysis) nach Kosten, oder sollte es NICHT gehandelt werden?
Das Modell lernt zuerst Abstinenz, nicht Richtung.

Kreislauf (geschlossen, nachprüfbar):
  Prognose/Kandidat (Candidate Ledger, Features eingefroren) → Outcome (feedback.py)
  → evidence_ledger.labeled(as_of): nur Labels mit label_available_at <= Trainingszeit
  → Training: Baseline L2-Logit gegen Challenger HGB (feste Hyperparameter, kein Tuning),
    Purged/Embargo-Walk-Forward (evidence_core), Vergleich mit Base Rate und Impact×Surprise
  → versioniertes Artefakt + Manifest (welche Labels, Hash, Cutoff, neue Labels seit Vorversion)
  → Shadow-Prognosen (nur für Kandidaten NACH dem Trainingszeitpunkt, unveränderlich)
  → Produktionswirkung ausschließlich über den registrierten Vertrag PROM-META-001
    (Adapter-Merkmal meta_p_take, PromotionController, Forward-Evidenz, höchstens ABSTENTION_ONLY automatisch)
  → Rollback auf eine Vorversion (registry history), Integritätsprüfung per sha256. Ein Rollback HÄLT den Zeiger:
    neue Versionen werden weiter trainiert und protokolliert, aber erst nach `release` aktiv.
  → Neutraining höchstens wöchentlich und erst ab RETRAIN_MIN_NEW_LABELS neuen Labels; Modell-Artefakte nur der
    letzten KEEP_ARTIFACT_VERSIONS Versionen (+ aktive) bleiben, ältere Versionen behalten ihr Manifest (Hash, Cutoff).

Die Spezifikation (SPEC) ist eingefroren: Änderungen ergeben einen neuen spec_hash und eine neue Modellfamilie.
Ohne genügend unabhängige Labels gibt es kein Modell (NEED_MORE_DATA). Es wird nie etwas aufgefüllt.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modules import evidence_core as ec
from modules import evidence_ledger as el
from modules.atomic_io import append_jsonl, atomic_write_json, read_jsonl

log = logging.getLogger(__name__)

ROOT = Path("outputs/models/meta_label")
RETRAIN_MIN_NEW_LABELS = 25      # Neutraining erst ab so vielen neuen Labels seit der letzten Version
RETRAIN_MIN_DAYS = 7             # höchstens eine neue Version je Woche (Versionen und Repo-Größe begrenzt)
KEEP_ARTIFACT_VERSIONS = 6       # Artefakte der letzten N Versionen (+ aktive) bleiben rollback-fähig
FEATURES = ("impact", "surprise", "rel_volume", "news_count", "log_market_cap", "sentiment_confidence",
            "short_pct_float", "options_skew_ratio", "options_skew_25d", "dealer_oi_ratio", "has_near_earnings",
            "insider_count", "ml_exp_dd_60", "ml_disagreement_sd", "mismatch", "sigma_30d", "quick_mc_hit_rate",
            "final_mc_hit_rate", "z_score_2d_scaled",
            # richtungsabhängig (mit der Signalrichtung multipliziert)
            "scan_day_ret", "sentiment_score", "sentiment_drift", "short_mom", "ml_exp_ret_60", "ml_q10_ret_60",
            "price_move_48h_signed")
SIGNED = frozenset({"scan_day_ret", "sentiment_score", "sentiment_drift", "short_mom", "ml_exp_ret_60",
                    "ml_q10_ret_60", "price_move_48h_signed"})
SPEC = {
    "name": "meta_label_take_v1",
    "population": "CANDIDATE", "horizon": 20, "horizon_unit": "calendar",
    "label": "Richtungsrendite des Basiswerts nach 20 Kalendertagen minus 20 bp > 0",
    "features": list(FEATURES), "signed_features": sorted(SIGNED),
    "models": {"logit_l2": {"C": 1.0}, "hgb": {"max_iter": 150, "max_depth": 3, "learning_rate": 0.05,
                                               "l2_regularization": 1.0, "min_samples_leaf": 20}},
    "cv": {"n_folds": 4, "embargo_days": 5, "min_train": 100, "min_test": 30},
    "readiness": {"min_labels": 200, "min_dates": 20},
    "decision": {"threshold": 0.5, "alpha": 0.05, "active_rule": "hgb nur wenn BETTER und Brier < logit, sonst logit"},
    "seed": 13,
}


def spec_hash(spec: dict | None = None) -> str:
    return hashlib.sha256(json.dumps(spec or SPEC, sort_keys=True).encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _vec(features: dict, direction: int) -> list[float]:
    out = []
    for f in FEATURES:
        v = (features or {}).get(f)
        if isinstance(v, bool):
            v = float(v)
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = float("nan")
        if not np.isfinite(x):
            x = float("nan")
        out.append(x * direction if f in SIGNED and np.isfinite(x) else x)
    return out


def dataset(recs: list[dict]) -> dict:
    rows = [r for r in recs if r.get("direction") in (1, -1) and r.get("outcome_net") is not None]
    rows.sort(key=lambda r: (r["decision_time"] or "", r["prediction_id"]))
    X = np.array([_vec(r.get("features") or {}, r["direction"]) for r in rows], dtype=float).reshape(len(rows), len(FEATURES))
    return {"X": X, "y": np.array([1 if r["outcome_net"] > 0 else 0 for r in rows], dtype=int),
            "net": np.array([float(r["outcome_net"]) for r in rows]), "dates": [r["decision_date"] for r in rows],
            "label_ends": [r["label_available_at"] for r in rows], "ids": [r["prediction_id"] for r in rows],
            "score_baseline": np.array([((r.get("features") or {}).get("impact") or 0) *
                                        ((r.get("features") or {}).get("surprise") or 0) for r in rows], dtype=float)}


def _model(name: str, spec: dict):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p = spec["models"][name]
    if name == "logit_l2":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                             StandardScaler(), LogisticRegression(C=p["C"], max_iter=2000))
    return HistGradientBoostingClassifier(random_state=spec["seed"], **p)


class _Masked:
    """Modell + Spaltenmaske: Merkmale ohne einen einzigen endlichen Wert im Trainingsfenster werden ausgeblendet
    (nie mit 0 oder Konstanten gefüllt). Die Maske ist Teil des versionierten Artefakts."""

    def __init__(self, model, mask):
        self.model, self.mask = model, mask

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=float)[:, self.mask])


def _fit(name: str, X, y, spec: dict):
    if len(set(y.tolist())) < 2:
        return None
    mask = np.isfinite(X).any(axis=0)
    if not mask.any():
        return None
    return _Masked(_model(name, spec).fit(X[:, mask], y), mask)


def _metrics(p: np.ndarray, y: np.ndarray, base: np.ndarray, net: np.ndarray, dates: list, thr: float,
             alpha_eff: float) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score
    eps = 1e-6
    out = {"n": int(len(y)), "brier": round(float(np.mean((p - y) ** 2)), 5),
           "brier_base_rate": round(float(np.mean((base - y) ** 2)), 5),
           "log_loss": round(float(log_loss(y, np.clip(p, eps, 1 - eps), labels=[0, 1])), 5),
           "auc": round(float(roc_auc_score(y, p)), 4) if len(set(y.tolist())) == 2 else None}
    keep = p >= thr
    out["kept_share"] = round(float(keep.mean()), 4)
    if keep.any() and (~keep).any():
        # Wert der Abstinenz je Beobachtung: kept -> Rendite, abstiniert -> 0; minus "alle handeln"
        delta = np.where(keep, net, 0.0) - net
        out["abstention_value_mean"] = round(float(delta.mean()), 6)
        lo, hi = ec.group_bootstrap_ci(delta.tolist(), dates, alpha=alpha_eff, two_sided=False)
        out["abstention_value_ci"] = [lo, hi]
    else:
        out["abstention_value_mean"], out["abstention_value_ci"] = None, [None, None]
    better = (out["brier"] < out["brier_base_rate"] and out["abstention_value_ci"][0] is not None
              and out["abstention_value_ci"][0] > 0)
    out["verdict"] = "BETTER_THAN_BASELINE" if better else "NOT_BETTER"
    return out


def evaluate(ds: dict, spec: dict | None = None) -> dict:
    """Out-of-sample-Vergleich (Purged/Embargo-Walk-Forward). Bonferroni über alle Challenger."""
    spec = spec or SPEC
    cv = spec["cv"]
    folds = ec.purged_walk_forward(ds["dates"], ds["label_ends"], n_folds=cv["n_folds"],
                                   embargo_days=cv["embargo_days"], min_train=cv["min_train"], min_test=cv["min_test"])
    names = list(spec["models"])
    alpha_eff = ec.bonferroni_alpha(spec["decision"]["alpha"], len(names))
    if not folds:
        return {"status": "NEED_MORE_DATA", "reason": "keine gültigen Purged-Folds", "folds": 0}
    oos = {n: [] for n in names}
    idx_all, base_all = [], []
    for f in folds:
        tr, te = f["train_idx"], f["test_idx"]
        idx_all += te
        base_all += [float(ds["y"][tr].mean())] * len(te)
        for n in names:
            m = _fit(n, ds["X"][tr], ds["y"][tr], spec)
            oos[n] += (m.predict_proba(ds["X"][te])[:, 1].tolist() if m is not None
                       else [float(ds["y"][tr].mean())] * len(te))
    idx = np.array(idx_all)
    y, net, base = ds["y"][idx], ds["net"][idx], np.array(base_all)
    dates = [ds["dates"][i] for i in idx_all]
    res = {"status": "OK", "folds": len(folds), "n_oos": int(len(idx)), "alpha_effective": alpha_eff,
           "models": {n: _metrics(np.array(oos[n]), y, base, net, dates, spec["decision"]["threshold"], alpha_eff)
                      for n in names}}
    from sklearn.metrics import roc_auc_score
    sb = ds["score_baseline"][idx]
    res["baseline_impact_x_surprise_auc"] = (round(float(roc_auc_score(y, sb)), 4)
                                             if len(set(y.tolist())) == 2 and len(set(sb.tolist())) > 1 else None)
    res["take_all_mean_net"] = round(float(net.mean()), 6)
    return res


# ── Registry (versionierte Artefakte, aktiver Zeiger, Rollback) ─────────────
def _registry(root: Path) -> dict:
    p = root / "registry.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"spec_hash": spec_hash(), "versions": [],
                                                                         "active": None, "history": []}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(*, as_of=None, recs: list[dict] | None = None, root: Path | None = None, now: datetime | None = None,
          spec: dict | None = None) -> dict:
    """Ein Lernzyklus. Gibt den Status zurück (NEED_MORE_DATA | UNCHANGED | TRAINED)."""
    spec, root, now = spec or SPEC, Path(root or ROOT), now or _now()
    as_of = as_of or now.date()
    recs = recs if recs is not None else el.records([spec["population"]])
    lab = el.labeled(recs, as_of=as_of, population=spec["population"], horizon=spec["horizon"])
    ds = dataset(lab)
    n, n_dates = len(ds["ids"]), len(set(ds["dates"]))
    status = {"generated": now.isoformat(timespec="seconds"), "as_of": str(as_of), "spec_hash": spec_hash(spec),
              "n_labels": n, "n_label_dates": n_dates, "required": spec["readiness"]}
    reg = _registry(root)
    if n < spec["readiness"]["min_labels"] or n_dates < spec["readiness"]["min_dates"]:
        status.update(status="NEED_MORE_DATA", active=reg.get("active"))
        atomic_write_json(root / "status.json", status, indent=1)
        return status
    ids_hash = hashlib.sha256("\n".join(sorted(ds["ids"])).encode()).hexdigest()
    # Vergleich mit der LETZTEN trainierten Version (nicht der aktiven): ein Rollback wird so nicht durch ein
    # Neutraining auf denselben Labels rückgängig gemacht.
    prev = reg["versions"][-1] if reg["versions"] else None
    prev_ids: set = set()
    if prev:
        mp = root / "versions" / prev["version"] / "manifest.json"
        prev_ids = set(json.loads(mp.read_text()).get("train_ids") or []) if mp.exists() else set()
    new_n = len(set(ds["ids"]) - prev_ids)
    if prev and prev["spec_hash"] == spec_hash(spec):
        age = (now - datetime.fromisoformat(prev["trained_at"])).days
        why = ("keine neuen Labels" if prev["train_ids_hash"] == ids_hash else
               f"{new_n} neue Labels < {RETRAIN_MIN_NEW_LABELS}" if new_n < RETRAIN_MIN_NEW_LABELS else
               f"letzte Version {age} T alt < {RETRAIN_MIN_DAYS}" if age < RETRAIN_MIN_DAYS else None)
        if why:
            status.update(status="UNCHANGED", active=reg.get("active"), reason=why, new_labels_since_previous=new_n)
            atomic_write_json(root / "status.json", status, indent=1)
            return status
    ev = evaluate(ds, spec)
    version = f"{spec['name']}-{now:%Y%m%dT%H%M%S}"
    vdir = root / "versions" / version
    vdir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in spec["models"]:
        m = _fit(name, ds["X"], ds["y"], spec)
        if m is None:
            continue
        p = vdir / f"{name}.pkl"
        p.write_bytes(pickle.dumps(m))
        artifacts[name] = {"path": str(p.relative_to(root)), "sha256": _sha(p)}
    ms = (ev.get("models") or {})
    lg, hg = ms.get("logit_l2") or {}, ms.get("hgb") or {}
    active_model = ("hgb" if hg.get("verdict") == "BETTER_THAN_BASELINE" and hg.get("brier") is not None
                    and lg.get("brier") is not None and hg["brier"] < lg["brier"] and "hgb" in artifacts
                    else "logit_l2")
    import sklearn
    from modules.promotion_controller import code_version
    manifest = {"version": version, "spec_hash": spec_hash(spec), "spec": spec, "trained_at": now.isoformat(timespec="seconds"),
                "as_of": str(as_of), "label_cutoff": max(ds["label_ends"]), "n_train": n, "n_train_dates": n_dates,
                "train_ids": sorted(ds["ids"]), "train_ids_hash": ids_hash,
                "new_labels_since_previous": new_n, "previous_version": prev and prev["version"],
                "evaluation": ev, "artifacts": artifacts, "active_model": active_model,
                "code_version": code_version(), "sklearn": sklearn.__version__}
    atomic_write_json(vdir / "manifest.json", manifest, indent=1)
    reg["versions"].append({k: manifest[k] for k in ("version", "spec_hash", "trained_at", "label_cutoff", "n_train",
                                                     "train_ids_hash", "new_labels_since_previous", "active_model")}
                           | {"artifacts": artifacts})
    held = reg.get("hold")
    reg["history"].append({"at": manifest["trained_at"], "action": "TRAINED_HELD" if held else "TRAINED_AND_ACTIVATED",
                           "version": version, "previous": reg.get("active")})
    if not held:                                     # nach einem Rollback bleibt der Zeiger bis `release`
        reg["active"], reg["spec_hash"] = version, spec_hash(spec)
    _prune(root, reg)
    atomic_write_json(root / "registry.json", reg, indent=1)
    status.update(status="TRAINED", version=version, active_model=active_model, activated=not held,
                  active=reg["active"], new_labels_since_previous=new_n, evaluation=ev)
    atomic_write_json(root / "status.json", status, indent=1)
    return status


def _prune(root: Path, reg: dict) -> None:
    """Artefakte älterer Versionen entfernen (die letzten KEEP_ARTIFACT_VERSIONS und die aktive bleiben).
    Manifest bleibt (Hash der Trainings-IDs, Cutoff, Evaluation); die ID-Liste ist aus Ledger + as_of rekonstruierbar."""
    keep = {v["version"] for v in reg["versions"][-KEEP_ARTIFACT_VERSIONS:]} | {reg.get("active")}
    for v in reg["versions"]:
        if v["version"] in keep or v.get("artifacts_pruned"):
            continue
        for a in (v.get("artifacts") or {}).values():
            (root / a["path"]).unlink(missing_ok=True)
        mp = root / "versions" / v["version"] / "manifest.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            m.pop("train_ids", None)
            atomic_write_json(mp, {**m, "artifacts_pruned": True}, indent=1)
        v["artifacts_pruned"] = True


def rollback(*, root: Path | None = None, to_version: str | None = None, reason: str = "",
             now: datetime | None = None) -> dict:
    """Aktiven Zeiger auf eine frühere Version setzen (Default: die vorherige) und HALTEN: neue Versionen werden
    trainiert, aber erst nach release() aktiv. Ziel muss rollback-fähige Artefakte haben."""
    root, now = Path(root or ROOT), now or _now()
    reg = _registry(root)
    by = {v["version"]: v for v in reg["versions"]}
    vs = list(by)
    cur = reg.get("active")
    target = to_version or (vs[vs.index(cur) - 1] if cur in vs and vs.index(cur) > 0 else None)
    if target not in by or by[target].get("artifacts_pruned"):
        raise ValueError(f"keine rollback-fähige Zielversion (aktiv {cur}, Ziel {target})")
    at = now.isoformat(timespec="seconds")
    reg["history"].append({"at": at, "action": "ROLLBACK", "version": target, "previous": cur, "reason": reason})
    reg["active"], reg["hold"] = target, {"at": at, "version": target, "reason": reason}
    atomic_write_json(root / "registry.json", reg, indent=1)
    return reg


def release(*, root: Path | None = None, reason: str = "", now: datetime | None = None) -> dict:
    """Halt nach Rollback aufheben und die neueste Version aktivieren (menschliche Entscheidung)."""
    root, now = Path(root or ROOT), now or _now()
    reg = _registry(root)
    if not reg["versions"]:
        raise ValueError("keine Version vorhanden")
    latest = reg["versions"][-1]["version"]
    reg["history"].append({"at": now.isoformat(timespec="seconds"), "action": "RELEASE", "version": latest,
                           "previous": reg.get("active"), "reason": reason})
    reg["active"] = latest
    reg.pop("hold", None)
    atomic_write_json(root / "registry.json", reg, indent=1)
    return reg


_CACHE: dict = {}


def active_model(root: Path | None = None):
    """(Versions-Metadaten, Modell) der aktiven Version; Integritätsprüfung per sha256 (Abweichung -> None)."""
    root = Path(root or ROOT)
    reg = _registry(root)
    v = next((x for x in reg["versions"] if x["version"] == reg.get("active")), None)
    if not v:
        return None, None
    art = (v.get("artifacts") or {}).get(v.get("active_model"))
    if not art:
        return None, None
    p = root / art["path"]
    key = (str(p), art["sha256"])
    if key in _CACHE:
        return v, _CACHE[key]
    if not p.exists() or _sha(p) != art["sha256"]:
        log.error(f"meta_label: Artefakt {p} fehlt oder Hash weicht ab – kein Modell (fail-safe)")
        return None, None
    m = pickle.loads(p.read_bytes())
    _CACHE[key] = m
    return v, m


def predict_one(features: dict, direction, *, as_of=None, root: Path | None = None) -> dict | None:
    """P(lohnend) für ein Signal. PIT: nur ein Modell, das VOR as_of trainiert wurde. Sonst None (nie geraten)."""
    try:
        d = el._dir(direction) if isinstance(direction, str) else direction
        if d not in (1, -1):
            return None
        v, m = active_model(root)
        if m is None:
            return None
        t = as_of or _now()
        if isinstance(t, str):
            t = datetime.fromisoformat(t)
        if datetime.fromisoformat(v["trained_at"]) >= (t if t.tzinfo else t.replace(tzinfo=timezone.utc)):
            return None
        p = float(m.predict_proba(np.array([_vec(features, d)], dtype=float))[:, 1][0])
        return {"p": round(p, 5), "version": v["version"], "model": v["active_model"]}
    except Exception as e:  # noqa: BLE001 – Prognose ist optional, nie entscheidungsblockierend
        log.warning(f"meta_label.predict_one: {type(e).__name__}: {e}")
        return None


def record_predictions(*, recs: list[dict] | None = None, root: Path | None = None, now: datetime | None = None) -> int:
    """Unveränderliche Shadow-Prognosen für Kandidaten, die NACH dem Training der aktiven Version entschieden wurden
    (Features eingefroren, Modell älter als die Entscheidung). Je (prediction_id, version) genau einmal."""
    root, now = Path(root or ROOT), now or _now()
    v, m = active_model(root)
    if m is None:
        return 0
    recs = recs if recs is not None else el.records([SPEC["population"]])
    trained = datetime.fromisoformat(v["trained_at"])
    path = root / "predictions" / f"{now:%Y-%m}.jsonl"
    seen = {(r["prediction_id"], r["version"]) for f in sorted((root / "predictions").glob("*.jsonl"))
            for r in read_jsonl(f)}                  # über alle Monatsdateien: je (Prognose, Version) genau einmal
    new = []
    for r in recs:
        if r.get("horizon") != SPEC["horizon"] or not r.get("decision_time"):
            continue
        dt = datetime.fromisoformat(str(r["decision_time"]))
        dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        if dt <= trained or (r["prediction_id"], v["version"]) in seen:
            continue
        pr = predict_one(r.get("features") or {}, r.get("direction"), as_of=dt, root=root)
        if pr:
            new.append({"prediction_id": r["prediction_id"], "entity": r.get("entity"), "version": v["version"],
                        "model": pr["model"], "p_take": pr["p"], "decision_time": r["decision_time"],
                        "recorded_at": now.isoformat(timespec="seconds")})
    append_jsonl(path, new, sort_keys=True)
    return len(new)


def status(root: Path | None = None) -> dict:
    p = Path(root or ROOT) / "status.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "NEVER_RUN"}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Meta-Label-Lernzyklus (SHADOW; Produktion nur über PROM-META-001)")
    ap.add_argument("cmd", choices=["train", "status", "rollback", "release"])
    ap.add_argument("--to", default=None)
    ap.add_argument("--reason", default="manuell")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        st = train()
        n = record_predictions() if st.get("status") in ("TRAINED", "UNCHANGED") else 0
        log.info(f"meta_label: {st.get('status')} (Labels {st.get('n_labels')}, Tage {st.get('n_label_dates')}), "
                 f"{n} neue Shadow-Prognosen")
        print(json.dumps({k: v for k, v in st.items() if k != "evaluation"}, ensure_ascii=False))
    elif a.cmd == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=1))
    elif a.cmd == "rollback":
        reg = rollback(to_version=a.to, reason=a.reason)
        print(json.dumps({"active": reg["active"], "hold": reg.get("hold")}, ensure_ascii=False))
    else:
        reg = release(reason=a.reason)
        print(json.dumps({"active": reg["active"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(main())
