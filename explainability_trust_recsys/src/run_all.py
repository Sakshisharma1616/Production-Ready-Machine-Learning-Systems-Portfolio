import json
from src.config import DATA_DIR, REPORTS_DIR, TOP_K
from src.download_data import download_and_extract
from src.make_dataset import load_raw, build_lightfm_dataset
from src.train import train_lightfm, save_model
from src.recommend import recommend_for_user
from src.explain import explain_recommendation, format_feature_explanation
from src.trust_metrics import item_popularity, summarize_trust_metrics
from src.counterfactual import counterfactual_explanation

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ml1m_dir = download_and_extract(DATA_DIR)
    ratings, movies, users = load_raw(ml1m_dir)

    dataset, interactions, weights, user_features, item_features, meta = build_lightfm_dataset(
        ratings, movies, users
    )

    model, metrics = train_lightfm(interactions, user_features, item_features)
    save_model(model, metrics)

    # Sample users for reports
    sample_user_internal_ids = list(range(min(25, interactions.shape[0])))

    inv_item_id_map = meta["inv_item_id_map"]        # internal item -> external movie_id
    movies_df = meta["movies_df"]                    # indexed by external movie_id
    inv_item_feature_map = meta["inv_item_feature_map"]

    all_recs = []
    feature_explanations = []
    counterfactuals = []

    for u in sample_user_internal_ids:
        # Standard LightFM recommendations
        recs, _ = recommend_for_user(model, u, interactions, user_features, item_features, k=TOP_K)
        recs = list(map(int, recs))
        all_recs.append(recs)

        # Feature attribution for top-1
        top_item_internal = recs[0]
        top_movie_id = int(inv_item_id_map[top_item_internal])

        feat_expl = explain_recommendation(
            model=model,
            user_internal_id=u,
            item_internal_id=top_item_internal,
            user_features=user_features,
            item_features=item_features,
            inv_item_feature_map=inv_item_feature_map,
            top_n_features=5,
        )

        feature_explanations.append({
            "user_internal_id": int(u),
            "top_item_internal_id": int(top_item_internal),
            "top_movie_id": int(top_movie_id),
            "text": format_feature_explanation(feat_expl, int(top_movie_id), movies_df),
            "raw": feat_expl,
        })

        # Counterfactual explanation (post-hoc)
        cf = counterfactual_explanation(
            model=model,
            interactions=interactions,
            user_internal_id=u,
            k=TOP_K,
            item_features=item_features,
        )

        # Add readable titles for JSON
        def to_movie(i_internal: int) -> int:
            return int(inv_item_id_map[int(i_internal)])

        cf_aug = dict(cf)
        cf_aug["original_topk_movie_ids"] = [to_movie(i) for i in cf["original_topk_internal"]]
        cf_aug["counterfactual_topk_movie_ids"] = [to_movie(i) for i in cf["counterfactual_topk_internal"]]

        removed = cf.get("counterfactual_removed_history_item_internal", None)
        if removed is not None:
            cf_aug["removed_history_movie_id"] = to_movie(int(removed))
            cf_aug["removed_history_title"] = str(movies_df.loc[cf_aug["removed_history_movie_id"], "title"]) \
                if cf_aug["removed_history_movie_id"] in movies_df.index else str(cf_aug["removed_history_movie_id"])
        else:
            cf_aug["removed_history_movie_id"] = None
            cf_aug["removed_history_title"] = None

        cf_aug["top1_title"] = str(movies_df.loc[to_movie(cf["top1_internal"]), "title"]) \
            if to_movie(cf["top1_internal"]) in movies_df.index else str(to_movie(cf["top1_internal"]))

        cf_aug["counterfactual_top1_title"] = str(movies_df.loc[to_movie(cf["counterfactual_top1_internal"]), "title"]) \
            if to_movie(cf["counterfactual_top1_internal"]) in movies_df.index else str(to_movie(cf["counterfactual_top1_internal"]))

        counterfactuals.append(cf_aug)

    # Trust metrics
    popularity = item_popularity(interactions)
    trust = summarize_trust_metrics(all_recs, popularity, inv_item_id_map, movies_df)

    # Save reports
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS_DIR / "trust_report.json").write_text(json.dumps(trust, indent=2))
    (REPORTS_DIR / "feature_explanations.json").write_text(json.dumps(feature_explanations, indent=2))
    (REPORTS_DIR / "counterfactual_explanations.json").write_text(json.dumps(counterfactuals, indent=2))

    # Human-readable summary
    lines = []
    lines.append("=== Model Metrics ===")
    for k, v in metrics.items():
        lines.append(f"{k}: {v}")

    lines.append("\n=== Trust Report ===")
    for k, v in trust.items():
        lines.append(f"{k}: {v}")

    lines.append("\n=== Feature Explanations (Top-1 per user) ===")
    for ex in feature_explanations[:10]:
        lines.append("\n---")
        lines.append(ex["text"])

    lines.append("\n=== Counterfactual Explanations (Top-1 per user) ===")
    for cf in counterfactuals[:10]:
        lines.append("\n---")
        lines.append(f"Original Top-1: {cf['top1_title']}")
        if cf["removed_history_title"] is not None:
            lines.append(f"Because you liked: {cf['removed_history_title']} (sim={cf.get('influence_similarity', 0.0):.3f})")
            lines.append(f"If you hadn't liked it, Top-1 becomes: {cf['counterfactual_top1_title']}")
        else:
            lines.append("No history available for counterfactual.")

    (REPORTS_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("[DONE] Reports saved in:", REPORTS_DIR)

if __name__ == "__main__":
    main()
