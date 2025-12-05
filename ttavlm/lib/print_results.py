from prettytable import PrettyTable


def print_results(
    results: dict,
) -> None:
    t = PrettyTable(["dataset", "corruption", "acc", "auroc", "fpr95"])

    avg_acc, avg_auc, avg_fpr = 0, 0, 0

    for dataset, res in results.items():
        for c, v in res.items():
            if "[0]" in c:
                continue

            if "overall" in c:
                avg_acc += v["acc"].avg
                avg_auc += v["auc"].avg
                avg_fpr += v["fpr"].avg

            t.add_row(
                [
                    dataset,
                    c,
                    f"{v['acc'].avg:.2%} ± {100 * v['acc'].std:.2f}",
                    f"{v['auc'].avg:.2%} ± {100 * v['auc'].std:.2f}",
                    f"{v['fpr'].avg:.2%} ± {100 * v['fpr'].std:.2f}",
                ]
            )

    t.add_row(
        [
            "Average",
            " ",
            f"{avg_acc / len(results.keys()):.2%} ± 0.00",
            f"{avg_auc / len(results.keys()):.2%} ± 0.00",
            f"{avg_fpr / len(results.keys()):.2%} ± 0.00",
        ]
    )
    t.align = "l"
    print(t)
